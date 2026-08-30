"""Train and compare MDLM and AR baselines on character-level Text8.

Example (12 GB GPU):
    python experiments/train_text8.py --device cuda --steps 150000 --batch-size 64 \
      --grad-accum 2 --precision bf16

TensorBoard logs, checkpoints, samples, and ``report.json`` are written below
``runs/text8/<run-name>``. Start TensorBoard with ``tensorboard --logdir runs``.
"""

import argparse
import copy
import json
import random
import sys
import urllib.request
import zipfile
from pathlib import Path

import torch
from torch.nn import functional as F
from torch.utils.tensorboard import SummaryWriter

# Permit ``python experiments/train_text8.py`` from any working directory.
REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from models.ar_transformer import CausalTransformerLM
from models.dit import DiT
from models.mdlm import build_masked_diffusion


TEXT8_URL = "http://mattmahoney.net/dc/text8.zip"
ALPHABET = "abcdefghijklmnopqrstuvwxyz "
CHAR_TO_ID = {char: index for index, char in enumerate(ALPHABET)}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data/text8"))
    parser.add_argument("--run-dir", type=Path, default=Path("runs/text8/default"))
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--steps", type=int, default=150_000)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--grad-accum", type=int, default=2)
    parser.add_argument("--seq-len", type=int, default=256)
    parser.add_argument("--hidden-size", type=int, default=384)
    parser.add_argument("--depth", type=int, default=8)
    parser.add_argument("--num-heads", type=int, default=6)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--ar-hidden-size", type=int, default=None,
                        help="AR width; use 448 with the default MDLM to match parameters.")
    parser.add_argument("--ar-depth", type=int, default=None)
    parser.add_argument("--ar-num-heads", type=int, default=None)
    parser.add_argument("--diffusion-steps", type=int, default=256)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-steps", type=int, default=2_000)
    parser.add_argument("--eval-every", type=int, default=2_000)
    parser.add_argument("--eval-batches", type=int, default=8)
    parser.add_argument("--bpd-steps", type=int, default=64)
    parser.add_argument("--time-bin-eval-batches", type=int, default=2,
                        help="Validation batches per noise-time diagnostic bin.")
    parser.add_argument("--sample-batch-size", type=int, default=4)
    parser.add_argument("--precision", choices=("fp32", "fp16", "bf16"), default="bf16")
    parser.add_argument("--ema-decay", type=float, default=0.9999)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--download-only", action="store_true")
    return parser.parse_args()


def download_text8(data_dir):
    """Download Text8 once and return its UTF-8 text file path."""
    data_dir.mkdir(parents=True, exist_ok=True)
    text_path = data_dir / "text8"
    if text_path.exists():
        return text_path
    zip_path = data_dir / "text8.zip"
    if not zip_path.exists():
        print(f"Downloading {TEXT8_URL} -> {zip_path}")
        urllib.request.urlretrieve(TEXT8_URL, zip_path)
    with zipfile.ZipFile(zip_path) as archive:
        archive.extract("text8", data_dir)
    return text_path


def load_splits(text_path):
    """Encode Text8 and split its 100M characters into 90M/5M/5M tensors."""
    text = text_path.read_text(encoding="utf-8")
    invalid = set(text) - set(ALPHABET)
    if invalid:
        raise ValueError(f"Unexpected Text8 characters: {invalid}")
    encoded = torch.tensor([CHAR_TO_ID[char] for char in text], dtype=torch.long)
    train_end = int(0.9 * len(encoded))
    valid_end = int(0.95 * len(encoded))
    return encoded[:train_end], encoded[train_end:valid_end], encoded[valid_end:]


def random_batch(tokens, batch_size, seq_len, device):
    """Sample non-wrapping integer token blocks with output shape ``(B, L)``."""
    starts = torch.randint(0, len(tokens) - seq_len, (batch_size,))
    indices = starts[:, None] + torch.arange(seq_len)[None, :]
    return tokens[indices].to(device, non_blocking=True)


def decode(tokens):
    """Decode a one-dimensional Text8 tensor, replacing no special tokens."""
    return "".join(ALPHABET[index] for index in tokens.tolist())


def count_parameters(model):
    return sum(parameter.numel() for parameter in model.parameters())


@torch.no_grad()
def update_ema(ema_model, model, decay):
    """Update EMA weights after an optimizer step without tracking gradients."""
    for ema_parameter, parameter in zip(ema_model.parameters(), model.parameters()):
        ema_parameter.lerp_(parameter, 1.0 - decay)
    for ema_buffer, buffer in zip(ema_model.buffers(), model.buffers()):
        ema_buffer.copy_(buffer)


def lr_for_step(step, args):
    """Linear warmup followed by cosine decay to 10% of the peak LR."""
    if step < args.warmup_steps:
        return args.lr * (step + 1) / max(1, args.warmup_steps)
    progress = (step - args.warmup_steps) / max(1, args.steps - args.warmup_steps)
    return args.lr * (0.1 + 0.9 * (1 + torch.cos(torch.tensor(progress * torch.pi)).item()) / 2)


@torch.no_grad()
def evaluate_mdlm(model, diffusion, valid_tokens, args, device):
    """Monte-Carlo validation BPD; returns a scalar in bits per character."""
    model.eval()
    values = []
    for _ in range(args.eval_batches):
        x = random_batch(valid_tokens, args.batch_size, args.seq_len, device)
        values.append(diffusion.calc_bpd_loop(model, x, args.bpd_steps)["total"].mean())
    return torch.stack(values).mean().item()


@torch.no_grad()
def evaluate_ar(model, valid_tokens, args, device):
    """Teacher-forced AR validation cross-entropy in bits per character."""
    model.eval()
    values = []
    for _ in range(args.eval_batches):
        x = random_batch(valid_tokens, args.batch_size, args.seq_len + 1, device)
        logits = model(x[:, :-1])
        values.append(F.cross_entropy(logits.flatten(0, 1), x[:, 1:].flatten()) / torch.log(torch.tensor(2.0, device=device)))
    return torch.stack(values).mean().item()


@torch.no_grad()
def evaluate_mdlm_time_bins(model, diffusion, valid_tokens, args, device):
    """Estimate weighted MDLM BPD in four continuous-time intervals.

    Each value has the same per-position BPD units as the training objective but
    is conditioned on one interval of ``t``. Their relative values diagnose
    whether failure is concentrated in nearly-clean or nearly-all-mask inputs.
    """
    model.eval()
    bins = ((0.0, 0.25), (0.25, 0.5), (0.5, 0.75), (0.75, 1.0))
    results = {}
    for low, high in bins:
        values = []
        for _ in range(args.time_bin_eval_batches):
            x = random_batch(valid_tokens, args.batch_size, args.seq_len, device)
            t = torch.empty(args.batch_size, device=device).uniform_(max(low, diffusion.eps), high)
            values.append(diffusion._nelbo_bpd_terms(model, x, t).mean())
        results[f"{int(low * 100)}_{int(high * 100)}"] = torch.stack(values).mean().item()
    return results


@torch.no_grad()
def sample_and_log(mdlm, diffusion, ar_model, step, writer, run_dir, args, device):
    """Generate MDLM samples at 64/128/256 steps and AR samples, then log NFE."""
    mdlm.eval()
    ar_model.eval()
    lines = [f"step={step}"]
    sampling_nfe = {}
    for sample_steps in (64, 128, 256):
        samples = diffusion.p_sample_loop_strided(
            mdlm, (args.sample_batch_size, args.seq_len), num_steps=sample_steps,
            device=device, greedy_final=True,
        )
        text = "\n---\n".join(decode(row.cpu()) for row in samples)
        writer.add_text(f"samples/mdlm_{sample_steps}", text, step)
        writer.add_scalar(f"sampling/mdlm_{sample_steps}_nfe", diffusion.last_nfe, step)
        sampling_nfe[str(sample_steps)] = {"nfe": diffusion.last_nfe}
        lines.extend((f"MDLM steps={sample_steps}, NFE={diffusion.last_nfe}", text))

    prefix = torch.full((args.sample_batch_size, 1), CHAR_TO_ID[" "], device=device)
    ar_samples = ar_model.generate(prefix, args.seq_len - 1)
    ar_text = "\n---\n".join(decode(row.cpu()) for row in ar_samples)
    writer.add_text("samples/ar", ar_text, step)
    writer.add_scalar("sampling/ar_nfe", ar_model.last_nfe, step)
    lines.extend((f"AR NFE={ar_model.last_nfe} (KV cache enabled)", ar_text))
    (run_dir / f"samples_step_{step:07d}.txt").write_text("\n\n".join(lines))
    return sampling_nfe, ar_model.last_nfe


def main():
    args = parse_args()
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable.")
    if args.precision != "fp32" and not args.device.startswith("cuda"):
        args.precision = "fp32"
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    text_path = download_text8(args.data_dir)
    if args.download_only:
        return
    train_tokens, valid_tokens, _ = load_splits(text_path)
    device = torch.device(args.device)
    args.run_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(args.run_dir)

    vocab_size, mask_id = len(ALPHABET), len(ALPHABET)
    diffusion = build_masked_diffusion(
        K=vocab_size, mask_id=mask_id, V=vocab_size + 1,
        num_timesteps=args.diffusion_steps, bpd_num_steps=args.bpd_steps,
    ).to(device)
    mdlm = DiT((args.seq_len,), vocab_size + 1, args.diffusion_steps,
               hidden_size=args.hidden_size, depth=args.depth, num_heads=args.num_heads,
               dropout=args.dropout).to(device)
    # EMA is used only for validation and sampling; the online model receives gradients.
    mdlm_ema = copy.deepcopy(mdlm).eval()
    for parameter in mdlm_ema.parameters():
        parameter.requires_grad_(False)
    ar_model = CausalTransformerLM(
        vocab_size, args.seq_len,
        hidden_size=args.ar_hidden_size or args.hidden_size,
        depth=args.ar_depth or args.depth,
        num_heads=args.ar_num_heads or args.num_heads,
    ).to(device)
    mdlm_optim = torch.optim.AdamW(mdlm.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    ar_optim = torch.optim.AdamW(ar_model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    amp_dtype = torch.bfloat16 if args.precision == "bf16" else torch.float16
    amp_enabled = args.precision != "fp32"
    scaler = torch.amp.GradScaler("cuda", enabled=args.precision == "fp16")
    print(f"MDLM parameters: {count_parameters(mdlm):,}; AR parameters: {count_parameters(ar_model):,}")

    for step in range(1, args.steps + 1):
        mdlm.train()
        ar_model.train()
        lr = lr_for_step(step - 1, args)
        for optimizer in (mdlm_optim, ar_optim):
            for group in optimizer.param_groups:
                group["lr"] = lr
            optimizer.zero_grad(set_to_none=True)
        mdlm_loss = ar_loss = 0.0
        for _ in range(args.grad_accum):
            x = random_batch(train_tokens, args.batch_size, args.seq_len + 1, device)
            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_enabled):
                t = diffusion.sample_t_timesteps(args.batch_size, device)
                loss_mdlm = diffusion.training_losses(mdlm, x[:, :-1], t).mean()
                logits = ar_model(x[:, :-1])
                loss_ar = F.cross_entropy(logits.flatten(0, 1), x[:, 1:].flatten())
                loss = (loss_mdlm + loss_ar) / args.grad_accum
            scaler.scale(loss).backward()
            mdlm_loss += loss_mdlm.detach().item() / args.grad_accum
            ar_loss += loss_ar.detach().item() / args.grad_accum
        for optimizer in (mdlm_optim, ar_optim):
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(optimizer.param_groups[0]["params"], 1.0)
            scaler.step(optimizer)
        scaler.update()
        # Warm up EMA so early validation is not dominated by the random init.
        ema_decay = min(args.ema_decay, (step + 1) / (step + 10))
        update_ema(mdlm_ema, mdlm, ema_decay)
        writer.add_scalar("train/mdlm_bpd", mdlm_loss, step)
        writer.add_scalar("train/ar_bpc", ar_loss / torch.log(torch.tensor(2.0)).item(), step)
        writer.add_scalar("train/lr", lr, step)

        if step % args.eval_every == 0 or step == args.steps:
            mdlm_bpd = evaluate_mdlm(mdlm_ema, diffusion, valid_tokens, args, device)
            ar_bpc = evaluate_ar(ar_model, valid_tokens, args, device)
            time_bin_bpd = evaluate_mdlm_time_bins(mdlm_ema, diffusion, valid_tokens, args, device)
            writer.add_scalar("validation/mdlm_bpd", mdlm_bpd, step)
            writer.add_scalar("validation/ar_bpc", ar_bpc, step)
            for time_bin, value in time_bin_bpd.items():
                writer.add_scalar(f"validation/mdlm_bpd_t_{time_bin}", value, step)
            sampling_nfe, ar_sampling_nfe = sample_and_log(
                mdlm_ema, diffusion, ar_model, step, writer, args.run_dir, args, device
            )
            report = {
                "step": step, "validation_mdlm_bpd": mdlm_bpd,
                "validation_ar_bpc": ar_bpc,
                "validation_mdlm_bpd_by_t": time_bin_bpd,
                "ema_decay": args.ema_decay,
                "mdlm_parameters": count_parameters(mdlm), "ar_parameters": count_parameters(ar_model),
                # Likelihood/BPD is a property of the trained MDLM and does not
                # change with the reverse-sampling discretization. It is repeated
                # here to make the 64/128/256 comparison table self-contained.
                "sampling": {
                    sample_steps: {**values, "validation_bpd": mdlm_bpd}
                    for sample_steps, values in sampling_nfe.items()
                },
                "ar_sampling_nfe": ar_sampling_nfe,
                "ar_kv_cache": True,
            }
            (args.run_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n")
            torch.save({"step": step, "mdlm": mdlm.state_dict(), "mdlm_ema": mdlm_ema.state_dict(), "ar": ar_model.state_dict(),
                        "args": vars(args)}, args.run_dir / "checkpoint.pt")
            print(json.dumps(report))
    writer.close()


if __name__ == "__main__":
    main()
