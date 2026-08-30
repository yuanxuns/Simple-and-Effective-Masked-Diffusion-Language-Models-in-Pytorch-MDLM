#  Simple and Effective Masked Diffusion Language Models (MDLM)

## Text8 experiment

`experiments/train_text8.py` trains a character-level MDLM and a matched-size
causal Transformer (AR) baseline on Text8. The script downloads and splits the
data 90%/5%/5%; it uses 27 character tokens plus one MDLM mask token. The
recommended AR width of 448 (7 heads) closely matches the default MDLM's
parameter count (about 19M each); all parameter counts are included in the
report.

On a 12 GB GPU, run:

```bash
python experiments/train_text8.py --device cuda --steps 150000 \
  --batch-size 64 --grad-accum 2 --precision bf16 \
  --ar-hidden-size 448 --ar-num-heads 7
tensorboard --logdir runs
python experiments/summarize_text8.py --run-dir runs/text8/default
```

Outputs are written to `runs/text8/default/`:

- TensorBoard: training loss, validation MDLM BPD, and validation AR BPC;
- `samples_step_*.txt`: MDLM samples for 64/128/256 reverse steps and AR samples;
- `report.json`: validation metric, parameter counts, and sampling NFE;
- `learning_curves.png`, `sampling_nfe.png`, and `summary.md`: static report;
- `checkpoint.pt`: both model weights and training configuration.

The validation BPD is a likelihood estimate for the MDLM, so it does not vary
with the sampling-step count. The 64/128/256 comparison measures generated text
and NFE; `report.json` repeats the same validation BPD in each row to make this
explicit. The AR comparison uses teacher-forced BPC and requires 255 sequential
model calls for a 256-character sample, but now caches the K/V tensors of every
layer: after the one-token prefix prefill, each call processes only one new token.
MDLM needs 64, 128, or 256 full-sequence calls respectively.

## References

https://arxiv.org/pdf/2406.07524
