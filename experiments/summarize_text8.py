"""Create static plots and a Markdown summary from a Text8 experiment run.

Example:
    python experiments/summarize_text8.py --run-dir runs/text8/default
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
from tensorboard.backend.event_processing import event_accumulator


def read_scalars(event_path):
    """Read TensorBoard scalar events as ``{tag: [(step, value), ...]}``."""
    accumulator = event_accumulator.EventAccumulator(str(event_path))
    accumulator.Reload()
    return {
        tag: [(event.step, event.value) for event in accumulator.Scalars(tag)]
        for tag in accumulator.Tags().get("scalars", [])
    }


def plot_series(axis, series, tag, label, **kwargs):
    """Plot a scalar series when it is present in the event file."""
    values = series.get(tag, [])
    if values:
        steps, metrics = zip(*values)
        axis.plot(steps, metrics, label=label, **kwargs)


def save_learning_curves(series, output_path):
    """Save train/validation BPD-BPC curves to ``output_path``."""
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
    plot_series(axes[0], series, "train/mdlm_bpd", "MDLM BPD")
    plot_series(axes[0], series, "train/ar_bpc", "AR BPC")
    axes[0].set(title="Training objective", xlabel="optimizer step", ylabel="bits / character")
    axes[0].legend()
    axes[0].grid(alpha=0.25)

    plot_series(axes[1], series, "validation/mdlm_bpd", "MDLM validation BPD", marker="o")
    plot_series(axes[1], series, "validation/ar_bpc", "AR validation BPC", marker="o")
    axes[1].set(title="Validation comparison", xlabel="optimizer step", ylabel="bits / character")
    axes[1].legend()
    axes[1].grid(alpha=0.25)
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def save_nfe_plot(report, output_path):
    """Save a bar chart comparing MDLM sampling NFE against AR NFE."""
    labels = [f"MDLM\n{steps} steps" for steps in report["sampling"]]
    nfe = [values["nfe"] for values in report["sampling"].values()]
    labels.append("AR\nautoregressive sample")
    nfe.append(report["ar_sampling_nfe"])
    figure, axis = plt.subplots(figsize=(7, 4.5), constrained_layout=True)
    bars = axis.bar(labels, nfe, color=["#4C78A8", "#4C78A8", "#4C78A8", "#F58518"])
    axis.set(title="Sequential model evaluations for one sample", ylabel="NFE")
    axis.bar_label(bars, padding=3)
    axis.grid(axis="y", alpha=0.25)
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def latest_event_file(run_dir):
    """Return the newest TensorBoard event file in a run directory."""
    events = sorted(run_dir.glob("events.out.tfevents.*"), key=lambda path: path.stat().st_mtime)
    if not events:
        raise FileNotFoundError(f"No TensorBoard event file found in {run_dir}.")
    return events[-1]


def write_summary(report, output_path):
    """Write a compact Markdown results table; samples remain in ``samples_step_*.txt``."""
    rows = [
        "# Text8 experiment summary",
        "",
        f"- Optimizer step: {report['step']}",
        f"- MDLM parameters: {report['mdlm_parameters']:,}",
        f"- AR parameters: {report['ar_parameters']:,}",
        f"- Validation MDLM BPD: {report['validation_mdlm_bpd']:.4f}",
        f"- Validation AR BPC: {report['validation_ar_bpc']:.4f}",
        "",
        "## Sampling comparison",
        "",
        "| Model / reverse steps | Validation BPD | NFE |",
        "| --- | ---: | ---: |",
    ]
    for steps, values in report["sampling"].items():
        rows.append(f"| MDLM / {steps} | {values['validation_bpd']:.4f} | {values['nfe']} |")
    rows.append(f"| AR / autoregressive sample | — | {report['ar_sampling_nfe']} |")
    rows.extend((
        "",
        "BPD is a likelihood metric of the trained MDLM, not a metric of an individual",
        "sampling discretization; it is therefore identical in the three MDLM rows.",
        "Compare the corresponding text in `samples_step_*.txt` to judge the quality/NFE trade-off.",
    ))
    output_path.write_text("\n".join(rows) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    report = json.loads((args.run_dir / "report.json").read_text())
    series = read_scalars(latest_event_file(args.run_dir))
    save_learning_curves(series, args.run_dir / "learning_curves.png")
    save_nfe_plot(report, args.run_dir / "sampling_nfe.png")
    write_summary(report, args.run_dir / "summary.md")
    print(f"Wrote plots and summary to {args.run_dir}")


if __name__ == "__main__":
    main()
