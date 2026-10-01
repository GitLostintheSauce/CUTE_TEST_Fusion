"""Run robustness validation on the ML surrogate and write a report.

Outputs:
    models/surrogate_validation.json   raw sweep numbers
    docs/surrogate_validation.png      two-panel robustness figure
    docs/validation_report.md          human-readable report

Run:
    python scripts/validate_surrogate.py
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def main():
    parser = argparse.ArgumentParser(description="Validate the CUTE ML surrogate")
    parser.add_argument("--n-eval", type=int, default=400,
                        help="Held-out samples per sweep point")
    parser.add_argument("--samples", type=int, default=8000,
                        help="Training samples if a model must be trained")
    parser.add_argument("--epochs", type=int, default=400)
    parser.add_argument("--noise", type=float, default=0.02,
                        help="Noise fraction the model is trained at")
    parser.add_argument("--dropout-aug", type=float, default=0.15,
                        help="Input-dropout rate for the robust variant")
    args = parser.parse_args()

    from src.ml.dataset import SensorLayout
    from src.ml.mlp import MLPRegressor
    from src.ml.surrogate import train_surrogate
    from src.ml.validation import run_validation

    models_dir = PROJECT_ROOT / "models"
    models_dir.mkdir(exist_ok=True)

    # --- baseline model -----------------------------------------------------
    model_path = models_dir / "surrogate.npz"
    if model_path.exists():
        print(f"Loading baseline surrogate from {model_path}")
        base = MLPRegressor.load(str(model_path))
        layout = SensorLayout.from_config()
    else:
        print("No saved model found; training the baseline first...")
        base, layout, _ = train_surrogate(
            n_samples=args.samples, noise_frac=args.noise, epochs=args.epochs,
        )
        base.save(str(model_path))

    # --- robust (dropout-augmented) model -----------------------------------
    print(f"Training dropout-augmented variant (input_dropout="
          f"{args.dropout_aug})...")
    robust, _, robust_metrics = train_surrogate(
        n_samples=args.samples, noise_frac=args.noise, epochs=args.epochs,
        input_dropout=args.dropout_aug,
    )
    robust.save(str(models_dir / "surrogate_robust.npz"))
    print(f"  robust clean R2 = {robust_metrics.r2_overall:.4f}")

    print(f"\nRunning robustness sweeps ({args.n_eval} samples per point)...")
    rep_base = run_validation(base, layout, n_eval=args.n_eval,
                              train_noise_frac=args.noise)
    rep_robust = run_validation(robust, layout, n_eval=args.n_eval,
                                train_noise_frac=args.noise)

    out = {
        "baseline": rep_base.as_dict(),
        "dropout_augmented": rep_robust.as_dict(),
        "dropout_aug_rate": args.dropout_aug,
    }
    with open(models_dir / "surrogate_validation.json", "w") as f:
        json.dump(out, f, indent=2)
    print("  wrote models/surrogate_validation.json")

    def show(title, pts_b, pts_r):
        print(f"\n  {title} (baseline -> dropout-augmented):")
        for pb, pr in zip(pts_b, pts_r):
            print(f"    {pb.setting * 100:5.1f}%   R2(Ip,R0,Z0) {_r2_well(pb):.4f} -> "
                  f"{_r2_well(pr):.4f}   R2(a) {pb.r2_per_param['a']:7.3f} -> "
                  f"{pr.r2_per_param['a']:7.3f}    Ip MAE "
                  f"{pb.mae_per_param['Ip'] / 1e3:6.2f} -> "
                  f"{pr.mae_per_param['Ip'] / 1e3:6.2f} kA")

    show("Noise robustness", rep_base.noise_sweep, rep_robust.noise_sweep)
    show("Sensor dropout", rep_base.dropout_sweep, rep_robust.dropout_sweep)

    try:
        _make_figure(rep_base, rep_robust, layout.n_sensors,
                     PROJECT_ROOT / "docs" / "surrogate_validation.png")
        print("\n  wrote docs/surrogate_validation.png")
    except Exception as exc:
        print(f"\n  (skipped figure: {exc})")

    _write_markdown(rep_base, rep_robust, args.dropout_aug, layout.n_sensors,
                    PROJECT_ROOT / "docs" / "validation_report.md")
    print("  wrote docs/validation_report.md")


WELL_CONSTRAINED = ("Ip", "R0", "Z0")


def _r2_well(point) -> float:
    """Mean held-out R2 over Ip, R0 and Z0.

    Minor radius a is reported on its own. The CUTE sensors all sit outside
    the vessel, where a changes the readings by only a few percent, so its R2
    is low and can go strongly negative under noise; averaged in, it would
    hide how the three well-constrained parameters behave.
    """
    return sum(point.r2_per_param[k] for k in WELL_CONSTRAINED) / len(WELL_CONSTRAINED)


def _at(points, setting: float):
    return next(p for p in points if abs(p.setting - setting) < 1e-9)


def _make_figure(rep_base, rep_robust, n_sensors: int, path: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    ax = axes[0]
    xs = [p.setting * 100 for p in rep_base.noise_sweep]
    ax.plot(xs, [_r2_well(p) for p in rep_base.noise_sweep], "o-",
            color="#0072B2", lw=2, label="baseline")
    ax.plot(xs, [_r2_well(p) for p in rep_robust.noise_sweep], "^--",
            color="#009E73", lw=2, label="dropout-augmented")
    ax.axvline(rep_base.train_noise_frac * 100, ls=":", color="#7f8c99",
               label=f"trained at {rep_base.train_noise_frac * 100:.0f}% noise")
    ax.set_xlabel("sensor noise (% of signal scale)")
    ax.set_ylabel("$R^2$, mean of Ip, R0, Z0")
    ax.set_title("Noise robustness")
    ax.set_ylim(0, 1.02)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=9)

    ax = axes[1]
    xs = [p.setting * 100 for p in rep_base.dropout_sweep]
    ax.plot(xs, [_r2_well(p) for p in rep_base.dropout_sweep], "s-",
            color="#D55E00", lw=2, label="baseline")
    ax.plot(xs, [_r2_well(p) for p in rep_robust.dropout_sweep], "^--",
            color="#009E73", lw=2, label="dropout-augmented")
    ax.set_xlabel(f"dead sensors (% of {n_sensors} channels)")
    ax.set_ylabel("$R^2$, mean of Ip, R0, Z0")
    ax.set_title("Sensor-dropout robustness")
    ax.set_ylim(0, 1.02)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=9)

    fig.suptitle("ML surrogate robustness (held-out synthetic shots; "
                 "minor radius a reported separately)",
                 fontsize=13)
    fig.tight_layout()
    path.parent.mkdir(exist_ok=True)
    fig.savefig(path, dpi=130)
    plt.close(fig)


def _rows(points):
    out = []
    for p in points:
        out.append(
            f"| {p.setting * 100:.0f}% | {_r2_well(p):.4f} | "
            f"{p.r2_per_param['a']:.3f} | "
            f"{p.mae_per_param['Ip'] / 1e3:.2f} | "
            f"{p.mae_per_param['R0'] * 1e3:.2f} | "
            f"{p.mae_per_param['Z0'] * 1e3:.2f} | "
            f"{p.mae_per_param['a'] * 1e3:.2f} |"
        )
    return out


HEADER = ("| Setting | R2 (Ip, R0, Z0) | R2 (a) | Ip MAE (kA) | R0 MAE (mm) | "
          "Z0 MAE (mm) | a MAE (mm) |")
DIVIDER = "|---|---|---|---|---|---|---|"


def _write_markdown(rep_base, rep_robust, dropout_aug, n_sensors: int, path: Path):
    noise_10 = _at(rep_base.noise_sweep, 0.10)
    clean_b = _at(rep_base.dropout_sweep, 0.0)
    dead_b = _at(rep_base.dropout_sweep, 0.20)
    dead_r = _at(rep_robust.dropout_sweep, 0.20)
    clean_r = _at(rep_robust.dropout_sweep, 0.0)
    a_clean = _at(rep_base.noise_sweep, 0.0).r2_per_param["a"]
    lines = [
        "# ML Surrogate Validation Report",
        "",
        "Robustness of the neural-network equilibrium-reconstruction surrogate,",
        "measured on held-out synthetic shots from the reduced forward model.",
        "",
        f"- Evaluation samples per sweep point: **{rep_base.n_eval}**",
        f"- Models trained at sensor noise: "
        f"**{rep_base.train_noise_frac * 100:.0f}%**",
        f"- Two variants compared: the **baseline** surrogate, and a "
        f"**dropout-augmented** variant trained with {dropout_aug * 100:.0f}% "
        "of input channels randomly masked each batch.",
        "",
        "All numbers are regenerated by `scripts/validate_surrogate.py`.",
        "",
        "![Robustness](surrogate_validation.png)",
        "",
        "## Headline findings",
        "",
        "R2 is reported as the mean over Ip, R0 and Z0, with minor radius a in",
        "its own column. Finding 4 explains why a is kept apart.",
        "",
        f"1. **Noise tolerance is strong.** At 10% sensor noise, "
        f"{0.10 / rep_base.train_noise_frac:.0f} times the",
        f"   training noise, the baseline keeps R2 {_r2_well(noise_10):.2f} on Ip, R0, Z0",
        f"   (Ip alone {noise_10.r2_per_param['Ip']:.3f}).",
        "2. **Sensor failure was the real weakness.** The baseline degrades",
        "   sharply when channels go dead, because it never saw dead channels",
        f"   during training: R2 {_r2_well(clean_b):.2f} with every sensor, "
        f"{_r2_well(dead_b):.2f} with 20% dead.",
        "3. **Dropout augmentation fixes it.** Training with randomly masked",
        f"   inputs holds R2 at {_r2_well(dead_r):.2f} with 20% of sensors dead, against",
        f"   {_r2_well(clean_r):.2f} with none, at a small cost in clean-signal accuracy.",
        "   Both models are shipped: `models/surrogate.npz` (baseline) and",
        "   `models/surrogate_robust.npz` (dropout-augmented).",
        f"4. **Minor radius a is barely constrained** (clean R2 {a_clean:.2f}). Every",
        "   CUTE sensor is outside the vessel, and from there a wider current",
        "   channel looks almost exactly like a thin one sitting slightly further",
        "   out: widening it from 12 to 16 cm is copied by moving R0 out about",
        "   3 mm, leaving less than one noise-width that only a can explain",
        "   (notebooks/05_what_the_sensors_can_see.ipynb, docs/primer.md Part 10).",
        "   Under more noise or dead channels its R2 goes negative, which means",
        "   worse than always guessing the average. This is a limit of the sensor",
        "   layout and the reduced model, not of the network.",
        "",
        "## 1. Noise robustness",
        "",
        "How accuracy degrades as diagnostic measurement noise grows.",
        "",
        "### Baseline",
        "",
        HEADER,
        DIVIDER,
    ]
    lines += _rows(rep_base.noise_sweep)
    lines += ["", "### Dropout-augmented", "", HEADER, DIVIDER]
    lines += _rows(rep_robust.noise_sweep)

    lines += [
        "",
        "## 2. Sensor-dropout robustness",
        "",
        "Magnetic probes fail. Dead channels are simulated at inference time by",
        f"mean-imputing a random subset of the {n_sensors} sensors, independently per",
        "shot, so each shot loses a different set of probes.",
        "",
        "### Baseline (no dropout augmentation)",
        "",
        HEADER,
        DIVIDER,
    ]
    lines += _rows(rep_base.dropout_sweep)
    lines += ["", "### Dropout-augmented", "", HEADER, DIVIDER]
    lines += _rows(rep_robust.dropout_sweep)

    lines += [
        "",
        "## Scope and caveats",
        "",
        f"- Sensor positions are CUTE's real diagnostic layout ({n_sensors} channels,",
        "  `config/cute_diagnostics.json`); the signals they read are synthetic.",
        "- Shots are **synthetic**, generated by the reduced forward model",
        "  (rigid current disk, analytic circular-loop Green's functions).",
        "  These are not experimental CUTE measurements.",
        "- The reduced forward model is validated to machine precision against",
        "  direct Biot-Savart quadrature and the exact on-axis field, so the",
        "  magnetics underlying these numbers are trustworthy, but it is not a",
        "  full free-boundary Grad-Shafranov solve.",
        "- Predicted parameters are Ip, R0, Z0, and a. q95, beta_pol, and li are",
        "  not predicted, because in this synthetic data they are computed from",
        "  Ip by formula and predicting them would be circular.",
        "- The dropout-augmented model can score slightly higher at moderate",
        "  dropout than at zero dropout, because it is optimized for the masked",
        "  regime it was trained in. Differences of a few hundredths in R2 are",
        "  within run-to-run variation and should not be over-interpreted.",
        "",
    ]
    path.parent.mkdir(exist_ok=True)
    path.write_text("\n".join(lines))


if __name__ == "__main__":
    main()
