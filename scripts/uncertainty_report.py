"""Train an uncertainty-aware surrogate ensemble and report its calibration.

Outputs:
    models/ensemble/member_*.npz          trained ensemble members
    models/uncertainty_calibration.json   scale factors + calibration metrics
    docs/surrogate_uncertainty.png        calibration figure
    docs/uncertainty_report.md            human-readable report

Run:
    python scripts/uncertainty_report.py
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def main():
    parser = argparse.ArgumentParser(description="Surrogate uncertainty report")
    parser.add_argument("--members", type=int, default=5)
    parser.add_argument("--samples", type=int, default=6000)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--noise", type=float, default=0.02)
    parser.add_argument("--n-cal", type=int, default=500,
                        help="Calibration-split size (fits the scale factors)")
    parser.add_argument("--n-test", type=int, default=500,
                        help="Test-split size (reports final calibration)")
    args = parser.parse_args()

    from src.ml.dataset import PARAM_NAMES, generate_dataset
    from src.ml.uncertainty import (
        NOMINAL_1SIGMA,
        NOMINAL_2SIGMA,
        calibration_report,
        fit_variance_scaling,
        train_ensemble,
    )

    print(f"Training {args.members}-member ensemble...")
    ensemble, layout = train_ensemble(
        n_members=args.members, n_samples=args.samples,
        noise_frac=args.noise, epochs=args.epochs,
    )

    models_dir = PROJECT_ROOT / "models"
    ensemble.save(models_dir / "ensemble")
    print(f"  saved ensemble to {models_dir / 'ensemble'}")

    # Calibration and test splits must be disjoint, or the reported
    # calibration is circular.
    X_cal, y_cal, _ = generate_dataset(n_samples=args.n_cal,
                                       noise_frac=args.noise, seed=555,
                                       layout=layout)
    X_test, y_test, _ = generate_dataset(n_samples=args.n_test,
                                         noise_frac=args.noise, seed=31337,
                                         layout=layout)

    mu_cal, sd_cal = ensemble.predict_with_std(X_cal)
    mu_test, sd_test = ensemble.predict_with_std(X_test)

    scales = fit_variance_scaling(y_cal, mu_cal, sd_cal)
    before = calibration_report(y_test, mu_test, sd_test)
    after = calibration_report(y_test, mu_test, sd_test * scales)

    print(f"\n  Nominal 1-sigma coverage = {NOMINAL_1SIGMA:.3f}")
    print(f"  {'param':6s} {'scale':>6s} {'cov1 raw':>9s} {'cov1 calib':>11s}")
    for j, name in enumerate(PARAM_NAMES):
        print(f"  {name:6s} {scales[j]:6.2f} "
              f"{before.coverage_1sigma[name]:9.3f} "
              f"{after.coverage_1sigma[name]:11.3f}")

    out = {
        "nominal_1sigma": NOMINAL_1SIGMA,
        "nominal_2sigma": NOMINAL_2SIGMA,
        "param_names": PARAM_NAMES,
        "scale_factors": {n: float(scales[j]) for j, n in enumerate(PARAM_NAMES)},
        "raw": before.as_dict(),
        "calibrated": after.as_dict(),
        "n_members": args.members,
        "n_cal": args.n_cal,
        "n_test": args.n_test,
    }
    with open(models_dir / "uncertainty_calibration.json", "w") as f:
        json.dump(out, f, indent=2)
    print("\n  wrote models/uncertainty_calibration.json")

    try:
        _make_figure(y_test, mu_test, sd_test, scales, before, after,
                     PARAM_NAMES, NOMINAL_1SIGMA,
                     PROJECT_ROOT / "docs" / "surrogate_uncertainty.png")
        print("  wrote docs/surrogate_uncertainty.png")
    except Exception as exc:
        print(f"  (skipped figure: {exc})")

    _write_markdown(out, before, after, scales, PARAM_NAMES,
                    NOMINAL_1SIGMA, NOMINAL_2SIGMA,
                    PROJECT_ROOT / "docs" / "uncertainty_report.md")
    print("  wrote docs/uncertainty_report.md")


def _make_figure(y_test, mu, sd, scales, before, after, names, nominal, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.6))

    # Panel 1: coverage before/after vs nominal.
    ax = axes[0]
    x = np.arange(len(names))
    w = 0.36
    ax.bar(x - w / 2, [before.coverage_1sigma[n] for n in names], w,
           label="raw ensemble spread", color="#D55E00")
    ax.bar(x + w / 2, [after.coverage_1sigma[n] for n in names], w,
           label="after recalibration", color="#009E73")
    ax.axhline(nominal, ls="--", color="#16202b",
               label=f"nominal ({nominal:.2f})")
    ax.set_xticks(x)
    ax.set_xticklabels(names)
    ax.set_ylabel("empirical 1-sigma coverage")
    ax.set_title("Are the error bars honest?")
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, axis="y")

    # Panel 2: Ip residuals with calibrated error bars. Plotting the residual
    # rather than prediction-vs-truth makes the error bars visible: they are
    # about 1 kA against a 20-250 kA axis.
    ax = axes[1]
    j = names.index("Ip")
    order = np.argsort(y_test[:, j])[::max(1, len(y_test) // 60)]
    yt = y_test[order, j] / 1e3
    resid = (mu[order, j] - y_test[order, j]) / 1e3
    ys = (sd[order, j] * scales[j]) / 1e3
    inside = np.abs(resid) <= ys
    ax.errorbar(yt, resid, yerr=ys, fmt="none", lw=1, capsize=2,
                ecolor="#7f8c99", alpha=0.9, zorder=1)
    ax.scatter(yt[inside], resid[inside], s=22, color="#0072B2", zorder=2,
               label="inside 1 sigma")
    ax.scatter(yt[~inside], resid[~inside], s=22, color="#D55E00", zorder=2,
               label="outside 1 sigma")
    ax.axhline(0.0, ls="--", color="#16202b", lw=1)
    ax.set_xlabel("true Ip (kA)")
    ax.set_ylabel("prediction error (kA)")
    ax.set_title("Ip residuals with calibrated 1-sigma bars")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    fig.suptitle("Surrogate uncertainty quantification (deep ensemble)",
                 fontsize=13)
    fig.tight_layout()
    path.parent.mkdir(exist_ok=True)
    fig.savefig(path, dpi=130)
    plt.close(fig)


def _write_markdown(out, before, after, scales, names, nom1, nom2, path):
    lines = [
        "# Surrogate Uncertainty Quantification",
        "",
        "A point estimate of `Ip = 184 kA` is far less useful to an operator",
        "than `Ip = 184 kA +/- 3 kA`. This report adds error bars to the",
        "surrogate and, more importantly, checks whether those error bars are",
        "**honest**.",
        "",
        f"- Method: deep ensemble of **{out['n_members']}** networks, each",
        "  trained on its own data draw and random initialization. The spread",
        "  of member predictions is the uncertainty estimate.",
        f"- Calibration split: {out['n_cal']} samples (fits the scale factors).",
        f"- Test split: {out['n_test']} samples (reports the numbers below).",
        "  The two are disjoint, so the reported calibration is not circular.",
        "",
        "Regenerate with `python scripts/uncertainty_report.py`.",
        "",
        "![Uncertainty](surrogate_uncertainty.png)",
        "",
        "## The finding: raw ensemble spread is not calibrated",
        "",
        "For a well-calibrated Gaussian error bar, the true value should fall",
        f"within 1 sigma about **{nom1:.1%}** of the time and within 2 sigma",
        f"about **{nom2:.1%}** of the time. Raw ensemble spread misses this in",
        "both directions: too wide for some parameters, too narrow for others.",
        "",
        "| Parameter | 1-sigma coverage (raw) | Verdict |",
        "|---|---|---|",
    ]
    for n in names:
        cov = before.coverage_1sigma[n]
        if cov > nom1 + 0.05:
            verdict = "too wide (over-cautious)"
        elif cov < nom1 - 0.05:
            verdict = "too narrow (overconfident)"
        else:
            verdict = "about right"
        lines.append(f"| {n} | {cov:.3f} | {verdict} |")

    lines += [
        "",
        "## The fix: per-parameter variance recalibration",
        "",
        "Each parameter's sigma is multiplied by a single scale factor, fitted",
        "on the held-out calibration split as the quantile of",
        "`|error| / sigma` at the target coverage level. This is a standard",
        "nonparametric variance recalibration.",
        "",
        f"| Parameter | Scale factor | 1-sigma raw | 1-sigma calibrated | "
        f"target {nom1:.3f} |",
        "|---|---|---|---|---|",
    ]
    for j, n in enumerate(names):
        lines.append(
            f"| {n} | {scales[j]:.2f} | {before.coverage_1sigma[n]:.3f} | "
            f"{after.coverage_1sigma[n]:.3f} | |"
        )

    lines += [
        "",
        "Scale factors below 1 shrink over-cautious error bars; factors above",
        "1 widen overconfident ones.",
        "",
        "## Predicted sigma versus realized error",
        "",
        "Coverage alone is not sufficient: a constant error bar can hit the",
        "right coverage while carrying no per-shot information. The",
        "correlation between predicted sigma and realized absolute error shows",
        "whether the estimate actually knows which shots are hard.",
        "",
        "| Parameter | corr(sigma, abs error) | Mean sigma (calibrated) | RMSE |",
        "|---|---|---|---|",
    ]
    for j, n in enumerate(names):
        lines.append(
            f"| {n} | {after.sigma_error_corr[n]:.3f} | "
            f"{after.mean_sigma[n]:.4g} | {after.rmse[n]:.4g} |"
        )

    lines += [
        "",
        "These correlations are positive but modest. The uncertainty estimate",
        "carries real signal about which reconstructions are less reliable, but",
        "it is not a precise per-shot error predictor, and it should not be",
        "presented as one.",
        "",
        "## Scope and caveats",
        "",
        "- Shots are **synthetic**, from the reduced forward model. These",
        "  numbers describe behaviour on that distribution only.",
        "- Ensemble spread captures uncertainty from training randomness and",
        "  data sampling. It does **not** capture systematic error from the",
        "  reduced forward model itself being an approximation.",
        "- Calibration is fit and reported on disjoint splits, but both come",
        "  from the same generator; calibration on real experimental data would",
        "  need to be refit.",
        "",
    ]
    path.parent.mkdir(exist_ok=True)
    path.write_text("\n".join(lines))


if __name__ == "__main__":
    main()
