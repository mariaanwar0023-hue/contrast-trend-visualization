"""
contrast-trend-visualization
Core module: contrast-based dose–response trend visualization
for repeated-measures mixed-effects model outputs.

This is intended for visualization only. Statistical inference should be
performed in appropriate mixed-effects modeling software (e.g., SPSS, R).
"""

from __future__ import annotations

import os
from typing import Dict, Optional, Sequence, Tuple, Union

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams


Number = Union[int, float]
ArrayLike = Union[Sequence[Number], np.ndarray]


def _as_1d_float(x: ArrayLike, name: str) -> np.ndarray:
    arr = np.asarray(x, dtype=float).reshape(-1)
    if arr.size == 0:
        raise ValueError(f"{name} is empty.")
    if np.any(~np.isfinite(arr)):
        raise ValueError(f"{name} contains non-finite values.")
    return arr


def _validate_lengths(*pairs: Tuple[np.ndarray, str]) -> None:
    lengths = {name: arr.size for arr, name in pairs}
    if len(set(lengths.values())) != 1:
        raise ValueError(f"Length mismatch: {lengths}")


def _normalize_contrast(coeffs: np.ndarray, name: str) -> np.ndarray:
    norm = np.sqrt(np.sum(coeffs ** 2))
    if norm == 0:
        raise ValueError(f"{name} contrast coefficients have zero norm.")
    return coeffs / norm


def plot_contrast_trend(
    *,
    means: ArrayLike,
    ci_lower: ArrayLike,
    ci_upper: ArrayLike,
    doses_original: ArrayLike,
    dose_labels: Sequence[str],
    variable_name: str = "Outcome",
    y_label: str = "Estimated Marginal Mean (95% CI)",
    x_label: str = r"Dose (Log$_{10}$ mg)",
    # Contrast coefficients used for projection (orthogonal polynomial or user-specified)
    lin_coeffs: ArrayLike,
    quad_coeffs: Optional[ArrayLike] = None,
    # Stats dictionaries can come from SPSS/R outputs; used only for textbox text + logic
    lin_stats: Optional[Dict[str, Number]] = None,
    quad_stats: Optional[Dict[str, Number]] = None,
    # Display logic (default mirrors your workflow: show linear, hide quadratic unless requested)
    show_linear: bool = True,
    show_quadratic: bool = False,
    show_full_model: Optional[bool] = None,
    p_significant: float = 0.05,
    p_marginal: float = 0.10,
    # Figure parameters
    figsize: Tuple[Number, Number] = (7, 6),
    dpi: int = 600,
    # Style parameters (kept explicit and stable for publication figures)
    point_color: str = "black",
    linear_color: str = "#FF1493",
    quad_color: str = "#FC0FC0",
    full_color: str = "green",
    # Text box behavior
    textbox_auto_side: bool = True,
    textbox_side: Optional[str] = None,  # "left" or "right" if you want to force
    textbox_text_align: str = "left",    # keep left for readability (your preference)
    textbox_include_ci: bool = True,
    textbox_note_if_nonsig: bool = True,
    # Output saving
    save_path: Optional[str] = None,
    tight_layout: bool = True,
) -> Tuple[plt.Figure, plt.Axes, Dict[str, np.ndarray]]:
    """
    Create a contrast-based trend plot using EMMs (means + 95% CIs) and
    contrast estimates (linear/quadratic) from a mixed-effects model.

    Parameters
    ----------
    means, ci_lower, ci_upper : array-like
        EMMs and confidence bounds (same length).
    doses_original : array-like
        Dose values on the original scale (same length).
    dose_labels : list[str]
        Tick labels to show on the x-axis (same length).
    lin_coeffs, quad_coeffs : array-like
        Contrast coefficient vectors (same length as means).
    lin_stats, quad_stats : dict
        Optional stats for textbox/logic. Expected keys (if provided):
        - estimate, se, t, p, df, ci_lower, ci_upper
        Only 'estimate' is required to draw projected lines.
    show_full_model : bool or None
        If None, full model is shown only if lin p < p_significant and quad p < p_marginal.
        If bool, forces show/hide.
    save_path : str or None
        If provided, saves PNG to this path.

    Returns
    -------
    fig, ax, outputs
        outputs includes: doses_log, yerr_lower, yerr_upper,
        predicted_linear, predicted_quadratic, predicted_full
    """

    # ---- Coerce inputs
    means = _as_1d_float(means, "means")
    ci_lower = _as_1d_float(ci_lower, "ci_lower")
    ci_upper = _as_1d_float(ci_upper, "ci_upper")
    doses_original = _as_1d_float(doses_original, "doses_original")
    lin_coeffs = _as_1d_float(lin_coeffs, "lin_coeffs")

    if len(dose_labels) != means.size:
        raise ValueError(f"dose_labels length ({len(dose_labels)}) must match means length ({means.size}).")

    _validate_lengths(
        (means, "means"),
        (ci_lower, "ci_lower"),
        (ci_upper, "ci_upper"),
        (doses_original, "doses_original"),
        (lin_coeffs, "lin_coeffs"),
    )

    if np.any(ci_lower > means) or np.any(ci_upper < means):
        # Not fatal, but usually indicates swapped bounds or non-95% intervals.
        pass

    if np.any(doses_original <= 0):
        raise ValueError("All doses_original must be > 0 to compute log10 spacing.")

    doses_log = np.log10(doses_original)

    # ---- Error bars
    yerr_lower = means - ci_lower
    yerr_upper = ci_upper - means

    # ---- Normalize contrasts for projection (as in your LINQ)
    lin_normed = _normalize_contrast(lin_coeffs, "lin_coeffs")

    quad_normed = None
    if quad_coeffs is not None:
        quad_coeffs = _as_1d_float(quad_coeffs, "quad_coeffs")
        _validate_lengths((quad_coeffs, "quad_coeffs"), (means, "means"))
        quad_normed = _normalize_contrast(quad_coeffs, "quad_coeffs")

    # ---- Determine estimates needed to draw lines
    def _get_est(stats: Optional[Dict[str, Number]], name: str) -> Optional[float]:
        if stats is None:
            return None
        if "estimate" not in stats:
            raise ValueError(f"{name}_stats provided but missing required key 'estimate'.")
        return float(stats["estimate"])

    lin_est = _get_est(lin_stats, "lin")
    quad_est = _get_est(quad_stats, "quad")

    grand_mean = float(np.mean(means))

    predicted_linear = None
    if show_linear:
        if lin_est is None:
            raise ValueError("show_linear=True requires lin_stats with key 'estimate'.")
        predicted_linear = grand_mean + lin_est * lin_normed

    predicted_quadratic = None
    if show_quadratic:
        if quad_normed is None:
            raise ValueError("show_quadratic=True requires quad_coeffs.")
        if quad_est is None:
            raise ValueError("show_quadratic=True requires quad_stats with key 'estimate'.")
        predicted_quadratic = grand_mean + quad_est * quad_normed

    # Full model logic
    auto_full = False
    if show_full_model is None:
        # Default: full model only if lin p < 0.05 and quad p < 0.10 (your prior pattern)
        lin_p = float(lin_stats.get("p", np.nan)) if lin_stats else np.nan
        quad_p = float(quad_stats.get("p", np.nan)) if quad_stats else np.nan
        auto_full = (np.isfinite(lin_p) and np.isfinite(quad_p) and (lin_p < p_significant) and (quad_p < p_marginal))
        show_full = auto_full
    else:
        show_full = bool(show_full_model)

    predicted_full = None
    if show_full:
        if quad_normed is None or quad_est is None:
            raise ValueError("show_full_model requires quad_coeffs and quad_stats['estimate'].")
        if lin_est is None:
            raise ValueError("show_full_model requires lin_stats['estimate'].")
        predicted_full = grand_mean + lin_est * lin_normed + quad_est * quad_normed

    # ---- Plot styling (stable, publication-oriented)
    rcParams["font.family"] = "sans-serif"
    rcParams["font.sans-serif"] = ["Arial", "DejaVu Sans"]

    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

    # EMM points with CI
    ax.errorbar(
        doses_log,
        means,
        yerr=[yerr_lower, yerr_upper],
        fmt="o",
        color=point_color,
        ecolor=point_color,
        elinewidth=2,
        capsize=5,
        markersize=10,
        label="Estimated Marginal Means (95% CI)",
        zorder=5,
    )

    # Linear
    if predicted_linear is not None:
        ax.plot(
            doses_log,
            predicted_linear,
            linestyle="-",
            marker="s",
            color=linear_color,
            linewidth=2,
            markersize=6,
            label="Linear Trend",
            zorder=3,
        )

    # Quadratic
    if predicted_quadratic is not None:
        ax.plot(
            doses_log,
            predicted_quadratic,
            linestyle=":",
            marker="^",
            color=quad_color,
            linewidth=2,
            markersize=6,
            label="Quadratic Trend",
            zorder=4,
        )

    # Full model
    if predicted_full is not None:
        ax.plot(
            doses_log,
            predicted_full,
            linestyle="--",
            marker=None,
            color=full_color,
            linewidth=2,
            label="Full Model (Linear+Quad)",
            zorder=4,
        )

    # Axes formatting
    ax.set_xticks(doses_log)
    ax.set_xticklabels(dose_labels, fontsize=11, fontweight="bold", color="black")
    ax.set_xlabel(x_label, fontsize=12, fontweight="bold", labelpad=10)
    ax.set_ylabel(y_label, fontsize=12, fontweight="bold")
    ax.set_title(f"{variable_name} Trend Analysis", fontsize=14, fontweight="bold", pad=20)

    ax.grid(True, linestyle=":", alpha=0.5)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.15), fontsize=9, frameon=False, ncol=2)

    # ---- Stats textbox (optional)
    if lin_stats is not None:
        # Decide side for box
        if textbox_side is not None:
            side = textbox_side.lower().strip()
            if side not in {"left", "right"}:
                raise ValueError("textbox_side must be 'left', 'right', or None.")
        elif textbox_auto_side:
            # LINQ-style heuristic based on first vs last CI top
            first_top = means[0] + yerr_upper[0]
            last_top = means[-1] + yerr_upper[-1]
            side = "right" if first_top <= last_top else "left"
        else:
            side = "right"

        box_x = 0.98 if side == "right" else 0.02
        box_ha = "right" if side == "right" else "left"

        lines = []
        lines.append("Contrast Results:")
        lines.append("Linear Trend:")

        # Only add fields if present
        def _fmt(key: str, fmt: str) -> Optional[str]:
            if key not in lin_stats or lin_stats[key] is None:
                return None
            return fmt.format(float(lin_stats[key]))

        est = _fmt("estimate", "  Est = {:.3f}")
        se = _fmt("se", "SE = {:.3f}")
        if est is not None:
            if se is not None:
                lines.append(f"{est}, {se}")
            else:
                lines.append(est)

        tval = _fmt("t", "{:.2f}")
        dfv = _fmt("df", "{:.2f}")
        pv = _fmt("p", "{:.3f}")

        if (tval is not None) and (dfv is not None) and (pv is not None):
            lines.append(f"  t({dfv}) = {tval}, p = {pv}")
        elif pv is not None:
            lines.append(f"  p = {pv}")

        if textbox_include_ci:
            lo = _fmt("ci_lower", "{:.3f}")
            hi = _fmt("ci_upper", "{:.3f}")
            if lo is not None and hi is not None:
                lines.append(f"  95% CI [{lo}, {hi}]")

        if textbox_note_if_nonsig and ("p" in lin_stats) and (lin_stats["p"] is not None):
            if float(lin_stats["p"]) >= p_significant:
                lines.append("  (Line shown for visualization)")

        full_text = "\n".join(lines)

        ax.text(
            box_x,
            0.98,
            full_text,
            transform=ax.transAxes,
            fontsize=9,
            verticalalignment="top",
            horizontalalignment=box_ha,
            multialignment=textbox_text_align,  # keep 'left' for readability
            bbox=dict(boxstyle="round", facecolor="white", edgecolor="gray", alpha=0.9),
        )

    if tight_layout:
        plt.tight_layout()

    outputs = {
        "doses_log": doses_log,
        "yerr_lower": yerr_lower,
        "yerr_upper": yerr_upper,
        "predicted_linear": predicted_linear,
        "predicted_quadratic": predicted_quadratic,
        "predicted_full": predicted_full,
    }

    if save_path:
        # Ensure directory exists
        out_dir = os.path.dirname(save_path)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")

    return fig, ax, outputs
