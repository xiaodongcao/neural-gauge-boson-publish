"""Compact training-history plots with site-resolved residual diagnostics."""

from __future__ import annotations

import os
import re
from typing import Any, Dict, Optional, Sequence

import numpy as np

from .utility import ensure_parent_dir


_ACTIVE_TOL = 1.0e-14
_LOG_FLOOR = 1.0e-300
_LOG_CEILING = 1.0e100
_LINEAR_CEILING = 1.0e100
_RESIDUAL_TERM_RE = re.compile(r"^loss_residual_gmm_m(\d+)_n(\d+)$")


def _residual_term_parts(key: str):
    """Return ``(stable_identity, display_label)`` for one physical channel."""

    match = _RESIDUAL_TERM_RE.match(key)
    if match is None:
        return None
    m_power, n_power = match.groups()
    identity = f"m{m_power}_n{n_power}"
    label = (
        "physical trace (0,0)"
        if m_power == "0" and n_power == "0"
        else f"({m_power},{n_power})"
    )
    return identity, label

_LABELS = {
    "loss": "loss",
    "loss_residual_gmm_time": "objective",
    "loss_residual_gmm_raw": "raw residual square",
    "residual_gmm_z_mean": "z mean",
    "residual_gmm_z_max": "z max",
    "residual_gmm_z_worst": "z worst",
    "residual_gmm_radius_mean": "radius mean",
    "residual_gmm_radius_max": "radius max",
    "residual_gmm_radius_worst": "radius worst",
    "residual_gmm_warning_fraction": "warning",
    "residual_gmm_bad_fraction": "bad",
    "loss_pareto_k_raw": "loss",
    "pareto_k_mean": "mean",
    "pareto_k_max": "max",
    "pareto_k_worst": "worst",
    "pareto_k_warning_fraction": "warning",
    "pareto_k_bad_fraction": "bad",
    "loss_gauge": "total",
    "loss_gauge_drift": "drift",
    "loss_gauge_diffusion": "diffusion",
    "loss_ess": "hinge",
    "log_weight_spread_mean": "spread mean",
    "log_weight_spread_max": "spread max",
    "log_weight_spread_total": "spread total",
    "ess_ratio_min": "ESS/N min",
    "ess_ratio_end": "ESS/N end",
    "loss_L2": "L2",
    "grads_norm": "gradient norm",
    "lr": "learning rate",
}

_COLORS = (
    "#1769AA",
    "#D95F59",
    "#2A9D8F",
    "#7B61A8",
    "#D99A2B",
    "#4D6575",
)
_NEUTRAL = "#263238"
_GRID = "#D9E0E4"
_MUTED = "#60727D"
_METRIC_COLORS = {
    # Gauge components remain visually related while being distinguishable.
    "loss_gauge": "#7B61A8",
    "loss_gauge_drift": "#1769AA",
    "loss_gauge_diffusion": "#2A9D8F",
    # Pareto-k uses purple for the aggregate and blue/amber/red for severity.
    "loss_pareto_k_raw": "#7B61A8",
    "pareto_k_mean": "#1769AA",
    "pareto_k_max": "#D99A2B",
    "pareto_k_worst": "#D95F59",
    "pareto_k_warning_fraction": "#D99A2B",
    "pareto_k_bad_fraction": "#D95F59",
}


def _series(history: Dict[str, Any], key: str, size: int) -> Optional[np.ndarray]:
    values = history.get(key)
    if values is None or len(values) != size:
        return None
    array = np.asarray(values, dtype=float)
    if array.ndim != 1 or not np.any(np.isfinite(array)):
        return None
    return array


def _requested(
    key: str,
    values: np.ndarray,
    active_metric_names: Optional[set[str]],
    *,
    always: bool = False,
) -> bool:
    if always:
        return True
    if active_metric_names is not None:
        return key in active_metric_names
    finite = values[np.isfinite(values)]
    return bool(finite.size and np.max(np.abs(finite)) > _ACTIVE_TOL)


def _metric_label(key: str) -> str:
    if key in _LABELS:
        return _LABELS[key]
    residual_parts = _residual_term_parts(key)
    if residual_parts is not None:
        _identity, label = residual_parts
        return label
    match = re.search(r"_m(\d+)_n(\d+)$", key)
    if match:
        return f"({match.group(1)},{match.group(2)})"
    return key.replace("loss_", "").replace("_", " ")


def _metric_color(
    key: str,
    index: int,
    color_map: Optional[Dict[str, str]] = None,
) -> str:
    if color_map is not None and key in color_map:
        return color_map[key]
    if key in _METRIC_COLORS:
        return _METRIC_COLORS[key]
    if key == "loss":
        return _NEUTRAL
    if key.startswith("loss_residual_gmm_m"):
        return _COLORS[index % len(_COLORS)]
    if "gauge" in key:
        return _COLORS[index % len(_COLORS)]
    if "pareto" in key:
        return _COLORS[index % len(_COLORS)]
    return _COLORS[index % len(_COLORS)]


def _plot_panel(
    ax,
    epochs: np.ndarray,
    title: str,
    series: Sequence[tuple[str, str, np.ndarray]],
    *,
    fraction: bool = False,
    color_map: Optional[Dict[str, str]] = None,
    note: Optional[str] = None,
) -> None:
    finite_values = [
        values[np.isfinite(values)]
        for _key, _label, values in series
        if np.any(np.isfinite(values))
    ]
    combined = (
        np.concatenate(finite_values)
        if finite_values
        else np.asarray([], dtype=float)
    )
    finite_positive = combined[combined > 0.0]
    dynamic_range = (
        float(np.max(finite_positive) / np.min(finite_positive))
        if finite_positive.size >= 2
        else 1.0
    )
    use_log = bool(
        not fraction
        and combined.size >= 2
        and np.all(combined > 0.0)
        and dynamic_range >= 20.0
    )
    plotted = []
    for index, (key, label, values) in enumerate(series):
        clean = np.where(np.isfinite(values), values, np.nan)
        if use_log:
            clean = np.where(clean > 0.0, clean, np.nan)
            clean = np.clip(clean, _LOG_FLOOR, _LOG_CEILING)
        else:
            clean = np.clip(clean, -_LINEAR_CEILING, _LINEAR_CEILING)
        line = ax.plot(
            epochs,
            clean,
            color=_metric_color(key, index, color_map),
            linewidth=1.8,
            marker="o" if len(epochs) <= 20 else None,
            markersize=4.0,
            markeredgewidth=0.0,
            label=label,
            zorder=3,
        )[0]
        plotted.append(line)
    if use_log:
        ax.set_yscale("log")
    if fraction:
        ax.set_ylim(-0.03, 1.03)
        from matplotlib.ticker import PercentFormatter

        ax.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
    all_zero = bool(combined.size and np.all(combined == 0.0))
    if all_zero:
        ax.axhline(0.0, color=_MUTED, linewidth=1.0, zorder=1)
        note = note or "0 throughout"
    if len(epochs) == 1:
        center = float(epochs[0])
        ax.set_xlim(center - 0.5, center + 0.5)
        if len(series) == 1 and combined.size and np.isfinite(combined[0]):
            ax.annotate(
                f"{float(combined[0]):.2e}",
                (center, float(combined[0])),
                xytext=(7, 7),
                textcoords="offset points",
                fontsize=8,
                color=_MUTED,
            )
    elif len(epochs) > 1:
        start = float(np.nanmin(epochs))
        stop = float(np.nanmax(epochs))
        padding = max(0.02 * (stop - start), 0.25)
        ax.set_xlim(start - padding, stop + padding)
    if note:
        ax.text(
            0.98,
            0.05,
            note,
            transform=ax.transAxes,
            ha="right",
            va="bottom",
            fontsize=8,
            color=_MUTED,
        )
    ax.set_title(title, loc="left", pad=8, fontweight="semibold")
    ax.set_xlabel("epoch")
    ax.set_ylabel("fraction" if fraction else "value")
    from matplotlib.ticker import MaxNLocator

    ax.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=6))
    ax.grid(True, which="major", color=_GRID, linewidth=0.7, alpha=0.65)
    ax.grid(False, which="minor")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#9EACB4")
    ax.spines["bottom"].set_color("#9EACB4")
    if len(plotted) > 1:
        ax.legend(
            loc="best",
            fontsize=8,
            frameon=False,
            handlelength=1.8,
            borderaxespad=0.3,
        )


def _build_panels(
    history: Dict[str, Any],
    epochs: np.ndarray,
    active_metric_names: Optional[set[str]],
):
    size = len(epochs)
    panels = []
    covered: set[str] = {
        "epoch",
        "epoch_time_sec",
        # The per-window mean is already represented by the time-objective
        # curve and should never appear as a duplicate fallback panel.
        "loss_residual_gmm",
    }

    def add_group(
        priority: int,
        title: str,
        members: Sequence[tuple[str, str]],
        *,
        fraction: bool = False,
        always: bool = False,
        require_nonzero: bool = False,
    ) -> None:
        available = []
        for key, label in members:
            values = _series(history, key, size)
            if values is None:
                continue
            if not _requested(
                key,
                values,
                active_metric_names,
                always=always and key == "loss",
            ):
                continue
            finite = values[np.isfinite(values)]
            if require_nonzero and (
                not finite.size or np.max(np.abs(finite)) <= _ACTIVE_TOL
            ):
                covered.add(key)
                continue
            available.append((key, label, values))
            covered.add(key)
        if available:
            panels.append((priority, title, available, fraction))

    add_group(0, "Training loss", (("loss", "loss"),), always=True)
    def add_monomial_group(
        priority: int,
        title: str,
        pattern: str,
    ) -> None:
        matcher = re.compile(pattern)
        terms = []
        for key in history:
            if not matcher.match(key):
                continue
            values = _series(history, key, size)
            if values is None or not _requested(key, values, active_metric_names):
                continue
            terms.append((key, _metric_label(key), values))
            covered.add(key)
        if terms:
            panels.append((priority, title, terms, False))

    # The projected-residual section is deliberately contiguous and precedes all
    # Pareto, gauge, gradient, and learning-rate diagnostics.
    add_group(
        30,
        "Integrated projected-residual objective",
        (
            ("loss_residual_gmm_time", "objective"),
            ("loss_residual_gmm_raw", "raw residual square"),
        ),
    )
    add_group(
        31,
        "Objective residual cloud: whitened mean",
        (
            ("residual_gmm_z_mean", "mean"),
            ("residual_gmm_z_max", "time-mean max"),
            ("residual_gmm_z_worst", "worst window"),
        ),
    )
    add_group(
        32,
        "Objective residual cloud: Mahalanobis radius",
        (
            ("residual_gmm_radius_mean", "mean"),
            ("residual_gmm_radius_max", "time-mean max"),
            ("residual_gmm_radius_worst", "worst window"),
        ),
    )
    add_group(
        33,
        "Objective residual cloud: flagged fractions",
        (
            ("residual_gmm_warning_fraction", "warning"),
            ("residual_gmm_bad_fraction", "bad"),
        ),
        fraction=True,
        require_nonzero=True,
    )
    residual_terms = []
    for key in history:
        parts = _residual_term_parts(key)
        if parts is None:
            continue
        values = _series(history, key, size)
        if values is None or not _requested(key, values, active_metric_names):
            continue
        identity, label = parts
        residual_terms.append((identity, label, key, values))
        covered.add(key)
    # History insertion order is the scientific channel order: trace first,
    # followed by operator_monomials in exact configuration order.
    for index, (_identity, label, key, values) in enumerate(residual_terms):
        is_trace = label == "physical trace (0,0)"
        title = (
            "Trace equation (0,0): raw channel monitor"
            if is_trace
            else f"{label}: site-averaged raw channel monitor"
        )
        panels.append(
            (
                40 + index,
                title,
                [(key, _metric_label(key), values)],
                False,
            )
        )

    post_base = 100 + len(residual_terms)
    add_group(
        post_base,
        "Gauge penalty",
        (
            ("loss_gauge", "total"),
            ("loss_gauge_drift", "drift"),
            ("loss_gauge_diffusion", "diffusion"),
        ),
    )
    add_group(
        post_base + 5,
        "Weight-entropy spend (nats/window)",
        (
            ("loss_ess", "hinge"),
            ("log_weight_spread_mean", "spread mean"),
            ("log_weight_spread_max", "spread max"),
        ),
    )
    add_group(
        post_base + 6,
        "Complex ESS ratio",
        (
            ("ess_ratio_min", "ESS/N min"),
            ("ess_ratio_end", "ESS/N end"),
        ),
        fraction=True,
    )
    add_group(
        post_base + 10,
        "Pareto-k",
        (
            ("loss_pareto_k_raw", "loss"),
            ("pareto_k_mean", "mean"),
            ("pareto_k_max", "max"),
            ("pareto_k_worst", "worst"),
        ),
    )
    add_group(
        post_base + 11,
        "Pareto-k fractions",
        (
            ("pareto_k_warning_fraction", "warning"),
            ("pareto_k_bad_fraction", "bad"),
        ),
        fraction=True,
    )
    add_monomial_group(
        post_base + 12,
        "Pareto-k channels",
        r"^loss_pareto_k_m\d+_n\d+$",
    )
    add_group(
        post_base + 40,
        "Optimization",
        (
            ("grads_norm", "gradient norm"),
            ("lr", "learning rate"),
            ("loss_L2", "L2"),
        ),
    )

    skip_suffixes = ("_ema_scale", "_normalized")
    skip_fragments = (
        "_site_terms",
        "_covariance_",
        "_terms",
    )
    for key in history:
        if key in covered or key.endswith(skip_suffixes):
            continue
        if any(fragment in key for fragment in skip_fragments):
            continue
        values = _series(history, key, size)
        if values is None or not _requested(key, values, active_metric_names):
            continue
        panels.append(
            (
                post_base + 100 + len(panels),
                _metric_label(key),
                [(key, _metric_label(key), values)],
                "fraction" in key,
            )
        )
        covered.add(key)

    panels.sort(key=lambda panel: panel[0])
    return panels


def _residual_panel_identity(panel):
    _priority, _title, panel_series, _fraction = panel
    if len(panel_series) != 1:
        return None
    return _residual_term_parts(panel_series[0][0])


def _residual_channel_color_map(history: Dict[str, Any]) -> Dict[str, str]:
    """Assign stable colors to trace and configured physical channels."""

    channel_order = []
    for key in history:
        parts = _residual_term_parts(key)
        if parts is None:
            continue
        identity, label = parts
        if label != "physical trace (0,0)" and identity not in channel_order:
            channel_order.append(identity)
    channel_colors = {
        identity: _COLORS[index % len(_COLORS)]
        for index, identity in enumerate(channel_order)
    }
    color_map = {}
    for key in history:
        parts = _residual_term_parts(key)
        if parts is None:
            continue
        identity, label = parts
        color_map[key] = (
            _NEUTRAL
            if label == "physical trace (0,0)"
            else channel_colors[identity]
        )
    return color_map


def _panel_chunks(panels, width: int):
    width = max(1, int(width))
    return [panels[index : index + width] for index in range(0, len(panels), width)]


def _section_columns(panel_count: int, maximum: int = 3) -> int:
    if panel_count <= 1:
        return 1
    if panel_count == 4 and maximum >= 4:
        return 4
    return min(int(maximum), int(panel_count))


def plot_training_history(
    history: Dict[str, Any],
    png_path: str,
    pdf_path: Optional[str] = None,
    active_metric_names: Optional[set[str]] = None,
):
    """Render a sectioned training dashboard; return ``False`` without history."""

    if len(history.get("epoch", [])) == 0:
        return False

    try:
        cache_dir = os.path.join(os.path.dirname(png_path), ".mpl_cache")
        os.makedirs(cache_dir, exist_ok=True)
        os.environ.setdefault("MPLCONFIGDIR", cache_dir)
        os.environ.setdefault("XDG_CACHE_HOME", cache_dir)
        import matplotlib

        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"warning: unable to create training plot ({exc})", flush=True)
        return False

    epochs = np.asarray(history["epoch"], dtype=float)
    panels = _build_panels(history, epochs, active_metric_names)
    if not panels:
        return False

    overview_panels = []
    residual_summary_panels = []
    residual_panels = []
    auxiliary_panels = []
    for panel in panels:
        identity = _residual_panel_identity(panel)
        if identity is not None:
            residual_panels.append(panel)
        elif panel[0] < 30:
            overview_panels.append(panel)
        elif panel[0] < 40:
            residual_summary_panels.append(panel)
        else:
            auxiliary_panels.append(panel)

    flagged_panel_present = any(
        panel[0] == 33 for panel in residual_summary_panels
    )
    color_map = _residual_channel_color_map(history)

    # A row descriptor is either a section heading or a list of panels that
    # share one horizontal grid row. Residual channels retain their configured
    # trace-first order without introducing a synthetic channel hierarchy.
    layout_rows = []

    def add_heading(title: str, *, level: int = 0) -> None:
        layout_rows.append(("heading", title, level))

    def add_panel_rows(section_panels, columns: int) -> None:
        for chunk in _panel_chunks(list(section_panels), columns):
            layout_rows.append(("panels", chunk, None))

    if overview_panels:
        add_heading("Training overview")
        add_panel_rows(
            overview_panels,
            _section_columns(len(overview_panels), maximum=3),
        )
    if residual_summary_panels:
        add_heading("Window-integrated site-resolved projected residual")
        add_panel_rows(
            residual_summary_panels,
            _section_columns(len(residual_summary_panels), maximum=4),
        )
    if residual_panels:
        add_heading(
            "Trace and configured onsite channels · averaged over sites",
            level=1,
        )
        add_panel_rows(residual_panels, min(4, len(residual_panels)))
    if auxiliary_panels:
        add_heading("Optimization and auxiliary diagnostics")
        add_panel_rows(
            auxiliary_panels,
            _section_columns(len(auxiliary_panels), maximum=3),
        )

    plt.rcParams.update(
        {
            "figure.dpi": 160,
            "savefig.dpi": 300,
            "font.size": 9.5,
            "axes.labelsize": 9,
            "axes.titlesize": 10,
            "axes.titleweight": "semibold",
            "axes.grid": False,
            "legend.frameon": False,
            "xtick.direction": "out",
            "ytick.direction": "out",
            "savefig.bbox": "tight",
            "font.family": "DejaVu Sans",
        }
    )

    height_ratios = [
        0.32 if row[0] == "heading" else 2.65
        for row in layout_rows
    ]
    figure_height = 1.15 + sum(height_ratios)
    fig = plt.figure(
        figsize=(15.2, figure_height),
        facecolor="white",
    )
    title_y = 1.0 - 0.12 / figure_height
    subtitle_y = 1.0 - 0.44 / figure_height
    dashboard_top = 1.0 - 0.72 / figure_height
    outer = fig.add_gridspec(
        len(layout_rows),
        1,
        height_ratios=height_ratios,
        left=0.055,
        right=0.985,
        top=dashboard_top,
        bottom=0.035,
        # Consecutive channel rows need enough air for the preceding x-labels
        # and the following monomial titles to remain visually separate.
        hspace=0.58,
    )

    latest_loss = _series(history, "loss", len(epochs))
    latest_loss_text = (
        f" · latest loss {float(latest_loss[-1]):.3e}"
        if latest_loss is not None and np.isfinite(latest_loss[-1])
        else ""
    )
    fig.suptitle(
        "NSGR training diagnostics",
        x=0.055,
        y=title_y,
        ha="left",
        va="top",
        fontsize=16,
        fontweight="semibold",
        color=_NEUTRAL,
    )
    fig.text(
        0.055,
        subtitle_y,
        f"epoch {int(epochs[-1])}{latest_loss_text}",
        ha="left",
        va="top",
        fontsize=9.5,
        color=_MUTED,
    )

    rendered_axes = []
    for row_index, row in enumerate(layout_rows):
        kind, payload, level = row
        if kind == "heading":
            heading_ax = fig.add_subplot(outer[row_index, 0])
            heading_ax.axis("off")
            heading_ax.text(
                0.0,
                0.55,
                payload,
                ha="left",
                va="center",
                fontsize=11 if level == 0 else 10,
                fontweight="semibold",
                color=_NEUTRAL if level == 0 else _MUTED,
            )
            heading_ax.axhline(
                0.02,
                color=_GRID,
                linewidth=0.8,
                xmin=0.0,
                xmax=1.0,
            )
            continue

        row_panels = payload
        nested = outer[row_index, 0].subgridspec(
            1,
            len(row_panels),
            wspace=0.28,
        )
        for panel_index, panel in enumerate(row_panels):
            _priority, title, panel_series, fraction = panel
            identity = _residual_panel_identity(panel)
            if identity is not None:
                _encoded, channel_label = identity
                title = (
                    "Trace equation (0,0) · raw residual square"
                    if channel_label == "physical trace (0,0)"
                    else channel_label
                )
            note = None
            if (
                title == "Objective residual cloud: Mahalanobis radius"
                and not flagged_panel_present
            ):
                note = "no flagged walkers"
            ax = fig.add_subplot(nested[0, panel_index])
            _plot_panel(
                ax,
                epochs,
                title,
                panel_series,
                fraction=fraction,
                color_map=color_map,
                note=note,
            )
            rendered_axes.append(ax)

    if rendered_axes:
        fig.align_ylabels(rendered_axes)
    fig.savefig(ensure_parent_dir(png_path), facecolor=fig.get_facecolor())
    if pdf_path is not None:
        fig.savefig(ensure_parent_dir(pdf_path), facecolor=fig.get_facecolor())
    plt.close(fig)
    return True


__all__ = ["plot_training_history"]
