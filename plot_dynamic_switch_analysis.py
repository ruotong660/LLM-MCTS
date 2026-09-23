from pathlib import Path
import argparse
import html

import numpy as np
import pandas as pd


DEFAULT_ACTIONS_PATH = Path("results/llm_cc_mcts_dynamic_tune_c_real_actions_debug.csv")
DEFAULT_PRICING_PATH = Path(
    "data/datasets/citylearn_challenge_2022_phase_1/pricing.csv"
)
DEFAULT_CARBON_PATH = Path(
    "data/datasets/citylearn_challenge_2022_phase_1/carbon_intensity.csv"
)
DEFAULT_OUTPUT_DIR = Path("results/figures")


COLORS = {
    "action": "#2563eb",
    "prior": "#d97706",
    "price": "#0891b2",
    "carbon": "#16a34a",
    "price_weight": "#7c3aed",
    "carbon_weight": "#16a34a",
    "peak_penalty": "#dc2626",
    "grid": "#e5e7eb",
    "axis": "#111827",
    "text": "#000000",
    "muted": "#000000",
}

SCENARIO_LABELS = {
    "cost_priority": "Cost priority",
    "low_carbon_priority": "Low-carbon",
    "peak_aware": "Peak-aware",
}

SCENARIO_COLORS = {
    "cost_priority": "#fde68a",
    "low_carbon_priority": "#bbf7d0",
    "peak_aware": "#fecaca",
    "n/a": "#e5e7eb",
}

COMBINED_WIDTH = 1120
COMBINED_HEIGHT = 650
SINGLE_WIDTH = 560
SINGLE_HEIGHT = 710


def load_first_numeric_column(csv_path):
    df = pd.read_csv(csv_path)
    numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()

    if len(numeric_columns) == 0:
        raise ValueError(f"{csv_path} has no numeric columns.")

    return df[numeric_columns[0]].astype(float).to_numpy()


def normalize_array(values):
    values = np.asarray(values, dtype=float)
    min_value = np.nanmin(values)
    max_value = np.nanmax(values)

    if abs(max_value - min_value) < 1e-9:
        return np.zeros_like(values)

    return (values - min_value) / (max_value - min_value)


def enrich_actions(actions, pricing_norm, carbon_norm):
    actions = actions.copy()
    idx = actions["time_step"].astype(int).to_numpy() % len(pricing_norm)
    actions["pricing_norm"] = pricing_norm[idx]
    actions["carbon_norm"] = carbon_norm[idx]
    actions["relative_hour"] = 0
    return actions


def pct(series, predicate):
    return float(predicate(series).mean())


def summarize_switch_windows(actions, switch_steps, window_hours):
    rows = []

    for switch_step in switch_steps:
        windows = [
            ("before", switch_step - window_hours, switch_step - 1),
            ("after", switch_step, switch_step + window_hours - 1),
        ]

        for window_name, start_step, end_step in windows:
            window = actions[
                (actions["time_step"] >= start_step)
                & (actions["time_step"] <= end_step)
            ].copy()

            if len(window) == 0:
                continue

            scenario_counts = window["semantic_scenario_key"].value_counts()
            scenario = scenario_counts.index[0]

            rows.append(
                {
                    "switch_step": switch_step,
                    "window": window_name,
                    "start_step": int(window["time_step"].min()),
                    "end_step": int(window["time_step"].max()),
                    "steps": len(window),
                    "dominant_scenario": scenario,
                    "mean_action": float(window["action_value"].mean()),
                    "charge_frac": pct(window["action_value"], lambda s: s > 0),
                    "discharge_frac": pct(window["action_value"], lambda s: s < 0),
                    "zero_frac": pct(window["action_value"], lambda s: s == 0),
                    "mean_prior_action": float(window["prior_action"].mean()),
                    "mean_score": float(window["score"].mean()),
                    "mean_pricing_norm": float(window["pricing_norm"].mean()),
                    "mean_carbon_norm": float(window["carbon_norm"].mean()),
                    "mean_planning_time_seconds": float(
                        window["planning_time_seconds"].mean()
                    ),
                }
            )

    return pd.DataFrame(rows)


def scale_x(x, x_min, x_max, plot_x, plot_w):
    if x_max == x_min:
        return plot_x

    return plot_x + (x - x_min) / (x_max - x_min) * plot_w


def scale_y(y, y_min, y_max, plot_y, plot_h):
    if y_max == y_min:
        return plot_y + plot_h / 2

    y = max(y_min, min(y_max, y))
    return plot_y + plot_h - (y - y_min) / (y_max - y_min) * plot_h


def polyline_points(df, x_column, y_column, x_min, x_max, y_min, y_max, geom):
    plot_x, plot_y, plot_w, plot_h = geom
    points = []

    for _, row in df.iterrows():
        x = scale_x(row[x_column], x_min, x_max, plot_x, plot_w)
        y = scale_y(row[y_column], y_min, y_max, plot_y, plot_h)
        points.append(f"{x:.1f},{y:.1f}")

    return " ".join(points)


def add_triangle(svg, x, y, size=8, fill="#111827"):
    points = f"{x:.1f},{y:.1f} {x - size:.1f},{y - size:.1f} {x + size:.1f},{y - size:.1f}"
    svg.append(f'<polygon points="{points}" fill="{fill}"/>')


def add_text(svg, x, y, text, size=13, weight="normal", fill="#000000", anchor="start"):
    size = max(size, 24)
    safe_text = html.escape(str(text))
    svg.append(
        f'<text x="{x}" y="{y}" font-size="{size}" font-weight="{weight}" '
        f'fill="{fill}" text-anchor="{anchor}" '
        f'font-family="Times New Roman, Times, serif">{safe_text}</text>'
    )


def add_rect(svg, x, y, w, h, fill, stroke="none", radius=0, opacity=1.0):
    svg.append(
        f'<rect x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}" '
        f'rx="{radius}" ry="{radius}" fill="{fill}" stroke="{stroke}" '
        f'opacity="{opacity}"/>'
    )


def add_line(svg, x1, y1, x2, y2, stroke, width=1.0, dash=None):
    dash_attr = f' stroke-dasharray="{dash}"' if dash else ""
    svg.append(
        f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" '
        f'stroke="{stroke}" stroke-width="{width}"{dash_attr}/>'
    )


def add_polyline(svg, points, color, width=2.5, dash=None):
    dash_attr = f' stroke-dasharray="{dash}"' if dash else ""
    svg.append(
        f'<polyline points="{points}" fill="none" stroke="{color}" '
        f'stroke-width="{width}" stroke-linejoin="round" '
        f'stroke-linecap="round"{dash_attr}/>'
    )


def scenario_label(key):
    return SCENARIO_LABELS.get(str(key), str(key).replace("_", " "))


def dominant_scenario(series):
    if len(series) == 0:
        return "n/a"

    return series.mode().iloc[0]


def add_series_panel(
    svg,
    df,
    plot_x,
    plot_y,
    plot_w,
    plot_h,
    y_min,
    y_max,
    y_label,
    lines,
    show_x_labels=False,
    show_y_label=True,
):
    geom = (plot_x, plot_y, plot_w, plot_h)
    x_min = float(df["relative_hour"].min())
    x_max = float(df["relative_hour"].max())

    add_rect(svg, plot_x, plot_y, plot_w, plot_h, "#ffffff", stroke="#cbd5e1")

    for frac in [0.25, 0.5, 0.75]:
        y = plot_y + plot_h * frac
        add_line(svg, plot_x, y, plot_x + plot_w, y, COLORS["grid"])

    tick_values = [-48, -24, 0, 24]
    for x_value in tick_values:
        if x_min <= x_value <= x_max:
            x = scale_x(x_value, x_min, x_max, plot_x, plot_w)
            if x_value == 0:
                add_line(svg, x, plot_y, x, plot_y + plot_h, COLORS["axis"], width=1.4, dash="4 4")
            else:
                add_line(svg, x, plot_y, x, plot_y + plot_h, COLORS["grid"])

            if show_x_labels:
                add_text(svg, x, plot_y + plot_h + 23, str(x_value), 18, fill=COLORS["muted"], anchor="middle")

    zero_y = scale_y(0.0, y_min, y_max, plot_y, plot_h)
    if y_min < 0.0 < y_max:
        add_line(svg, plot_x, zero_y, plot_x + plot_w, zero_y, "#94a3b8", width=1.2)

    add_text(svg, plot_x - 7, plot_y + 16, f"{y_max:g}", 18, fill=COLORS["muted"], anchor="end")
    add_text(svg, plot_x - 7, plot_y + plot_h - 3, f"{y_min:g}", 18, fill=COLORS["muted"], anchor="end")

    if show_y_label:
        label_x, label_y = plot_x - 65, plot_y + plot_h / 2
        svg.append(f'<g transform="rotate(-90 {label_x} {label_y})">')
        add_text(svg, label_x, label_y, y_label, 24, fill=COLORS["text"], anchor="middle")
        svg.append('</g>')

    for _, column, color, dash in lines:
        points = polyline_points(df, "relative_hour", column, x_min, x_max, y_min, y_max, geom)
        add_polyline(svg, points, color, width=2.4, dash=dash)


def add_stage_band(svg, df, plot_x, plot_y, plot_w, band_h, switch_end_y):
    x_min = float(df["relative_hour"].min())
    x_max = float(df["relative_hour"].max())
    switch_x = scale_x(0, x_min, x_max, plot_x, plot_w)

    before_key = dominant_scenario(df[df["relative_hour"] < 0]["semantic_scenario_key"])
    after_key = dominant_scenario(df[df["relative_hour"] >= 0]["semantic_scenario_key"])
    before_color = SCENARIO_COLORS.get(before_key, SCENARIO_COLORS["n/a"])
    after_color = SCENARIO_COLORS.get(after_key, SCENARIO_COLORS["n/a"])

    add_rect(svg, plot_x, plot_y, switch_x - plot_x, band_h, before_color, stroke="#d1d5db")
    add_rect(svg, switch_x, plot_y, plot_x + plot_w - switch_x, band_h, after_color, stroke="#d1d5db")
    add_line(svg, switch_x, plot_y - 4, switch_x, switch_end_y, COLORS["axis"], width=1.4, dash="4 4")

    add_text(svg, (plot_x + switch_x) / 2, plot_y + 24, scenario_label(before_key), 24, weight="bold", fill=COLORS["text"], anchor="middle")
    add_text(svg, (switch_x + plot_x + plot_w) / 2, plot_y + 24, scenario_label(after_key), 24, weight="bold", fill=COLORS["text"], anchor="middle")


def add_switch_column(svg, actions, switch_step, window_hours, column_x, column_y, column_w, title):
    start_step = switch_step - window_hours
    end_step = switch_step + window_hours - 1
    df = actions[
        (actions["time_step"] >= start_step) & (actions["time_step"] <= end_step)
    ].copy()

    if len(df) == 0:
        raise ValueError(f"No data found around switch step {switch_step}.")

    df["relative_hour"] = df["time_step"] - switch_step
    df["peak_penalty_scaled"] = df["peak_charge_penalty_weight"] / 2.0

    add_text(svg, column_x + column_w / 2, column_y, title, 21, weight="bold", fill=COLORS["text"], anchor="middle")

    band_y = column_y + 20
    panel_x = column_x + 76
    panel_w = column_w - 84
    panel_h = 94
    row_gap = 44

    action_y = band_y + 42
    signal_y = action_y + panel_h + row_gap
    param_y = signal_y + panel_h + row_gap

    add_stage_band(svg, df, panel_x, band_y, panel_w, 32, param_y + panel_h)

    add_series_panel(
        svg,
        df,
        panel_x,
        action_y,
        panel_w,
        panel_h,
        -0.12,
        0.12,
        "Action",
        [
            ("Executed action", "action_value", COLORS["action"], None),
            ("Semantic prior", "prior_action", COLORS["prior"], "6 4"),
        ],
    )
    add_text(svg, panel_x + panel_w, action_y + panel_h + 25, "+ charge / - discharge", 18, fill=COLORS["muted"], anchor="end")

    add_series_panel(
        svg,
        df,
        panel_x,
        signal_y,
        panel_w,
        panel_h,
        0.0,
        1.0,
        "Signals",
        [
            ("Price", "pricing_norm", COLORS["price"], None),
            ("Carbon", "carbon_norm", COLORS["carbon"], None),
        ],
    )

    add_series_panel(
        svg,
        df,
        panel_x,
        param_y,
        panel_w,
        panel_h,
        0.0,
        1.0,
        "Weights",
        [
            ("Cost weight", "price_weight", COLORS["price_weight"], None),
            ("Carbon weight", "carbon_weight", COLORS["carbon_weight"], None),
            ("Peak penalty / 2", "peak_penalty_scaled", COLORS["peak_penalty"], "5 4"),
        ],
        show_x_labels=True,
    )

    add_text(svg, panel_x + panel_w / 2, param_y + panel_h + 49, "Hours relative to switch", 19, fill=COLORS["muted"], anchor="middle")

    before = df[df["relative_hour"] < 0]
    after = df[df["relative_hour"] >= 0]
    summary_y = param_y + panel_h + 84
    summary = (
        f"Mean action {before['action_value'].mean():+.3f} -> {after['action_value'].mean():+.3f}"
    )
    add_text(svg, column_x + column_w / 2, summary_y, summary, 18, fill=COLORS["muted"], anchor="middle")
    idle = f"Idle {100 * (before['action_value'] == 0).mean():.0f}% -> {100 * (after['action_value'] == 0).mean():.0f}%"
    add_text(svg, column_x + column_w / 2, summary_y + 28, idle, 24, anchor="middle")


def add_global_legend(svg, x, y, max_width):
    legend_items = [
        ("Executed action", COLORS["action"], None),
        ("Semantic prior", COLORS["prior"], "6 4"),
        ("Price", COLORS["price"], None),
        ("Carbon", COLORS["carbon"], None),
        ("Cost weight", COLORS["price_weight"], None),
        ("Carbon weight", COLORS["carbon_weight"], None),
        ("Peak penalty / 2", COLORS["peak_penalty"], "5 4"),
    ]
    current_x = x
    current_y = y

    for index, (label, color, dash) in enumerate(legend_items):
        item_width = 35 + len(label) * 12.5 + 24
        if current_x + item_width > x + max_width or (index == 4 and max_width >= 900):
            current_x = x
            current_y += 28

        dash_attr = f' stroke-dasharray="{dash}"' if dash else ""
        svg.append(
            f'<line x1="{current_x}" y1="{current_y}" x2="{current_x + 28}" y2="{current_y}" '
            f'stroke="{color}" stroke-width="2.8"{dash_attr}/>'
        )
        add_text(svg, current_x + 35, current_y + 6, label, 18, fill=COLORS["text"])
        current_x += item_width

    return current_y


def write_combined_switch_svg(actions, switch_steps, window_hours, output_path):
    if len(switch_steps) != 2:
        raise ValueError("The combined Fig. 2 layout expects exactly two switch steps.")

    svg = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{COMBINED_WIDTH}" height="{COMBINED_HEIGHT}" '
        f'viewBox="0 0 {COMBINED_WIDTH} {COMBINED_HEIGHT}">',
        f'<rect width="{COMBINED_WIDTH}" height="{COMBINED_HEIGHT}" fill="#ffffff"/>',
    ]

    legend_y = add_global_legend(svg, 48, 26, COMBINED_WIDTH - 96)
    title_y = legend_y + 36
    add_switch_column(svg, actions, switch_steps[0], window_hours, 28, title_y, 520, "(a) Step 3000: cost -> low-carbon")
    add_switch_column(svg, actions, switch_steps[1], window_hours, 572, title_y, 520, "(b) Step 6000: low-carbon -> peak-aware")

    svg.append("</svg>")
    output_path.write_text("\n".join(svg), encoding="utf-8")


def write_switch_svg(actions, switch_step, window_hours, output_path):
    svg = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{SINGLE_WIDTH}" height="{SINGLE_HEIGHT}" '
        f'viewBox="0 0 {SINGLE_WIDTH} {SINGLE_HEIGHT}">',
        f'<rect width="{SINGLE_WIDTH}" height="{SINGLE_HEIGHT}" fill="#ffffff"/>',
    ]
    legend_y = add_global_legend(svg, 36, 26, SINGLE_WIDTH - 72)
    title_map = {
        3000: "Step 3000: cost -> low-carbon",
        6000: "Step 6000: low-carbon -> peak-aware",
    }
    add_switch_column(
        svg,
        actions,
        switch_step,
        window_hours,
        20,
        legend_y + 36,
        520,
        title_map.get(switch_step, f"Step {switch_step} semantic switch"),
    )
    svg.append("</svg>")
    output_path.write_text("\n".join(svg), encoding="utf-8")


def build_arg_parser():
    parser = argparse.ArgumentParser(
        description="Create switch-point SVG plots and summary CSV for dynamic LLM-MCTS runs."
    )
    parser.add_argument("--actions-path", type=Path, default=DEFAULT_ACTIONS_PATH)
    parser.add_argument("--pricing-path", type=Path, default=DEFAULT_PRICING_PATH)
    parser.add_argument("--carbon-path", type=Path, default=DEFAULT_CARBON_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--switch-steps", type=int, nargs="+", default=[3000, 6000])
    parser.add_argument("--window-hours", type=int, default=48)
    return parser


def main():
    args = build_arg_parser().parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    actions = pd.read_csv(args.actions_path)
    pricing_norm = normalize_array(load_first_numeric_column(args.pricing_path))
    carbon_norm = normalize_array(load_first_numeric_column(args.carbon_path))
    actions = enrich_actions(actions, pricing_norm, carbon_norm)

    stats = summarize_switch_windows(actions, args.switch_steps, args.window_hours)
    stats_path = args.output_dir / "dynamic_switch_window_stats.csv"
    stats.to_csv(stats_path, index=False)

    combined_path = args.output_dir / "dynamic_switch_fig2_revised.svg"
    if len(args.switch_steps) == 2:
        write_combined_switch_svg(
            actions,
            args.switch_steps,
            args.window_hours,
            combined_path,
        )

    for switch_step in args.switch_steps:
        output_path = args.output_dir / f"dynamic_switch_step_{switch_step}.svg"
        write_switch_svg(actions, switch_step, args.window_hours, output_path)

    print("Saved switch-window stats to:", stats_path)
    if len(args.switch_steps) == 2:
        print("Saved combined Fig. 2 plot to:", combined_path)
    for switch_step in args.switch_steps:
        print(
            "Saved switch plot to:",
            args.output_dir / f"dynamic_switch_step_{switch_step}.svg",
        )


if __name__ == "__main__":
    main()
