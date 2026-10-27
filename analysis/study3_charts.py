"""Study 3 figure: how far a rounding grid gets each backend, and what the
guarded CPU fallback costs. Reads the open-loop tables written by
analysis/prevention.py and draws one PNG with two panels.

  python3 analysis/study3_charts.py OPENLOOP_XL.csv OPENLOOP_TINY.csv OUT.png
"""

import csv
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# reference categorical palette (dataviz skill, light mode), fixed slot order
SERIES = [("np_f32", "another CPU kernel (f32, ULP-level)", "#2a78d6"),
          ("np_f16", "fp16 storage, fp32 accumulate", "#eb6834"),
          ("np_f16acc", "fp16 storage and accumulate", "#1baf7a"),
          ("lite_int8", "int8 model (quantised weights)", "#eda100")]
INK, INK2, MUTED, GRID, AXIS, SURFACE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#fcfcfb"
GRIDS = [2, 3, 4, 5, 6, 7, 8, 10, 12]


def load(path):
    rows = list(csv.DictReader(open(path)))
    out = {}
    for r in rows:
        key = r["replay"].replace(".bin", "").replace("replay_", "")
        out.setdefault(key, {})[r["method"]] = r
    return out


def style(ax, title, ylabel):
    ax.set_facecolor(SURFACE)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(AXIS)
        ax.spines[s].set_linewidth(0.8)
    ax.tick_params(colors=MUTED, labelsize=8, length=0)
    ax.grid(True, axis="y", color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    ax.set_title(title, loc="left", fontsize=10, color=INK, pad=10)
    ax.set_ylabel(ylabel, fontsize=8, color=INK2)
    ax.set_xlabel("rounding grid step  2^-k", fontsize=8, color=INK2)
    ax.set_xticks(GRIDS)
    ax.set_xticklabels([f"k={k}" if k in (2, 12) else str(k) for k in GRIDS])


def panel_mismatch(ax, tables, variant_label):
    floor = 3e-4   # 0 % plotted on the floor of the log axis, marked by an open marker
    for key, label, color in SERIES:
        t = tables.get(key)
        if not t:
            continue
        ys = [float(t[f"grid:{k}"]["rows_mismatch_pct"]) for k in GRIDS]
        ax.plot(GRIDS, [max(y, floor) for y in ys], color=color, linewidth=2, label=label, zorder=3)
        hit = [k for k, y in zip(GRIDS, ys) if y > 0]
        zero = [k for k, y in zip(GRIDS, ys) if y == 0]
        ax.scatter(hit, [y for y in ys if y > 0], s=28, color=color, zorder=4)
        ax.scatter(zero, [floor] * len(zero), s=28, facecolors=SURFACE, edgecolors=color, linewidths=1.5, zorder=4)
    raw = {key: float(tables[key]["none"]["rows_mismatch_pct"]) for key, _, _ in SERIES if key in tables}
    ax.set_yscale("log")
    ax.set_ylim(floor * 0.7, 200)
    ax.set_yticks([floor, 1e-3, 1e-2, 1e-1, 1, 10, 100])
    ax.set_yticklabels(["0", "0.001", "0.01", "0.1", "1", "10", "100"])
    style(ax, f"Rows still differing after snapping to the grid, {variant_label} model", "rows differing between peers, %")
    ax.legend(fontsize=7.5, frameon=False, loc="center right", bbox_to_anchor=(1.0, 0.42), labelcolor=INK2)
    ax.text(0.0, -0.22, "open marker = no row differed. Without a grid: "
            + ", ".join(f"{k.replace('np_', '').replace('lite_', '')} {v:.0f}%" for k, v in raw.items()),
            transform=ax.transAxes, fontsize=7.5, color=MUTED)


def panel_fallback(ax, tables, variant_label):
    for key, label, color in SERIES:
        t = tables.get(key)
        if not t or key == "lite_int8":
            continue
        for delta, ls in (("0.0001", "-"), ("0.01", (0, (4, 2)))):
            ks = [k for k in (3, 4, 5, 7)]
            ys = [float(t[f"grid:{k}+fallback:{delta}"]["fallback_rows_pct"]) for k in ks]
            resid = [float(t[f"grid:{k}+fallback:{delta}"]["rows_mismatch_pct"]) for k in ks]
            short = {"np_f32": "f32 kernel", "np_f16": "fp16", "np_f16acc": "fp16 acc"}[key]
            ax.plot(ks, ys, color=color, linewidth=2, linestyle=ls, zorder=3,
                    label=f"{short}, delta {delta}" if key == "np_f32" or delta == "0.01" else None)
            for k, y, r in zip(ks, ys, resid):
                ax.scatter([k], [y], s=28, color=color if r == 0 else SURFACE,
                           edgecolors=color, linewidths=1.5, zorder=4)
    style(ax, f"Cost of the guarded CPU fallback, {variant_label} model", "rows recomputed on the CPU reference, %")
    ax.set_xticks([3, 4, 5, 7])
    ax.set_xticklabels(["k=3", "4", "5", "7"])
    ax.set_ylim(0, 105)
    ax.text(0.0, -0.22, "solid: guard delta 1e-4 (covers ULP-level noise); dashed: delta 0.01 (covers fp16).\n"
            "filled marker: no residual mismatch; open marker: some rows still differ",
            transform=ax.transAxes, fontsize=7.5, color=MUTED, va="top")


def main(xl_csv, tiny_csv, out_png):
    xl, tiny = load(xl_csv), load(tiny_csv)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), facecolor=SURFACE)
    panel_mismatch(axes[0], xl, "XL")
    panel_fallback(axes[1], xl, "XL")
    axes[1].legend(fontsize=7.5, frameon=False, loc="center right", bbox_to_anchor=(1.0, 0.5), labelcolor=INK2)
    fig.subplots_adjust(left=0.06, right=0.98, bottom=0.26, top=0.86, wspace=0.3)
    fig.text(0.07, 0.95, "Offline desync prevention on the enemy-AI outputs, 123,354 enemy rows over one game (Mac co-simulation)",
             fontsize=11, color=INK)
    fig.savefig(out_png, dpi=160)
    print("wrote", out_png)
    # tiny model: same first panel, own file
    fig2, ax2 = plt.subplots(figsize=(6.4, 4.2), facecolor=SURFACE)
    panel_mismatch(ax2, tiny, "tiny")
    fig2.subplots_adjust(left=0.12, right=0.97, bottom=0.26, top=0.86)
    out2 = out_png.replace(".png", "_tiny.png")
    fig2.savefig(out2, dpi=160)
    print("wrote", out2)


if __name__ == "__main__":
    main(*sys.argv[1:4])
