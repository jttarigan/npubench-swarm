"""Study 3 paper figures (static PNG + PDF for the Word manuscript).

  python3 analysis/study3_figures.py [OUT_DIR]     default paper/study3/figures

Figure 2: prevention. (a) Rows whose output vector still differs from the CPU
reference after rounding to a grid of 2^-K, per peer (XL replays,
`analysis/out_study3/*/prevention_replay_xl*.csv`). (b) Rows a guarded grid (K = 3)
sends back to the CPU against the guard width, with each peer's largest
difference marked: a guard wider than that difference makes the peers identical.

Figure 3: closed loop. Mean distance between matching live enemies of an
accelerator game and the same device's CPU game of the same precision, per
second of play (XL), log scale; gaps where the two games are in different waves.

Figure 1: largest absolute output difference between every pair of XL output
classes (the replays of one game, `analysis/out_study3/crosschip_replay_pairs.csv`),
log colour scale, one blue ramp. The caption lives in the draft (section 7).
Figure S1 (supplement): the same for the nine tiny-model classes
(`crosschip_replay_pairs_tiny.csv`, Android devices and the Mac).
"""
import csv
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

plt.rcParams["pdf.fonttype"] = 42  # TrueType in the PDFs (submission systems reject Type 3 fonts)
import numpy as np  # noqa: E402
from matplotlib.colors import BoundaryNorm, LinearSegmentedColormap, ListedColormap, LogNorm  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from determinism import read_bin  # noqa: E402

PAIRS = "analysis/out_study3/crosschip_replay_pairs.csv"
REF = "runs_study3/s6lite/files/run_20261013_223444_game_cpu_xlarge_f32_trace.bin"

# one representative replay per XL class, grouped by arithmetic regime
# (class ids as in analysis/out_study3/crosschip.md)
CLASSES = [
    ("R1", "Android CPU F32 (5 chips)", "s6lite cpu_f32"),
    ("R11", "iPhone CPU F32 (2 chips)", "iphone12 cpu_f32"),
    ("R16", "Mac CPU F32 (XNNPACK)", "mac f32"),
    ("R19", "iPhone CPU F32, LiteRT 2.17", "iphone12_lite217 cpu_f32"),
    ("R10", "Xclipse 540 GPU", "a56 gpu_f32"),
    ("R5", "MediaTek NPU F32", "poco npu_f32"),
    ("R3", "Mali-G72 GPU", "s6lite gpu_f32"),
    ("R4", "Mali-G615 GPU", "poco gpu_f32"),
    ("R15", "Core ML F32", "iphone12 npu_f32"),
    ("R13", "Metal GPU F32 (A14)", "iphone12 gpu_f32"),
    ("R17", "Metal GPU F32 (A19 Pro)", "iphone17pm gpu_f32"),
    ("R9", "Adreno 650 GPU", "tab gpu_f32"),
    ("R20", "Adreno 840 GPU", "fold8 gpu_f32"),
    ("R8", "Qualcomm NNAPI F32", "tab npu_f32"),
    ("R2", "Android CPU INT8 (5 chips)", "s6lite cpu_int8"),
    ("R6", "MediaTek NPU INT8", "poco npu_int8"),
    ("R12", "Apple CPU INT8 (iOS, Mac)", "iphone12 cpu_int8"),
    ("R14", "Metal GPU INT8 (A14)", "iphone12 gpu_int8"),
    ("R18", "Metal GPU INT8 (A19 Pro)", "iphone17pm gpu_int8"),
    ("R7", "Qualcomm NNAPI INT8", "tab npu_int8"),
]
GROUPS = [(0, 4, "float32, CPU"), (4, 14, "float32, GPU / NPU"), (14, 20, "int8")]

# tiny model: classes T1-T10 as in crosschip.md (no iOS tiny replays)
PAIRS_TINY = "analysis/out_study3/crosschip_replay_pairs_tiny.csv"
CLASSES_TINY = [
    ("T1", "Android CPU F32 (5 chips)", "s6lite cpu_f32"),
    ("T9", "Mac CPU F32 (XNNPACK)", "mac f32"),
    ("T8", "Xclipse 540 GPU", "a56 gpu_f32"),
    ("T4", "MediaTek NPU F32", "poco npu_f32"),
    ("T3", "Mali-G72 / G615 GPU", "s6lite gpu_f32"),
    ("T7", "Adreno 650 GPU", "tab gpu_f32"),
    ("T11", "Adreno 840 GPU", "fold8 gpu_f32"),
    ("T6", "Qualcomm NNAPI F32", "tab npu_f32"),
    ("T2", "Android CPU INT8, MediaTek NPU INT8", "s6lite cpu_int8"),
    ("T10", "Mac CPU INT8", "mac int8"),
    ("T5", "Qualcomm NNAPI INT8", "tab npu_int8"),
]
GROUPS_TINY = [(0, 2, "f32, CPU"), (2, 8, "f32, GPU / NPU"), (8, 11, "int8")]

# sequential blue ramp (dataviz reference palette, steps 100..700)
BLUES = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#1c5cab", "#104281", "#0d366b"]
INK, MUTED, GRID = "#1f1f1e", "#6b6a66", "#d9d8d4"
# Figure 1: one blue per decade from 1e-7 to 1, evenly spaced in lightness
DECADES = ["#e3eefb", "#b9d4f5", "#8ab6ec", "#5a96e0", "#2f74c9", "#1d4f94", "#0f2a55"]


def load_pairs(path=PAIRS):
    d, values = {}, None
    with open(path) as f:
        for r in csv.DictReader(f):
            d[(r["a"], r["b"])] = d[(r["b"], r["a"])] = float(r["max_abs_diff"])
            values = int(r["values"])
    return d, values


def fig1(out_dir, classes=CLASSES, groups=GROUPS, pairs_csv=PAIRS,
         name="fig1_output_classes", figsize=(7.4, 6.8)):
    pairs, values = load_pairs(pairs_csv)
    n = len(classes)
    m = np.full((n, n), np.nan)
    for i in range(n):
        for j in range(i):
            m[i, j] = pairs[(classes[i][2], classes[j][2])]
    # one step of the blue ramp per decade, so ULP-, fp16- and int8-sized cells read apart
    bounds = 10.0 ** np.arange(-7, 1)
    cmap = ListedColormap(DECADES)
    cmap.set_bad("white")
    norm = BoundaryNorm(bounds, cmap.N)

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 7.5})
    fig, ax = plt.subplots(figsize=figsize, dpi=300)
    # classes differ by definition, so the triangle has no empty diagonal: drop the
    # first row and the last column
    shown = m[1:, :-1]
    im = ax.imshow(shown, cmap=cmap, norm=norm, interpolation="none")
    for i in range(n - 1):
        for j in range(i + 1):
            v = shown[i, j]
            txt = f"{v:.0e}".replace("e-0", "e-").replace("e+00", "")
            ax.text(j, i, txt, ha="center", va="center", fontsize=5.6,
                    color="white" if norm(v) >= 4 else INK)
    ax.set_yticks(range(n - 1), [f"{c[0]}  {c[1]}" for c in classes[1:]])
    ax.set_xticks(range(n - 1), [c[0] for c in classes[:-1]])
    ax.tick_params(length=0, colors=INK)
    for s in ax.spines.values():
        s.set_visible(False)
    # white cell gaps and group separators
    ax.set_xticks(np.arange(-.5, n - 1, 1), minor=True)
    ax.set_yticks(np.arange(-.5, n - 1, 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=1.2)
    ax.tick_params(which="minor", length=0)
    for start, end, group in groups:
        if start:  # boundaries between regimes (row index = class index - 1)
            ax.plot([start - .5, start - .5], [start - 1.5, n - 1.5], color=MUTED, linewidth=0.8)
            ax.plot([-.5, start - .5], [start - 1.5, start - 1.5], color=MUTED, linewidth=0.8)
        ax.text(n - 1.3, (max(start, 1) + end) / 2 - 1.5, group, rotation=-90, va="center",
                ha="left", color=MUTED, fontsize=7)
    cb = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.09, ticks=bounds,
                      format=matplotlib.ticker.LogFormatterMathtext())
    sep = "\n" if figsize[0] < 6 else " "   # the small figure needs a two-line label
    cb.set_label(f"largest absolute output difference{sep}over {values:,} values", color=INK)
    cb.outline.set_visible(False)
    cb.ax.tick_params(colors=INK, length=2)
    fig.tight_layout()
    os.makedirs(out_dir, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(out_dir, f"{name}.{ext}"), bbox_inches="tight")
    plt.close(fig)


# categorical slots 1-6 (dataviz reference palette, validated light: CVD 9.1,
# contrast WARN on 3, 4, 5 -> every line is direct-labelled)
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]
F = "runs_study3/{}/files/run_{}_trace.bin"


def _first_trace(dev, cfg):
    import glob
    hits = sorted(glob.glob(F.format(dev, f"*_game_{cfg}")))
    if not hits:
        raise FileNotFoundError(f"{dev} {cfg}")
    return hits[0]


def drift_per_second(pa, pb):
    _, A = read_bin(pa)
    _, B = read_bin(pb)
    n = min(len(A), len(B))
    d = np.full(n, np.nan)
    for i in range(n):
        a, b = A[i], B[i]
        if a["wave"] != b["wave"] or len(a["pos"]) != len(b["pos"]):
            continue
        both = a["alive_mask"] & b["alive_mask"]
        if both.any():
            d[i] = np.linalg.norm(a["pos"][both] - b["pos"][both], axis=1).mean()
    secs = n // 60
    with np.errstate(all="ignore"):
        per_s = np.array([np.nanmean(d[k * 60:(k + 1) * 60]) if np.isfinite(d[k * 60:(k + 1) * 60]).any()
                          else np.nan for k in range(secs)])
    waves = [t["tick"] / 60 for p, t in zip(B, B[1:]) if t["wave"] != p["wave"]]  # CPU game
    return per_s, waves


def fig3(out_dir):
    cfgs = {"Mali-G615 GPU F32 (POCO)": ("poco", "gpu_xlarge_f32", "cpu_xlarge_f32"),
            "MediaTek NPU F32 (POCO)": ("poco", "npu_xlarge_f32", "cpu_xlarge_f32"),
            "Xclipse 540 GPU F32 (A56)": ("a56", "gpu_xlarge_f32", "cpu_xlarge_f32"),
            "Adreno 650 GPU F32 (Tab S7+)": ("tab", "gpu_xlarge_f32", "cpu_xlarge_f32"),
            "Qualcomm NNAPI F32 (Tab S7+)": ("tab", "npu_xlarge_f32", "cpu_xlarge_f32"),
            "Qualcomm NNAPI INT8 (Tab S7+)": ("tab", "npu_xlarge_int8", "cpu_xlarge_int8"),
            "Core ML F32 (iPhone 12)": ("iphone12", "npu_xlarge_f32", "cpu_xlarge_f32")}
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 7.5})
    fig, ax = plt.subplots(figsize=(7.0, 4.0), dpi=300)
    waves = None
    ends = []
    for k, (label, (dev, acc, cpu)) in enumerate(cfgs.items()):
        y, w = drift_per_second(_first_trace(dev, acc), _first_trace(dev, cpu))
        waves = waves or w
        x = np.arange(len(y)) + 0.5
        ls = "-" if "GPU" in label else (0, (4, 1.5))
        ax.plot(x, y, color=SERIES7[k], linewidth=1.2, linestyle=ls, label=label)
        last = np.flatnonzero(np.isfinite(y))[-1]
        ends.append([y[last], x[last], label, SERIES7[k]])
    # direct labels at the right edge, nudged apart in log space
    ends.sort(key=lambda e: e[0])
    ly = [np.log10(e[0]) for e in ends]
    for i in range(1, len(ly)):
        ly[i] = max(ly[i], ly[i - 1] + 0.2)
    xmax = max(e[1] for e in ends)
    for (yv, xv, label, col), l10 in zip(ends, ly):
        ax.annotate(label, xy=(xv, yv), xytext=(xmax + 3, 10 ** l10), fontsize=6.5, color=INK,
                    va="center", arrowprops=dict(arrowstyle="-", color=col, lw=0.8))
    for t in waves:
        ax.axvline(t, color=GRID, linewidth=0.6, zorder=0)
    ax.set_yscale("log")
    ax.set_xlim(0, xmax)
    ax.set_xlabel("time in the game (s; vertical lines: wave starts of the CPU game)", color=INK)
    ax.set_ylabel("mean distance between matching\nlive enemies (px, 1 s mean)", color=INK)
    ax.grid(axis="y", color=GRID, linewidth=0.5)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(MUTED)
    ax.tick_params(colors=INK, length=2)
    leg = ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), frameon=False, fontsize=6.5, ncol=3,
                    title="solid lines: GPU; dashed lines: NPU", title_fontsize=6.5)
    leg.get_title().set_color(INK)
    fig.tight_layout()
    os.makedirs(out_dir, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(out_dir, f"fig3_closed_loop_drift.{ext}"), bbox_inches="tight")
    plt.close(fig)


SERIES7 = SERIES + ["#6250d6"]  # validated light: CVD 9.1, contrast WARN -> direct labels
PREV_PEERS = [  # label, csv, replay suffix, largest difference from R1 (crosschip.md)
    ("Mali-G72 / G615 GPU", "poco", "152456_replay_gpu_xlarge_f32_replay.bin", 4.77e-3),
    ("MediaTek NPU F32", "poco", "152515_replay_npu_xlarge_f32_replay.bin", 4.44e-3),
    ("Xclipse 540 GPU", "a56", "190737_replay_gpu_xlarge_f32_replay.bin", 2.63e-3),
    ("Core ML F32", "iphone12", "020_replay_npu_xlarge_f32_replay.bin", 5.17e-3),
    ("Adreno 650 GPU", "tab", "084652_replay_gpu_xlarge_f32_replay.bin", 1.12e-2),
    ("Metal GPU F32 (A14)", "iphone12", "012_replay_gpu_xlarge_f32_replay.bin", 7.74e-3),
    ("Qualcomm NNAPI F32", "tab", "084633_replay_npu_xlarge_f32_replay.bin", 1.62e-2),
]
ULP_PEERS = [("Mac CPU F32", "s6lite/prevention_replay_xl_mac.csv", "mac_lite_f32.bin", 1.31e-6),
             ("iPhone CPU F32", "iphone12/prevention_replay_xl.csv", "004_replay_cpu_xlarge_f32_replay.bin", 1.19e-6)]
KS = (2, 3, 4, 5, 6, 7, 8, 10, 12)


def _grid_curve(csv_path, suffix):
    with open(csv_path) as f:
        rows = [r for r in csv.DictReader(f) if r["replay"].endswith(suffix)]
    d = {r["method"]: float(r["rows_mismatch_pct"]) for r in rows}
    if not d:
        raise KeyError(f"{suffix} not in {csv_path}")
    return np.array([d[f"grid:{k}"] for k in KS])


def _guard_curve(widths, k=3):
    from cosim import Prevent
    _, ticks = read_bin(REF)
    ref = np.concatenate([t["out"] for t in ticks if t["alive"]])
    out = []
    for w in widths:
        _, n_fb = Prevent(f"grid:{k}+fallback:{w:g}").apply(ref, ref)
        out.append(100.0 * n_fb / len(ref))
    return np.array(out)


def _label_right(ax, ends, x, step=0.16, fs=6.5):
    ends.sort(key=lambda e: e[0])
    ly = [np.log10(e[0]) for e in ends]
    for i in range(1, len(ly)):
        ly[i] = max(ly[i], ly[i - 1] + step)
    for (yv, xv, label, col), l10 in zip(ends, ly):
        ax.annotate(label, xy=(xv, yv), xytext=(x, 10 ** l10), fontsize=fs, color=INK,
                    va="center", arrowprops=dict(arrowstyle="-", color=col, lw=0.8))


def _style(ax):
    ax.grid(axis="y", color=GRID, linewidth=0.5)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(MUTED)
    ax.tick_params(colors=INK, length=2)


def fig2(out_dir):
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 7.5})
    fig, (a, b) = plt.subplots(1, 2, figsize=(7.2, 3.6), dpi=300, gridspec_kw={"wspace": 0.3})
    for k, (label, dev, suffix, _) in enumerate(PREV_PEERS):
        y = _grid_curve(f"analysis/out_study3/{dev}/prevention_replay_xl.csv", suffix)
        a.plot(KS, y, color=SERIES7[k], linewidth=1.2, marker="o", markersize=2.5, label=label)
    for (label, path, suffix, _), ls in zip(ULP_PEERS, ("-", (0, (3, 1.5)))):
        y = _grid_curve(f"analysis/out_study3/{path}", suffix)
        y = np.where(y > 0, y, np.nan)  # zero rows cannot sit on a log axis
        a.plot(KS, y, color=MUTED, linewidth=1.0, linestyle=ls, marker="o", markersize=2, label=label)
    a.annotate("other CPU kernels:\n0 rows up to K = 4", xy=(2.1, 4e-3), fontsize=6.5, color=MUTED, ha="left")
    a.set_yscale("log")
    a.set_xticks(KS)
    a.set_xlabel(r"grid $2^{-K}$: K", color=INK)
    a.set_ylabel("rows still differing (% of 123,354)", color=INK)
    a.set_title("(a) rounding alone", color=INK, fontsize=8, loc="left")
    _style(a)

    widths = np.logspace(-6.5, -1.2, 80)
    cost = _guard_curve(widths, k=2)
    b.plot(widths, _guard_curve(widths, k=3), color=MUTED, linewidth=1.0, linestyle=(0, (3, 1.5)))
    b.plot(widths, cost, color=INK, linewidth=1.2)
    b.annotate(r"grid $2^{-3}$", xy=(2.2e-2, float(np.interp(np.log10(2.2e-2), np.log10(widths), _guard_curve(widths, k=3)))),
               xytext=(-6, 0), textcoords="offset points", ha="right", va="center", fontsize=6.5, color=MUTED)
    peers = [(lab, w, SERIES7[k]) for k, (lab, _, _, w) in enumerate(PREV_PEERS)]
    peers += [(lab, w, MUTED) for lab, _, _, w in ULP_PEERS]
    for lab, w, col in peers:
        c = np.interp(np.log10(w), np.log10(widths), cost)
        b.plot([w], [c], marker="o", markersize=4, color=col, markeredgecolor="white",
               markeredgewidth=0.6, zorder=3)
    for w, txt, off, ha in ((1e-4, "guard 1e-4\n(other CPU kernels)", (0, 14), "center"),
                            (1e-2, "guard 0.01", (-14, 18), "right"), (2e-2, "guard 0.02", (-14, 22), "right")):
        c = np.interp(np.log10(w), np.log10(widths), cost)
        b.plot([w], [c], marker="|", markersize=7, color=INK, zorder=4)
        b.annotate(f"{txt}: {c:.2g} %", xy=(w, c), xytext=off, textcoords="offset points",
                   ha=ha, fontsize=6.5, color=INK,
                   arrowprops=dict(arrowstyle="-", color=MUTED, linewidth=0.5, shrinkA=0, shrinkB=2))
    b.set_xscale("log")
    b.set_xlabel("guard width\n(dots: each peer's largest difference)", color=INK)
    b.set_ylabel("rows recomputed with the portable kernel (%)", color=INK)
    b.set_title(r"(b) grid $2^{-2}$ with a guard", color=INK, fontsize=8, loc="left")
    _style(b)
    fig.legend(*a.get_legend_handles_labels(), loc="upper center", bbox_to_anchor=(0.5, -0.04),
               ncol=3, frameon=False, fontsize=6.5)
    fig.tight_layout()
    os.makedirs(out_dir, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(out_dir, f"fig2_prevention.{ext}"), bbox_inches="tight")
    plt.close(fig)


def main(argv):
    out_dir = argv[0] if argv else "paper/study3/figures"
    fig1(out_dir)
    fig1(out_dir, CLASSES_TINY, GROUPS_TINY, PAIRS_TINY, "figS1_output_classes_tiny", (5.2, 4.4))
    fig2(out_dir)
    fig3(out_dir)


if __name__ == "__main__":
    main(sys.argv[1:])
