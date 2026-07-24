"""Paper analysis of pulled run CSVs -> tables + figures in analysis/out/.

Usage: python analysis\\analyze.py [runs_dir]     (default: ../runs)

Reads the three CSV kinds per run id (ticks / gestures / coldstart), skips
malformed or empty files, drops warmup=1 ticks and -1 sentinel values, then
emits tick_stats.csv, summary.md and the paper figures (PNG).
"""
import csv
import glob
import os
import re
import sys
from datetime import datetime

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BUDGET_MS = 16.667  # 60 Hz tick budget
PARAMS = {"tiny": 291, "small": 1635, "medium": 5315, "large": 35331,
          "l192": 77571, "l256": 136195, "xlarge": 534531}
BACKEND_ORDER = {"cpu": 0, "gpu": 1, "npu": 2}
MODEL_ORDER = {m: i for i, m in enumerate(PARAMS)}

HERE = os.path.dirname(os.path.abspath(__file__))
RUNS = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "..", "runs")
OUT = os.path.join(HERE, "out")


# ---------------------------------------------------------------- loading

def pct(a, p):
    return float(np.percentile(a, p)) if len(a) else float("nan")


def fnum(row, key, default=float("nan")):
    try:
        return float(row.get(key, ""))
    except (TypeError, ValueError):
        return default


def read_rows(path):
    """DictReader with per-row salvage: recover coldstart rows whose previous
    row's note ate the newline (extra fields = a whole second row minus run_id)."""
    try:
        with open(path, newline="") as f:
            reader = csv.DictReader(f)
            if not reader.fieldnames:
                return [], 0
            rows, bad = [], 0
            for row in reader:
                extra = row.pop(None, None)
                rows.append(row)
                if extra:
                    if len(extra) == len(reader.fieldnames) - 1:  # merged line
                        rows.append(dict(zip(reader.fieldnames[1:], extra)))
                    else:
                        bad += 1
            return rows, bad
    except OSError:
        return [], 1


def run_meta(path):
    """Parse run_<YYYYMMDD_HHMMSS_kind_backend_model_prec>_<filekind>.csv."""
    base = os.path.basename(path)
    m = re.match(r"run_(\d{8})_(\d{6})_(game|bench)_(\w+?)_(\w+?)_(f32|int8)_", base)
    if not m:
        return None
    date, time_, kind, backend, model, prec = m.groups()
    return {
        "start": datetime.strptime(date + time_, "%Y%m%d%H%M%S"),
        "kind": kind, "backend": backend, "model": model, "precision": prec,
    }


tick_runs, gesture_rows, cold_rows = [], [], []
n_files = n_rows = n_skipped = n_badrows = 0

for path in sorted(glob.glob(os.path.join(RUNS, "run_*.csv"))):
    meta = run_meta(path)
    rows, bad = read_rows(path)
    n_badrows += bad
    if meta is None or not rows:
        n_skipped += 1
        continue
    n_files += 1
    n_rows += len(rows)
    if path.endswith("_ticks.csv"):
        tick_runs.append({**meta, "path": path, "rows": rows})
    elif path.endswith("_gestures.csv"):
        gesture_rows.extend(rows)
    elif path.endswith("_coldstart.csv"):
        cold_rows.extend(rows)

os.makedirs(OUT, exist_ok=True)
print(f"loaded {n_files} files / {n_rows} rows from {os.path.abspath(RUNS)} "
      f"({len(tick_runs)} tick runs, {len(gesture_rows)} gestures, "
      f"{len(cold_rows)} coldstart rows; {n_skipped} files skipped, "
      f"{n_badrows} malformed rows)")


# ---------------------------------------------------------------- tick stats

def steady(run):
    return [r for r in run["rows"] if r.get("warmup") == "0"]


def infer_stats(rows):
    inf = np.array([fnum(r, "policy_infer_ms") for r in rows])
    inf = inf[np.isfinite(inf) & (inf >= 0)]
    tick = np.array([fnum(r, "tick_total_ms") for r in rows])
    tick = tick[np.isfinite(tick) & (tick >= 0)]
    rend = np.array([fnum(r, "render_ms") for r in rows])
    rend = rend[np.isfinite(rend) & (rend >= 0)]
    miss = sum(1 for r in rows if r.get("deadline_missed") == "1")
    return {
        "n_ticks": len(rows), "miss": miss,
        "miss_rate": miss / len(rows) if rows else float("nan"),
        "mean": float(inf.mean()) if len(inf) else float("nan"),
        "std": float(inf.std()) if len(inf) else float("nan"),
        "p50": pct(inf, 50), "p95": pct(inf, 95), "p99": pct(inf, 99),
        "render_mean": float(rend.mean()) if len(rend) else float("nan"),
        "tick_p95": pct(tick, 95),
    }


# group steady ticks by (backend, model, precision, wave, N)
by_wave = {}
for run in tick_runs:
    for r in steady(run):
        key = (run["backend"], run["model"], run["precision"],
               int(fnum(r, "wave", -1)), int(fnum(r, "batch_size", -1)))
        by_wave.setdefault(key, []).append(r)

wave_keys = sorted(by_wave, key=lambda k: (BACKEND_ORDER.get(k[0], 9),
                                           MODEL_ORDER.get(k[1], 9), k[2], k[3]))
tick_stats_path = os.path.join(OUT, "tick_stats.csv")
with open(tick_stats_path, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["backend", "model", "precision", "wave", "N", "n_ticks",
                "infer_mean_ms", "infer_std_ms", "infer_p50_ms", "infer_p95_ms",
                "infer_p99_ms", "miss_count", "miss_rate", "render_mean_ms",
                "tick_total_p95_ms"])
    for key in wave_keys:
        s = infer_stats(by_wave[key])
        w.writerow(list(key) + [s["n_ticks"],
                   f"{s['mean']:.4f}", f"{s['std']:.4f}", f"{s['p50']:.4f}",
                   f"{s['p95']:.4f}", f"{s['p99']:.4f}", s["miss"],
                   f"{s['miss_rate']:.4f}", f"{s['render_mean']:.4f}",
                   f"{s['tick_p95']:.4f}"])
print(f"wrote {tick_stats_path} ({len(wave_keys)} rows)")


# ---------------------------------------------------------------- summary.md

def md_table(lines, header, rows):
    lines.append("| " + " | ".join(header) + " |")
    lines.append("|" + "|".join("---" for _ in header) + "|")
    for r in rows:
        lines.append("| " + " | ".join(str(c) for c in r) + " |")
    lines.append("")


md = ["# NPU Bench Swarm — run summary", "",
      f"Generated from `{os.path.abspath(RUNS)}` on {datetime.now():%Y-%m-%d %H:%M}. "
      f"{len(tick_runs)} tick runs, {len(gesture_rows)} gesture rows, "
      f"{len(cold_rows)} coldstart rows. Warmup ticks and -1 sentinels excluded; "
      f"tick budget {BUDGET_MS} ms.", ""]

# per run-config, aggregated across waves
by_cfg = {}
for run in tick_runs:
    key = (run["backend"], run["model"], run["precision"])
    by_cfg.setdefault(key, {"runs": 0, "rows": []})
    by_cfg[key]["runs"] += 1
    by_cfg[key]["rows"].extend(steady(run))

md.append("## Steady-state policy inference (game runs, all waves pooled)")
md.append("")
rows = []
for key in sorted(by_cfg, key=lambda k: (BACKEND_ORDER.get(k[0], 9),
                                         MODEL_ORDER.get(k[1], 9), k[2])):
    g = by_cfg[key]
    s = infer_stats(g["rows"])
    rows.append([key[0], key[1], key[2], g["runs"], s["n_ticks"],
                 f"{s['mean']:.3f}", f"{s['p50']:.3f}", f"{s['p95']:.3f}",
                 f"{s['p99']:.3f}", s["miss"], f"{100*s['miss_rate']:.2f}%",
                 f"{s['render_mean']:.3f}", f"{s['tick_p95']:.3f}"])
md_table(md, ["backend", "model", "prec", "runs", "ticks", "inf mean", "p50",
              "p95", "p99", "miss", "miss%", "render mean", "tick p95"], rows)

# cold-start table (phase == cold)
md.append("## Cold start (phase=cold rows; ms, mean over runs, -1 = failed excluded)")
md.append("")
cold_by = {}
for r in cold_rows:
    if r.get("phase") != "cold":
        continue
    cold_by.setdefault((r.get("backend"), r.get("model"), r.get("precision")), []).append(r)
rows = []
for key in sorted(cold_by, key=lambda k: (BACKEND_ORDER.get(k[0], 9),
                                          MODEL_ORDER.get(k[1] or "", 9), k[2])):
    g = cold_by[key]

    def m(col):
        v = np.array([fnum(r, col) for r in g])
        v = v[np.isfinite(v) & (v >= 0)]
        return f"{v.mean():.3f}" if len(v) else "-"
    rows.append([key[0], key[1], key[2], len(g), m("load_model_ms"),
                 m("delegate_create_ms"), m("interpreter_create_ms"),
                 m("first_infer_ms"), m("second_infer_ms")])
md_table(md, ["backend", "model", "prec", "n", "model load", "delegate create",
              "interp create", "first infer", "second infer"], rows)

# bench table (steady_iters > 0, batch 32)
md.append("## Micro-bench steady state (batch 32, 200 iters; ms, mean over runs)")
md.append("")
bench_by = {}
for r in cold_rows:
    if int(fnum(r, "steady_iters", 0)) > 0:
        bench_by.setdefault((r.get("backend"), r.get("model"), r.get("precision")), []).append(r)
rows = []
for key in sorted(bench_by, key=lambda k: (BACKEND_ORDER.get(k[0], 9),
                                           MODEL_ORDER.get(k[1] or "", 9), k[2])):
    g = bench_by[key]

    def m(col):
        v = np.array([fnum(r, col) for r in g])
        v = v[np.isfinite(v) & (v >= 0)]
        return f"{v.mean():.4f}" if len(v) else "-"
    rows.append([key[0], key[1], key[2], len(g), m("steady_mean_ms"),
                 m("steady_p50_ms"), m("steady_p95_ms"), m("steady_p99_ms")])
md_table(md, ["backend", "model", "prec", "n", "steady mean", "p50", "p95", "p99"], rows)

# gesture table
md.append("## Gestures (per backend x precision; -1 up->effect = rejected, excluded)")
md.append("")
gest_by = {}
for r in gesture_rows:
    gest_by.setdefault((r.get("backend"), r.get("precision")), []).append(r)
rows = []
for key in sorted(gest_by, key=lambda k: (BACKEND_ORDER.get(k[0], 9), k[1])):
    g = gest_by[key]
    inf = np.array([fnum(r, "infer_ms") for r in g])
    inf = inf[np.isfinite(inf) & (inf >= 0)]
    eff = np.array([fnum(r, "finger_up_to_effect_ms") for r in g])
    eff = eff[np.isfinite(eff) & (eff >= 0)]
    acc = sum(1 for r in g if r.get("accepted") == "1")
    rows.append([key[0], key[1], len(g), acc, f"{inf.mean():.3f}" if len(inf) else "-",
                 f"{pct(inf, 95):.3f}" if len(inf) else "-",
                 f"{eff.mean():.2f}" if len(eff) else "-",
                 f"{pct(eff, 95):.2f}" if len(eff) else "-"])
md_table(md, ["backend", "prec", "strokes", "accepted", "infer mean", "infer p95",
              "up->effect mean", "up->effect p95"], rows)

summary_path = os.path.join(OUT, "summary.md")
with open(summary_path, "w", encoding="utf-8") as f:
    f.write("\n".join(md))
print(f"wrote {summary_path}")


# ---------------------------------------------------------------- figures

def save(fig, name):
    p = os.path.join(OUT, name)
    fig.tight_layout()
    fig.savefig(p, dpi=150)
    plt.close(fig)
    print(f"wrote {p}")


def budget_line(ax):
    ax.axhline(BUDGET_MS, color="red", ls=":", lw=1, label=f"{BUDGET_MS} ms budget")


# latency_vs_N.png -- xlarge game runs, mean with p95 band vs batch size
fig, ax = plt.subplots(figsize=(7, 4.5))
xl = {k: v for k, v in by_wave.items() if k[1] == "xlarge"}
for bp in sorted({(k[0], k[2]) for k in xl}, key=lambda t: (BACKEND_ORDER.get(t[0], 9), t[1])):
    per_n = {}
    for k, v in xl.items():
        if (k[0], k[2]) == bp:
            per_n.setdefault(k[4], []).extend(v)
    ns = sorted(per_n)
    means = [infer_stats(per_n[n])["mean"] for n in ns]
    p95s = [infer_stats(per_n[n])["p95"] for n in ns]
    line, = ax.plot(ns, means, marker="o", label=f"{bp[0]} {bp[1]}")
    ax.fill_between(ns, means, p95s, alpha=0.2, color=line.get_color())
budget_line(ax)
ax.set_xlabel("batch size N (swarm size)")
ax.set_ylabel("policy_infer_ms (mean, band to p95)")
ax.set_title("xlarge policy net: inference latency vs batch size (game runs)")
ax.legend()
save(fig, "latency_vs_N.png")

# crossover.png -- bench steady mean vs model size
fig, ax = plt.subplots(figsize=(7, 4.5))
for (backend, prec) in sorted({(k[0], k[2]) for k in bench_by},
                              key=lambda t: (BACKEND_ORDER.get(t[0], 9), t[1])):
    pts = []
    for (b, model, p), g in bench_by.items():
        if (b, p) != (backend, prec) or model not in PARAMS:
            continue
        v = np.array([fnum(r, "steady_mean_ms") for r in g])
        v = v[np.isfinite(v) & (v > 0)]
        if len(v):
            pts.append((PARAMS[model], float(v.mean())))
    if not pts:
        continue
    pts.sort()
    ax.plot([x for x, _ in pts], [y for _, y in pts], marker="o",
            ls="--" if prec == "int8" else "-", label=f"{backend} {prec}")
if any(fnum(r, "steady_mean_ms") > 2 for g in bench_by.values() for r in g):
    budget_line(ax)
ax.set_xscale("log")
ax.set_xlabel("policy-net parameters")
ax.set_ylabel("steady_mean_ms (batch 32)")
ax.set_title("backend crossover: steady inference vs model size (micro-bench)")
ax.legend()
save(fig, "crossover.png")

# thermal.png -- longest back-to-back series of same-config game runs
series_by = {}
for run in tick_runs:
    if run["kind"] == "game":
        series_by.setdefault((run["backend"], run["model"], run["precision"]), []).append(run)
best = []
for key, runs in series_by.items():
    runs.sort(key=lambda r: r["start"])
    chain = [runs[0]]
    for r in runs[1:]:
        if (r["start"] - chain[-1]["start"]).total_seconds() < 600:
            chain.append(r)
        else:
            if len(chain) > 1 and sum(len(c["rows"]) for c in chain) > sum(len(c["rows"]) for c in best):
                best = chain
            chain = [r]
    if len(chain) > 1 and sum(len(c["rows"]) for c in chain) > sum(len(c["rows"]) for c in best):
        best = chain
if best:
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax2 = ax.twinx()
    t0 = best[0]["start"]
    win = 120
    for run in best:
        inf = np.array([fnum(r, "policy_infer_ms") for r in run["rows"]])
        fps = np.array([max(fnum(r, "fps", 60.0), 1.0) for r in run["rows"]])
        temp = np.array([fnum(r, "battery_temp_c") for r in run["rows"]])
        t = (run["start"] - t0).total_seconds() + np.cumsum(1.0 / fps)
        roll = np.array([np.median(inf[max(0, i - win + 1):i + 1]) for i in range(len(inf))])
        ax.plot(t / 60.0, roll, color="C0")
        ax2.plot(t / 60.0, temp, color="C3", alpha=0.7)
    budget_line(ax)
    ax.set_xlabel("minutes since first run in series")
    ax.set_ylabel(f"policy_infer_ms (rolling median, {win} ticks)", color="C0")
    ax2.set_ylabel("battery_temp_c", color="C3")
    cfg = best[0]
    ax.set_title(f"sustained series: {cfg['backend']} {cfg['model']} {cfg['precision']} "
                 f"x{len(best)} runs")
    ax.legend(loc="upper left")
    save(fig, "thermal.png")
else:
    print("thermal.png skipped: no back-to-back same-config series found")

# jitter.png -- violin of xlarge configs at largest common N
want = [("cpu", "f32"), ("cpu", "int8"), ("npu", "f32"), ("npu", "int8")]
per_cfg_n = {bp: {k[4] for k in xl if (k[0], k[2]) == bp} for bp in want}
common = set.intersection(*[s for s in per_cfg_n.values() if s]) if any(per_cfg_n.values()) else set()
if common:
    n_star = max(common)
    data, labels = [], []
    for bp in want:
        rows_ = [r for k in xl if (k[0], k[2]) == bp and k[4] == n_star for r in xl[k]]
        inf = np.array([fnum(r, "policy_infer_ms") for r in rows_])
        inf = inf[np.isfinite(inf) & (inf >= 0)]
        if len(inf):
            data.append(inf)
            labels.append(f"{bp[0]}\n{bp[1]}")
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.violinplot(data, showmedians=True)
    ax.set_xticks(range(1, len(labels) + 1), labels)
    budget_line(ax)
    ax.set_ylabel("policy_infer_ms per tick")
    ax.set_title(f"per-tick jitter, xlarge @ N={n_star} (steady ticks)")
    ax.legend()
    save(fig, "jitter.png")
else:
    print("jitter.png skipped: no common N across xlarge configs")

# gesture_delay.png -- finger_up_to_effect_ms histogram per backend
eff_by = {}
for r in gesture_rows:
    v = fnum(r, "finger_up_to_effect_ms")
    if np.isfinite(v) and v >= 0:
        eff_by.setdefault(r.get("backend", "?"), []).append(v)
if eff_by:
    fig, ax = plt.subplots(figsize=(7, 4.5))
    lo = min(min(v) for v in eff_by.values())
    hi = max(max(v) for v in eff_by.values())
    bins = np.linspace(lo, hi, 40)
    for backend in sorted(eff_by, key=lambda b: BACKEND_ORDER.get(b, 9)):
        ax.hist(eff_by[backend], bins=bins, alpha=0.55,
                label=f"{backend} (n={len(eff_by[backend])})")
    ax.axvline(BUDGET_MS, color="red", ls=":", lw=1, label=f"{BUDGET_MS} ms budget")
    ax.set_xlabel("finger_up_to_effect_ms")
    ax.set_ylabel("gestures")
    ax.set_title("gesture cast latency: finger up -> world-state change")
    ax.legend()
    save(fig, "gesture_delay.png")
else:
    print("gesture_delay.png skipped: no accepted gestures")

# wave_setup.png -- wave-boundary setup_ms by backend x precision (xlarge)
setup_by = {}
for run in tick_runs:
    if run["model"] != "xlarge" or run["kind"] != "game":
        continue
    for r in run["rows"]:
        m = re.search(r"setup_ms=([0-9.]+)", r.get("note") or "")
        if m:
            setup_by.setdefault((run["backend"], run["precision"]), []).append(float(m.group(1)))
if setup_by:
    keys = sorted(setup_by, key=lambda k: (BACKEND_ORDER.get(k[0], 9), k[1]))
    means = [float(np.mean(setup_by[k])) for k in keys]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    bars = ax.bar(range(len(keys)), means)
    ax.bar_label(bars, fmt="%.2f")
    ax.set_xticks(range(len(keys)), [f"{b}\n{p}" for b, p in keys])
    ax.set_yscale("log")
    ax.set_ylabel("wave-boundary setup_ms (mean, log scale)")
    ax.set_title("per-wave batch resize/recompile cost, xlarge game runs")
    save(fig, "wave_setup.png")
else:
    print("wave_setup.png skipped: no setup_ms notes found")

print("done.")
