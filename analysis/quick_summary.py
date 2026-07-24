"""Quick per-wave summary of pulled run CSVs (full analysis script comes in step 6)."""
import csv
import glob
import os
import sys

import numpy as np

RUNS = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(__file__), "..", "runs")


def pct(a, p):
    return float(np.percentile(a, p)) if len(a) else -1


for path in sorted(glob.glob(os.path.join(RUNS, "*_ticks.csv"))):
    with open(path) as f:
        rows = list(csv.DictReader(f))
    if not rows:
        continue
    steady = [r for r in rows if r["warmup"] == "0"]
    print(f"\n=== {os.path.basename(path)}  ({len(rows)} ticks, {len(steady)} steady) ===")
    print("wave batch ticks  inf_mean  inf_p95  inf_p99  tick_p95  miss  feat_ms  render  fps   temp")
    for wave in sorted({int(r["wave"]) for r in steady}):
        w = [r for r in steady if int(r["wave"]) == wave]
        inf = np.array([float(r["policy_infer_ms"]) for r in w])
        tick = np.array([float(r["tick_total_ms"]) for r in w])
        feat = np.array([float(r["feat_ms"]) for r in w])
        rend = np.array([float(r["render_ms"]) for r in w])
        fps = np.array([float(r["fps"]) for r in w])
        temp = np.array([float(r["battery_temp_c"]) for r in w])
        miss = sum(int(r["deadline_missed"]) for r in w)
        batch = w[0]["batch_size"]
        print(f"{wave:4d} {batch:>5} {len(w):5d}  {inf.mean():8.3f} {pct(inf,95):8.3f} "
              f"{pct(inf,99):8.3f} {pct(tick,95):9.3f} {miss:5d} {feat.mean():8.3f} "
              f"{rend.mean():7.3f} {fps.mean():5.0f} {temp.mean():6.1f}")

for path in sorted(glob.glob(os.path.join(RUNS, "*_gestures.csv"))):
    with open(path) as f:
        rows = list(csv.DictReader(f))
    if not rows:
        continue
    print(f"\n=== {os.path.basename(path)} ({len(rows)} gestures) ===")
    for r in rows:
        print(f"  {r['gesture_class']:8s} conf={float(r['confidence']):.2f} "
              f"infer={float(r['infer_ms']):.2f}ms up->effect={float(r['finger_up_to_effect_ms']):.1f}ms "
              f"acc={r['accepted']} {r['note']}")
