"""Study 3: open-loop comparison of desync-prevention methods on recorded
policy outputs. Works on any NPBT/NPBR pair the app or the Mac tools wrote:
a free-running trace (the inputs + the reference outputs) and replays of
the same inputs on other backends, precisions or devices.

  python3 analysis/prevention.py table --ref REF_trace.bin OUT.csv X_replay.bin [Y.bin ...]
      [--cpu CPU_replay.bin] [--label-from-name]

For every replay X and every prevention method, over all ticks whose
inputs are identical (rows aligned by tick index like determinism.py):
  rows_mismatch_pct   rows where the game would receive a different output
                      vector on the two peers after the method
  flips_per_1k        fire decisions (> 0.35) that still differ, per 1000 rows
  fallback_rows_pct   rows the hybrid methods hand to the CPU reference
  steer_err_rms       behaviour cost of the method itself: RMS change of the
                      reference steer caused by snapping (0 for none/fallback)
Methods: none; grid:K for K in GRIDS; fallback:D for D in DELTAS; grid+fallback
combinations. "fallback" takes the CPU reference's rows where the fast
output is within D of a decision boundary; the CPU reference is --cpu if
given (a second device's or backend's CPU replay) else REF itself.
"""

import csv
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from cosim import Prevent, FIRE_THRESHOLD  # noqa: E402
from determinism import read_bin  # noqa: E402

GRIDS = (2, 3, 4, 5, 6, 7, 8, 10, 12)
DELTAS = (0.001, 0.005, 0.02, 0.05)


def methods():
    out = ["none"] + [f"grid:{k}" for k in GRIDS] + [f"fallback:{d}" for d in DELTAS]
    out += [f"grid:{k}+fallback:{d}" for k in (3, 4, 5, 7) for d in (0.0001, 0.001, 0.01)]
    out += [f"grid:{k}+fallback:0.02" for k in (2, 3)]  # wide enough for the Qualcomm paths
    return out


def aligned(ref_ticks, other_ticks):
    """(ref_out, other_out) stacked over ticks with identical inputs."""
    by_tick = {t["tick"]: t for t in ref_ticks}
    a, b = [], []
    for t in other_ticks:
        u = by_tick.get(t["tick"])
        if u is None or u["out"].shape != t["out"].shape or t["alive"] == 0:
            continue
        if "feat" in t and not np.array_equal(u["feat"], t["feat"]):
            continue
        a.append(u["out"])
        b.append(t["out"])
    return np.concatenate(a), np.concatenate(b)


def evaluate(ref, fast, cpu, spec):
    prev = Prevent(spec)
    r_out, _ = prev.apply(ref, ref)          # the reference peer: grid only
    f_out, n_fb = prev.apply(fast, cpu)      # the fast peer: fallback rows from the CPU reference
    mism_rows = np.any(r_out != f_out, axis=1)
    flips = (r_out[:, 2] > FIRE_THRESHOLD) != (f_out[:, 2] > FIRE_THRESHOLD)
    steer_err = np.sqrt(np.mean((r_out[:, :2].astype(np.float64) - ref[:, :2]) ** 2))
    return {
        "method": spec,
        "rows": len(ref),
        "rows_mismatch_pct": 100.0 * mism_rows.mean(),
        "values_mismatch_pct": 100.0 * np.mean(r_out != f_out),
        "flips_per_1k": 1000.0 * flips.mean(),
        "fallback_rows_pct": 100.0 * n_fb / len(ref),
        "steer_err_rms": steer_err,
        "max_abs_residual": float(np.abs(r_out.astype(np.float64) - f_out).max()),
    }


def cmd_table(ref_path, out_csv, files, cpu_path, label_from_name):
    _, ref_ticks = read_bin(ref_path)
    cpu_ticks = read_bin(cpu_path)[1] if cpu_path else None
    rows = []
    for p in files:
        _, other = read_bin(p)
        ref, fast = aligned(ref_ticks, other)
        if cpu_ticks is not None:
            _, cpu = aligned(ref_ticks, cpu_ticks)
            if cpu.shape != fast.shape:
                sys.exit(f"--cpu replay {cpu_path} does not align with {p}")
        else:
            cpu = ref
        label = os.path.basename(p).replace("_replay.bin", "").replace("replay_", "") if label_from_name else p
        d = np.abs(ref.astype(np.float64) - fast)
        print(f"\n{label}: {len(ref)} rows; raw max |diff| {d.max():.3g}, mean {d.mean():.3g}, "
              f"bit-exact values {100 * np.mean(d == 0):.2f}%")
        print(f"  {'method':24s} {'rows_mism%':>10s} {'flips/1k':>9s} {'fallback%':>10s} {'steer_rms':>10s} {'max_resid':>10s}")
        for spec in methods():
            e = evaluate(ref, fast, cpu, spec)
            rows.append({"replay": label, "ref": ref_path, **e})
            print(f"  {spec:24s} {e['rows_mismatch_pct']:10.3f} {e['flips_per_1k']:9.3f} {e['fallback_rows_pct']:10.3f} "
                  f"{e['steer_err_rms']:10.4f} {e['max_abs_residual']:10.3g}")
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()), lineterminator="\n")
        w.writeheader()
        w.writerows(rows)
    print(f"\nwrote {out_csv}")


def main(argv):
    cpu = None
    if "--cpu" in argv:
        i = argv.index("--cpu")
        cpu = argv[i + 1]
        del argv[i:i + 2]
    label = "--label-from-name" in argv
    argv = [a for a in argv if a != "--label-from-name"]
    if len(argv) >= 5 and argv[0] == "table" and argv[1] == "--ref":
        cmd_table(argv[2], argv[3], argv[4:], cpu, label)
    else:
        sys.exit(__doc__)


if __name__ == "__main__":
    main(sys.argv[1:])
