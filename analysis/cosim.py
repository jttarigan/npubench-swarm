"""Study 3: closed-loop co-simulation on the Mac. The real game code
(tools/hostsim, built from app/src/main/cpp/src/game.cpp) runs the
simulation; this driver answers every tick's policy request with the real
XL model computed in a chosen numeric backend (analysis/policy_backends.py),
optionally after a desync-prevention method. Two peers of a lockstep game
are modelled as two runs with the same prevention and different backends:
if their traces are byte-identical, they would have stayed in sync.

  python3 analysis/cosim.py run OUT_PREFIX --backend B [--prevent P] [--variant V]
  python3 analysis/cosim.py matrix OUT_DIR [--variant V] [--quick]
  python3 analysis/cosim.py compare OUT_DIR          (pairs table from a matrix)

Prevention specs (P), applied by every peer to the model output before the
game uses it:
  none
  grid:K            snap every output to a multiple of 2^-K (round half even)
  fallback:D        hybrid: rows whose fire value is within D of the fire
                    threshold (and, with a grid, whose steer is within D of a
                    rounding boundary) are recomputed on the CPU reference
                    (lite_f32) instead; the fast backend serves the rest
  grid:K+fallback:D both
Outputs per run: OUT_PREFIX_trace.bin (NPBT, same format as the app),
OUT_PREFIX_ticks.csv, OUT_PREFIX_run.json (outcome, fallback counts).
"""

import csv
import itertools
import json
import os
import struct
import subprocess
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
HOSTSIM = os.path.normpath(os.path.join(HERE, "..", "tools", "hostsim", "hostsim"))
FEAT, OUT = 14, 3
FIRE_THRESHOLD = 0.35
HDR = struct.Struct("<iiii")


# ------------------------------------------------------------------ prevention

class Prevent:
    def __init__(self, spec):
        self.spec = spec
        self.k = None
        self.delta = None
        for part in spec.split("+"):
            if part == "none":
                continue
            kind, val = part.split(":")
            if kind == "grid":
                self.k = int(val)
            elif kind == "fallback":
                self.delta = float(val)
            else:
                raise ValueError(f"bad prevention spec {spec}")
        self.needs_ref = self.delta is not None

    def near_boundary(self, out):
        """Rows where a perturbation smaller than delta could change what the
        game receives: the fire value within delta of the threshold, or (with a
        grid) any of the three values within delta of a rounding edge. The
        rounding edge nearest 0.35 becomes the effective fire threshold once a
        grid is on, so the fire column must be edge-guarded too."""
        near = np.abs(out[:, 2] - FIRE_THRESHOLD) < self.delta
        if self.k is not None:
            scaled = out.astype(np.float64) * (2.0 ** self.k)
            frac = scaled - np.floor(scaled)
            dist = np.abs(frac - 0.5) / (2.0 ** self.k)     # distance to a rounding edge
            near |= np.any(dist < self.delta, axis=1)
        return near

    def apply(self, out, ref=None):
        """Returns (output the game gets, number of rows served by the reference)."""
        out = out.astype(np.float32, copy=True)
        n_fb = 0
        if self.delta is not None:
            near = self.near_boundary(out)
            n_fb = int(near.sum())
            if n_fb:
                out[near] = ref[near]
        if self.k is not None:
            step = np.float32(2.0 ** -self.k)
            out = (np.round(out / step) * step).astype(np.float32)
        return out, n_fb


# ------------------------------------------------------------------ one run

def run_game(prefix, backend_name, prevent_spec, variant="xlarge", ref_name="lite_f32", seed=None):
    from policy_backends import make_backend
    prev = Prevent(prevent_spec)
    fast = make_backend(backend_name, variant)
    ref = fast if backend_name == ref_name else (make_backend(ref_name, variant) if prev.needs_ref else None)
    t0 = time.time()
    env = dict(os.environ, HOSTSIM_SEED=str(seed)) if seed is not None else None
    p = subprocess.Popen([HOSTSIM, prefix, "--pipe"], stdin=subprocess.PIPE,
                         stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=env)
    ticks = fb_rows = rows = 0
    fb_ticks = 0
    while True:
        hdr = p.stdout.read(HDR.size)
        if len(hdr) < HDR.size:
            raise RuntimeError(f"hostsim ended early: {p.stderr.read().decode()}")
        tick, wave, batch, alive = HDR.unpack(hdr)
        if tick < 0:
            final = {"wave": wave, "score": batch, "hp": alive}
            break
        ticks += 1
        if alive == 0:
            continue
        feat = np.frombuffer(p.stdout.read(4 * alive * FEAT), "<f4").reshape(alive, FEAT)
        out = fast.run(feat, batch)
        r = out
        if prev.needs_ref and ref is not fast:  # the reference only on the rows the guard sends it
            near = prev.near_boundary(out)
            if near.any():
                r = out.copy()
                r[near] = ref.run(feat[near], batch)
        out, n_fb = prev.apply(out, r)
        rows += alive
        fb_rows += n_fb
        fb_ticks += n_fb > 0
        p.stdin.write(np.ascontiguousarray(out, np.float32).tobytes())
        p.stdin.flush()
    p.stdin.close()
    p.wait()
    info = {
        "prefix": prefix, "backend": backend_name, "prevent": prevent_spec, "variant": variant,
        "seed": seed, "ticks": ticks, "rows": rows, "fallback_rows": fb_rows,
        "fallback_row_pct": 100.0 * fb_rows / max(rows, 1),
        "fallback_ticks": fb_ticks, "fallback_tick_pct": 100.0 * fb_ticks / max(ticks, 1),
        **{f"final_{k}": v for k, v in final.items()},
        "hostsim": p.stderr.read().decode().strip(), "seconds": round(time.time() - t0, 1),
    }
    with open(prefix + "_run.json", "w") as f:
        json.dump(info, f, indent=1)
    return info


# ------------------------------------------------------------------ matrix

BACKENDS = ["lite_f32", "lite_int8", "np_f32", "np_f16", "np_f16acc"]
# fallback deltas: 1e-4 covers ULP-level (another CPU kernel) differences,
# 0.01 covers the emulated fp16 backends (max |diff| 0.0035 / 0.0053 on XL)
PREVENTS = ["none", "grid:3", "grid:4", "grid:5", "grid:7", "grid:10",
            "fallback:0.005", "fallback:0.02",
            "grid:3+fallback:0.0001", "grid:4+fallback:0.0001", "grid:5+fallback:0.0001",
            "grid:3+fallback:0.01", "grid:4+fallback:0.01", "grid:5+fallback:0.01",
            "grid:7+fallback:0.01"]
QUICK_BACKENDS = ["lite_f32", "np_f16"]
QUICK_PREVENTS = ["none", "grid:5", "fallback:0.005", "grid:5+fallback:0.005"]


def name_of(backend, prevent):
    return f"{backend}__{prevent.replace(':', '').replace('+', '_').replace('.', 'p')}"


def cmd_matrix(out_dir, variant, quick):
    os.makedirs(out_dir, exist_ok=True)
    backends = QUICK_BACKENDS if quick else BACKENDS
    prevents = QUICK_PREVENTS if quick else PREVENTS
    rows = []
    for b, pv in itertools.product(backends, prevents):
        if b == "lite_f32" and "fallback" in pv:
            continue   # the reference peer never falls back to itself
        name = name_of(b, pv)
        info = run_game(os.path.join(out_dir, name), b, pv, variant)
        rows.append(info)
        print(f"{name:40s} {info['ticks']:5d} ticks  wave {info['final_wave']:2d}  score {info['final_score']:4d}"
              f"  hp {info['final_hp']:2d}  fallback {info['fallback_row_pct']:6.2f}% rows  {info['seconds']}s", flush=True)
    # determinism check: the reference twice
    info = run_game(os.path.join(out_dir, "lite_f32__none_repeat"), "lite_f32", "none", variant)
    rows.append(info)
    with open(os.path.join(out_dir, "runs.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()), lineterminator="\n")
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {os.path.join(out_dir, 'runs.csv')}")


# ------------------------------------------------------------------ compare

def compare_traces(pa, pb):
    """Closed-loop pair summary (both NPBT)."""
    from determinism import read_bin
    _, A = read_bin(pa)
    _, B = read_bin(pb)
    n = min(len(A), len(B))
    first_out = next((A[i]["tick"] for i in range(n)
                      if not np.array_equal(A[i]["out"], B[i]["out"])), None)
    diff = [i for i in range(n) if A[i]["state"] != B[i]["state"]]
    first_state = A[diff[0]]["tick"] if diff else None
    reconv = bool(diff) and diff[-1] < n - 1 and len(A) == len(B)
    # fire flips while the inputs were still identical
    flips = 0
    same_in = 0
    for i in range(n):
        if A[i]["feat"].shape == B[i]["feat"].shape and np.array_equal(A[i]["feat"], B[i]["feat"]):
            same_in += 1
            flips += int(np.sum((A[i]["out"][:, 2] > FIRE_THRESHOLD) != (B[i]["out"][:, 2] > FIRE_THRESHOLD)))
    # behavioural drift: mean distance between the same enemy slot in both runs
    # (same wave, alive in both), averaged over the whole run
    dists = []
    for i in range(n):
        a, b = A[i], B[i]
        if a["wave"] != b["wave"] or len(a["pos"]) != len(b["pos"]):
            continue
        both = a["alive_mask"] & b["alive_mask"]
        if both.any():
            dists.append(np.linalg.norm(a["pos"][both] - b["pos"][both], axis=1).mean())
    return {
        "mean_drift_px": float(np.mean(dists)) if dists else float("nan"),
        "max_drift_px": float(np.max(dists)) if dists else float("nan"),
        "ticks_a": len(A), "ticks_b": len(B), "identical": first_state is None and len(A) == len(B),
        "first_out_diff_tick": first_out, "first_state_diff_tick": first_state,
        "state_diff_ticks": len(diff), "state_diff_pct": 100.0 * len(diff) / n,
        "reconverged_at_end": reconv, "same_input_ticks": same_in, "fire_flips_same_input": flips,
        "score_a": A[-1]["score"], "score_b": B[-1]["score"], "hp_a": A[-1]["hp"], "hp_b": B[-1]["hp"],
        "wave_a": A[-1]["wave"], "wave_b": B[-1]["wave"],
    }


def cmd_compare(out_dir):
    runs = list(csv.DictReader(open(os.path.join(out_dir, "runs.csv"))))
    by = {(r["backend"], r["prevent"]): r for r in runs}
    rows = []
    for r in runs:
        b, pv = r["backend"], r["prevent"]
        if b == "lite_f32":
            continue
        # the reference peer applies the same grid but needs no fallback
        ref_pv = "+".join(p for p in pv.split("+") if not p.startswith("fallback")) or "none"
        ref = by.get(("lite_f32", ref_pv))
        if ref is None:
            continue
        s = compare_traces(ref["prefix"] + "_trace.bin", r["prefix"] + "_trace.bin")
        rows.append({"backend": b, "prevent": pv, "fallback_row_pct": float(r["fallback_row_pct"]),
                     "fallback_tick_pct": float(r["fallback_tick_pct"]), **s})
    rep = compare_traces(os.path.join(out_dir, "lite_f32__none_trace.bin"),
                         os.path.join(out_dir, "lite_f32__none_repeat_trace.bin"))
    rows.append({"backend": "lite_f32 (repeat)", "prevent": "none", "fallback_row_pct": 0.0,
                 "fallback_tick_pct": 0.0, **rep})
    # behavioural cost of each grid: the reference with and without it
    for r in runs:
        if r["backend"] == "lite_f32" and r["prevent"].startswith("grid"):
            s = compare_traces(by[("lite_f32", "none")]["prefix"] + "_trace.bin", r["prefix"] + "_trace.bin")
            rows.append({"backend": "lite_f32 vs ungridded", "prevent": r["prevent"], "fallback_row_pct": 0.0,
                         "fallback_tick_pct": 0.0, **s})
    path = os.path.join(out_dir, "pairs.csv")
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()), lineterminator="\n")
        w.writeheader()
        w.writerows(rows)
    print(f"{'backend':24s} {'prevent':24s} {'fb%rows':>8s} {'same':>5s} {'1st_out':>8s} {'1st_state':>9s} "
          f"{'diff%':>6s} {'reconv':>6s} {'flips':>5s} {'drift_px':>8s}  ticks a/b  hp a/b")
    for s in rows:
        print(f"{s['backend']:24s} {s['prevent']:24s} {s['fallback_row_pct']:8.2f} {str(s['identical']):>5s} "
              f"{str(s['first_out_diff_tick']):>8s} {str(s['first_state_diff_tick']):>9s} {s['state_diff_pct']:6.1f} "
              f"{str(s['reconverged_at_end']):>6s} {s['fire_flips_same_input']:5d} {s['mean_drift_px']:8.3f}  "
              f"{s['ticks_a']:4d}/{s['ticks_b']:<4d}  {s['hp_a']:2d}/{s['hp_b']:<2d}")
    print(f"wrote {path}")


def main(argv):
    opt = {}
    for key in ("--backend", "--prevent", "--variant"):
        if key in argv:
            i = argv.index(key)
            opt[key] = argv[i + 1]
            del argv[i:i + 2]
    quick = "--quick" in argv
    argv = [a for a in argv if a != "--quick"]
    variant = opt.get("--variant", "xlarge")
    if len(argv) == 2 and argv[0] == "run":
        info = run_game(argv[1], opt["--backend"], opt.get("--prevent", "none"), variant)
        print(json.dumps(info, indent=1))
    elif len(argv) == 2 and argv[0] == "matrix":
        cmd_matrix(argv[1], variant, quick)
    elif len(argv) == 2 and argv[0] == "compare":
        cmd_compare(argv[1])
    else:
        sys.exit(__doc__)


if __name__ == "__main__":
    main(sys.argv[1:])
