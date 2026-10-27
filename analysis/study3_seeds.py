"""Study 3: the closed-loop comparison over many game seeds (Mac co-simulation).

The device study plays one game seed per model. Here the host build of the game
(tools/hostsim, HOSTSIM_SEED) plays N seeds per model, and every seed is played
by several peers through analysis/cosim.py:

  float32, no prevention:  lite_f32 (LiteRT XNNPACK, the reference), np_f32 (a second
                           CPU kernel, ULP-level), np_f16 (fp16 values, fp32 sums),
                           np_f16acc (fp16 values and sums)
  float32, prevented:      the same four with grid 2^-2 and guard 0.02, guarded rows
                           recomputed by the portable kernel (as in the app)
  int8:                    lite_int8_ref (TFLite reference kernels, two roundings, the
                           Android CPU class) and lite_int8 (one rounding, the Apple class)

Each peer is compared with the reference of its group by the per-tick state hashes:
identical game, first differing tick, and outcome (length, kills, hp). Seed index 0
is the study's own game (HOSTSIM_SEED unset). Bit results do not depend on machine
load, so this needs no quiet Mac.

  python3 analysis/study3_seeds.py [--seeds 100] [--workers 4] [--tmp DIR]
  -> analysis/out_study3/seeds/seeds.csv, seeds.md
"""
import argparse
import csv
import math
import os
import shutil
import sys
import tempfile
from multiprocessing import Pool

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
OUT_DIR = os.path.join(HERE, "out_study3", "seeds")
GUARD = "grid:2+fallback:0.02"
FLOAT_PEERS = ["lite_f32", "np_f32", "np_f16", "np_f16acc"]
INT8_PEERS = ["lite_int8_ref", "lite_int8"]
FIELDS = ["model", "seed_index", "seed", "group", "peer", "ticks", "wave", "kills", "hp",
          "guarded_row_pct", "identical", "first_state_diff_tick", "outcome_changed", "hp_changed"]


def seed_of(index):
    """Index 0: the study game; otherwise a well-mixed nonzero xorshift seed."""
    return None if index == 0 else ((index * 2654435761) & 0xFFFFFFFF) | 1


def state_hashes(prefix):
    with open(prefix + "_ticks.csv") as f:
        return [r["state_hash"] for r in csv.DictReader(f)]


def play(task):
    model, index, tmp_root = task
    import cosim  # noqa: E402  (imports TensorFlow lazily per worker)
    seed = seed_of(index)
    tmp = tempfile.mkdtemp(prefix=f"{model}_{index}_", dir=tmp_root)
    runs = [("f32", p, "none") for p in FLOAT_PEERS] + [("f32_guard", p, GUARD) for p in FLOAT_PEERS] \
        + [("int8", p, "none") for p in INT8_PEERS]
    played = {}
    for group, peer, spec in runs:
        prefix = os.path.join(tmp, f"{group}_{peer}")
        info = cosim.run_game(prefix, peer, spec, model, ref_name="pp", seed=seed)
        played[(group, peer)] = (info, state_hashes(prefix))
    rows = []
    for (group, peer), (info, hashes) in played.items():
        ref_info, ref_hashes = played[(group, FLOAT_PEERS[0] if group != "int8" else INT8_PEERS[0])]
        first = next((t for t, (a, b) in enumerate(zip(hashes, ref_hashes)) if a != b), None)
        if first is None and len(hashes) != len(ref_hashes):
            first = min(len(hashes), len(ref_hashes))
        out = (info["ticks"], info["final_score"], info["final_hp"])
        ref_out = (ref_info["ticks"], ref_info["final_score"], ref_info["final_hp"])
        rows.append({"model": model, "seed_index": index, "seed": "" if seed is None else seed,
                     "group": group, "peer": peer, "ticks": info["ticks"], "wave": info["final_wave"],
                     "kills": info["final_score"], "hp": info["final_hp"],
                     "guarded_row_pct": round(info["fallback_row_pct"], 3),
                     "identical": int(first is None), "first_state_diff_tick": "" if first is None else first,
                     "outcome_changed": int(out != ref_out), "hp_changed": int(info["final_hp"] != ref_info["final_hp"])})
    shutil.rmtree(tmp)
    return rows


def lower_priority():
    """Workers run at the lowest priority; already there when started under nice."""
    try:
        os.setpriority(os.PRIO_PROCESS, 0, 19)
    except OSError:
        pass


def wilson(k, n, z=1.96):
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def summary(rows, n_seeds):
    lines = ["# Study 3: closed-loop comparison over many seeds (`analysis/study3_seeds.py`)", "",
             f"Host co-simulation, {n_seeds} seeds per model (index 0 = the study's game). Each peer is "
             "compared with the reference of its group (float32: LiteRT XNNPACK; int8: TFLite reference "
             "kernels) by per-tick state hashes. Outcome = length, kills and final hp. 95 % Wilson intervals.", ""]
    for model in ("xlarge", "tiny"):
        ms = [r for r in rows if r["model"] == model]
        ref = [r for r in ms if r["group"] == "f32" and r["peer"] == "lite_f32"]
        if not ref:
            continue
        ticks = sorted(r["ticks"] for r in ref)
        lines += [f"## {model}", "",
                  f"Reference games: {len(ref)} seeds, {ticks[0]} to {ticks[-1]} ticks, final hp "
                  f"{min(r['hp'] for r in ref)} to {max(r['hp'] for r in ref)}, waves reached "
                  f"{min(r['wave'] for r in ref)} to {max(r['wave'] for r in ref)}.", "",
                  "| group | peer | seeds | games differing | outcome changed (95 % CI) | hp changed | "
                  "first differing tick, median | guarded rows |", "|---|---|---|---|---|---|---|---|"]
        for group, peers in (("f32", FLOAT_PEERS[1:]), ("f32_guard", FLOAT_PEERS), ("int8", INT8_PEERS[1:])):
            for peer in peers:
                rs = [r for r in ms if r["group"] == group and r["peer"] == peer]
                n = len(rs)
                diff = [r for r in rs if not r["identical"]]
                k = sum(r["outcome_changed"] for r in rs)
                lo, hi = wilson(k, n)
                firsts = sorted(r["first_state_diff_tick"] for r in diff)
                med = firsts[len(firsts) // 2] if firsts else ""
                g = sum(r["guarded_row_pct"] for r in rs) / n if group == "f32_guard" and n else None
                lines.append(f"| {group} | {peer} | {n} | {len(diff)} | {k} ({100 * k / n:.0f} %, "
                             f"{100 * lo:.0f} to {100 * hi:.0f} %) | {sum(r['hp_changed'] for r in rs)} | {med} | "
                             + (f"{g:.1f} %" if g is not None else "") + " |")
        lines.append("")
    return "\n".join(lines) + "\n"


def main(argv):
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=100)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--tmp", default=None)
    a = ap.parse_args(argv)
    os.makedirs(OUT_DIR, exist_ok=True)
    tmp_root = tempfile.mkdtemp(prefix="seeds_", dir=a.tmp)
    tasks = [(m, i, tmp_root) for i in range(a.seeds) for m in ("xlarge", "tiny")]
    rows = []
    csv_path = os.path.join(OUT_DIR, "seeds.csv")
    with open(csv_path, "w", newline="") as f, Pool(a.workers, initializer=lower_priority) as pool:
        w = csv.DictWriter(f, fieldnames=FIELDS, lineterminator="\n")
        w.writeheader()
        for k, part in enumerate(pool.imap_unordered(play, tasks), 1):
            w.writerows(part)
            f.flush()
            rows += part
            print(f"{k}/{len(tasks)} {part[0]['model']} seed {part[0]['seed_index']}", flush=True)
    shutil.rmtree(tmp_root, ignore_errors=True)
    rows.sort(key=lambda r: (r["model"], r["seed_index"], r["group"], r["peer"]))
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS, lineterminator="\n")
        w.writeheader()
        w.writerows(rows)
    md = summary(rows, a.seeds)
    with open(os.path.join(OUT_DIR, "seeds.md"), "w") as f:
        f.write(md)
    print(md)


if __name__ == "__main__":
    main(sys.argv[1:])
