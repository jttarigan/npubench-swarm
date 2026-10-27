"""Determinism study (study 3): compare enemy-AI outputs and game states
between runs recorded on different backends, precisions or devices.

Inputs are the binary files written by the app (layout in
app/src/main/cpp/src/trace.h):
  run_<id>_trace.bin   NPBT, free-running game: inputs, outputs, state per tick
  run_<id>_replay.bin  NPBR, fixed-input replay of a trace: outputs per tick

Commands
  free A_trace.bin B_trace.bin
      Two free-running games. First tick where outputs / game state differ,
      output differences while the inputs were still identical, drift of
      enemy positions afterwards, and the outcome of each run.
  outputs A.bin B.bin [--grid]
      Two runs over the SAME inputs (a trace and its replay, or two replays
      of one trace). Bit-exact rate, largest differences, fire-decision
      flips. --grid adds the rounding-prevention table.
  matrix --ref REF.bin OUT.csv X.bin [Y.bin ...]
      outputs-comparison of each X against REF, one CSV row per file.
"""

import csv
import struct
import sys

import numpy as np

FIRE_THRESHOLD = 0.35          # kPolicyFireThreshold in common.h
TICK_HDR = struct.Struct("<IHHHH")


# ------------------------------------------------------------------ reading

def read_bin(path):
    """Returns (kind, ticks). kind is 'NPBT' or 'NPBR'. Each tick is a dict
    with tick, wave, batch, alive, out (alive x outDim float32) and, for
    NPBT, feat, idx, pos (E x 2), alive_mask, hp, score, bullets, rng and
    the raw state bytes."""
    with open(path, "rb") as f:
        data = f.read()
    kind = data[:4].decode("ascii", "replace")
    if kind not in ("NPBT", "NPBR"):
        raise ValueError(f"{path}: not an NPBT/NPBR file")
    ver, fd, od = struct.unpack_from("<III", data, 4)
    if ver != 1:
        raise ValueError(f"{path}: unsupported version {ver}")
    off, ticks = 16, []
    while off < len(data):
        tick, wave, batch, alive, _ = TICK_HDR.unpack_from(data, off)
        off += TICK_HDR.size
        t = {"tick": tick, "wave": wave, "batch": batch, "alive": alive}
        if kind == "NPBT":
            t["feat"] = np.frombuffer(data, "<f4", alive * fd, off).reshape(alive, fd)
            off += 4 * alive * fd
        t["out"] = np.frombuffer(data, "<f4", alive * od, off).reshape(alive, od)
        off += 4 * alive * od
        if kind == "NPBT":
            t["idx"] = np.frombuffer(data, "<u2", alive, off)
            off += 2 * alive
            s0 = off
            (E,) = struct.unpack_from("<H", data, off)
            off += 2
            rec = np.frombuffer(data, np.dtype([("x", "<f4"), ("y", "<f4"), ("a", "u1")]),
                                E, off)
            off += 9 * E
            hp, score, bullets, rng = struct.unpack_from("<hiHI", data, off)
            off += 12
            t.update(pos=np.stack([rec["x"], rec["y"]], 1), alive_mask=rec["a"] == 1,
                     hp=hp, score=score, bullets=bullets, rng=rng, state=data[s0:off])
        ticks.append(t)
    return kind, ticks


# ------------------------------------------------------------------ helpers

def ordered_bits(x):
    """float32 -> integers whose difference counts ULPs (sign-aware)."""
    i = np.asarray(x, np.float32).view(np.int32).astype(np.int64)
    return np.where(i < 0, -(i & 0x7FFFFFFF), i)


def output_diff(a, b):
    """Compare two equally shaped output arrays (n x 3)."""
    d = np.abs(a.astype(np.float64) - b.astype(np.float64))
    ulp = np.abs(ordered_bits(a) - ordered_bits(b))
    fire_flip = (a[:, 2] > FIRE_THRESHOLD) != (b[:, 2] > FIRE_THRESHOLD)
    return d, ulp, fire_flip


def summarize_outputs(pairs):
    """pairs: list of (tick, outA, outB) over identical inputs."""
    if not pairs:
        return None
    a = np.concatenate([p[1] for p in pairs])
    b = np.concatenate([p[2] for p in pairs])
    d, ulp, flip = output_diff(a, b)
    exact_rows = np.all(d == 0, axis=1)
    first_diff = next((t for t, x, y in pairs if not np.array_equal(x, y)), None)
    return {
        "ticks": len(pairs),
        "rows": len(a),
        "values": a.size,
        "bitexact_value_pct": 100.0 * np.mean(d == 0),
        "bitexact_row_pct": 100.0 * np.mean(exact_rows),
        "bitexact_tick_pct": 100.0 * np.mean([np.array_equal(x, y) for _, x, y in pairs]),
        "max_abs_diff": float(d.max()),
        "mean_abs_diff": float(d.mean()),
        "max_ulp": int(ulp.max()),
        "median_ulp_nonzero": float(np.median(ulp[ulp > 0])) if np.any(ulp > 0) else 0.0,
        "fire_flips": int(flip.sum()),
        "first_diff_tick": first_diff,
    }


def grid_table(pairs, ks=range(2, 24, 2)):
    """Rounding prevention: snap outputs to a 2^-k grid before use. Rows:
    (k, step, values still mismatching, fire decisions still flipping)."""
    a = np.concatenate([p[1] for p in pairs]).astype(np.float64)
    b = np.concatenate([p[2] for p in pairs]).astype(np.float64)
    rows = []
    for k in ks:
        step = 2.0 ** -k
        ra, rb = np.round(a / step) * step, np.round(b / step) * step
        mism = int(np.sum(ra != rb))
        flips = int(np.sum((ra[:, 2] > FIRE_THRESHOLD) != (rb[:, 2] > FIRE_THRESHOLD)))
        rows.append((k, step, mism, flips))
    return rows


def print_summary(s, title):
    print(f"\n{title}")
    if s is None:
        print("  (no comparable ticks)")
        return
    for k, v in s.items():
        print(f"  {k:22s} {v:.6g}" if isinstance(v, float) else f"  {k:22s} {v}")


# ------------------------------------------------------------------ commands

def cmd_free(pa, pb):
    ka, A = read_bin(pa)
    kb, B = read_bin(pb)
    if ka != "NPBT" or kb != "NPBT":
        sys.exit("free needs two NPBT trace files")
    n = min(len(A), len(B))
    first_out = next((i for i in range(n) if not np.array_equal(A[i]["out"], B[i]["out"])), None)
    diff_ticks = [i for i in range(n) if A[i]["state"] != B[i]["state"]]
    first_state = diff_ticks[0] if diff_ticks else None
    # outputs on ticks whose inputs are still bit-identical
    same_in = [(A[i]["tick"], A[i]["out"], B[i]["out"]) for i in range(n)
               if A[i]["feat"].shape == B[i]["feat"].shape
               and np.array_equal(A[i]["feat"], B[i]["feat"])]
    print(f"A: {pa}\nB: {pb}")
    print(f"ticks A {len(A)}  B {len(B)}")
    print(f"first output difference: tick {A[first_out]['tick'] if first_out is not None else 'none'}")
    print(f"first state difference:  tick {A[first_state]['tick'] if first_state is not None else 'none'}")
    if diff_ticks:
        healed = diff_ticks[-1] < n - 1
        print(f"state differs on {len(diff_ticks)} ticks; last at tick {A[diff_ticks[-1]]['tick']}"
              f" -> {'RECONVERGED (identical again afterwards)' if healed else 'still split at the end'}")
    print_summary(summarize_outputs(same_in), "outputs while inputs were identical")
    if first_state is not None:
        print("\ndrift after the split (same enemy slot alive in both, same wave):")
        print("  ticks_after    mean_px     max_px  alive_mismatch")
        for dt in (0, 1, 10, 60, 300, 600, 1200):
            i = first_state + dt
            if i >= n:
                break
            a, b = A[i], B[i]
            if a["wave"] != b["wave"] or len(a["pos"]) != len(b["pos"]):
                print(f"  {dt:11d}  (runs are in different waves or wave sizes)")
                continue
            both = a["alive_mask"] & b["alive_mask"]
            dist = np.linalg.norm(a["pos"][both] - b["pos"][both], axis=1)
            mism = int(np.sum(a["alive_mask"] != b["alive_mask"]))
            print(f"  {dt:11d}  {dist.mean() if dist.size else 0:9.3g}  "
                  f"{dist.max() if dist.size else 0:9.3g}  {mism}")
    print("\noutcome         A        B")
    for key in ("tick", "wave", "score", "hp"):
        print(f"  final {key:6s} {A[-1][key]:8d} {B[-1][key]:8d}")
    wa = [t["tick"] for p, t in zip(A, A[1:]) if t["wave"] != p["wave"]]
    wb = [t["tick"] for p, t in zip(B, B[1:]) if t["wave"] != p["wave"]]
    print(f"  wave starts A {wa}\n  wave starts B {wb}")


def same_input_pairs(pa, pb):
    ka, A = read_bin(pa)
    kb, B = read_bin(pb)
    # NPBR has no inputs: it is aligned to its source trace by tick index
    ia = {t["tick"]: t for t in A}
    pairs, skipped = [], 0
    for t in B:
        u = ia.get(t["tick"])
        if u is None or u["out"].shape != t["out"].shape:
            skipped += 1
            continue
        if "feat" in u and "feat" in t and not np.array_equal(u["feat"], t["feat"]):
            skipped += 1
            continue
        pairs.append((t["tick"], u["out"], t["out"]))
    return pairs, skipped


def cmd_outputs(pa, pb, grid):
    pairs, skipped = same_input_pairs(pa, pb)
    print(f"A: {pa}\nB: {pb}\ncompared ticks {len(pairs)}, skipped {skipped}")
    print_summary(summarize_outputs(pairs), "outputs over identical inputs")
    if grid and pairs:
        print("\nrounding to a 2^-k grid before use:")
        print("   k        step   values_differing  fire_flips")
        for k, step, mism, flips in grid_table(pairs):
            print(f"  {k:2d}  {step:10.3g}  {mism:16d}  {flips:10d}")


def cmd_matrix(ref, out_csv, files):
    cols = None
    with open(out_csv, "w", newline="") as f:
        w = None
        for p in files:
            pairs, skipped = same_input_pairs(ref, p)
            s = summarize_outputs(pairs) or {}
            row = {"ref": ref, "file": p, "skipped_ticks": skipped, **s}
            if w is None:
                cols = list(row.keys())
                w = csv.DictWriter(f, fieldnames=cols, lineterminator="\n")
                w.writeheader()
            w.writerow({k: row.get(k, "") for k in cols})
            print(f"{p}: bitexact {s.get('bitexact_value_pct', float('nan')):.3f}%  "
                  f"max_ulp {s.get('max_ulp', '-')}  fire_flips {s.get('fire_flips', '-')}")
    print(f"wrote {out_csv}")


def main(argv):
    if len(argv) >= 3 and argv[0] == "free":
        cmd_free(argv[1], argv[2])
    elif len(argv) >= 3 and argv[0] == "outputs":
        cmd_outputs(argv[1], argv[2], "--grid" in argv)
    elif len(argv) >= 5 and argv[0] == "matrix" and argv[1] == "--ref":
        cmd_matrix(argv[2], argv[3], argv[4:])
    else:
        sys.exit(__doc__)


if __name__ == "__main__":
    main(sys.argv[1:])
