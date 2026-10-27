"""Study 3 experiment C: on-device prevention (grid 2^-2 + guard 0.02, portable recompute).

  python3 analysis/study3_prevent_device.py > analysis/out_study3/prevent_device.md

Reads runs_study3/{s6lite,tab}_prevent/files (2026-10-18 campaign) and reports:
1. whole-game identity of the prevented games across backends and tablets, and the
   control game against the study's CPU F32 game of the same device;
2. a Mac peer along the device trajectory: the Mac's own outputs (LiteRT XNNPACK, and an
   emulated fp16 peer) on the recorded inputs, after the same prevention with the portable
   kernel (tools/portable_policy/pp_cli), against the outputs the tablets' game received;
3. the bit check: the S6 Lite replay with every row on the portable kernel against pp_cli;
4. cost from ticks.csv (prevent_rows, prevent_ms; warmup ticks excluded).
"""
import csv
import glob
import hashlib
import os
import subprocess
import sys
import tempfile
from collections import defaultdict

import numpy as np

sys.path.insert(0, "analysis")
from cosim import Prevent  # noqa: E402
from determinism import read_bin  # noqa: E402

SPEC = "grid:2+fallback:0.02"
DEVS = {"s6lite": "Tab S6 Lite", "tab": "Tab S7+"}
REF_GAME = {"s6lite": "runs_study3/s6lite/files/run_20261013_223444_game_cpu_xlarge_f32_trace.bin"}
PP_CLI = "tools/portable_policy/pp_cli"
MODELS = "app/src/main/assets/models"
VARIANT = {"xlarge": "XL", "tiny": "tiny"}


def md5(path):
    return hashlib.md5(open(path, "rb").read()).hexdigest()


def games(dev):
    out = []
    for tr in sorted(glob.glob(f"runs_study3/{dev}_prevent/files/run_*_game_*_trace.bin")):
        rid = os.path.basename(tr)[4:-10]
        ticks = tr[:-10] + "_ticks.csv"
        rows = list(csv.DictReader(open(ticks)))
        _, back, variant, prec = rid.split("_")[2:6]
        out.append(dict(dev=dev, rid=rid, backend=back, variant=variant, prec=prec,
                        prevent=rows[0]["prevent"], trace=tr, md5=md5(tr), rows=rows))
    return out


def pp_outputs(variant, feat):
    with tempfile.TemporaryDirectory() as d:
        np.ascontiguousarray(feat, "<f4").tofile(f"{d}/in.f32")
        subprocess.run([PP_CLI, f"{MODELS}/policy_{variant}_f32.nppw", f"{d}/in.f32", f"{d}/out.f32"],
                       check=True, capture_output=True)
        return np.fromfile(f"{d}/out.f32", "<f4").reshape(-1, 3)


def trajectory_check(trace, variant, backends):
    """Mac peers on the recorded inputs: does prevention give them the device's outputs?"""
    from policy_backends import make_backend
    _, T = read_bin(trace)
    T = [t for t in T if t["alive"] > 0]
    feat = np.concatenate([t["feat"] for t in T])
    dev_out = np.concatenate([t["out"] for t in T])
    common = pp_outputs(variant, feat)
    p = Prevent(SPEC)
    res = {}
    for name in backends:
        b = make_backend(name, variant)
        raw = np.concatenate([b.run(t["feat"], t["batch"]) for t in T])
        got, n = p.apply(raw, common)
        bits = np.any(got.view(np.uint32) != dev_out.view(np.uint32), axis=1)
        vals = np.any(got != dev_out, axis=1)
        res[name] = (len(feat), int(vals.sum()), int((bits & ~vals).sum()), 100.0 * n / len(feat),
                     float(np.abs(raw - common).max()))
    on_grid = bool(np.all(dev_out * 4 == np.round(dev_out * 4)))
    return res, on_grid


def tick_stats(rows):
    r = [x for x in rows if x["warmup"] == "0"]
    ms = np.array([float(x["prevent_ms"]) for x in r])
    pr = np.array([int(x["prevent_rows"]) for x in r])
    alive = np.array([int(x["alive"]) for x in r])
    inf = np.array([float(x["policy_infer_ms"]) for x in r])
    tot = np.array([float(x["tick_total_ms"]) for x in r])
    miss = np.array([int(x["deadline_missed"]) for x in r])
    return dict(ticks=len(r), rows=int(alive.sum()), rec=int(pr.sum()), ms=ms, inf=inf, tot=tot,
                miss=int(miss.sum()), alive_mean=float(alive.mean()))


def main():
    allg = [g for d in DEVS for g in games(d)]
    print("# Experiment C: prevention on the devices (`analysis/study3_prevent_device.py`)\n")
    print(f"Campaign 2026-10-18 17:11-17:43 WIB, spec `{SPEC}` (grid 2^-2, guard 0.02, portable "
          "recompute), both tablets on battery. Games: 3 per configuration.\n")

    print("## 1. Whole-game identity\n")
    print("| device | model | backend | prevention | games | distinct traces | ticks | end |")
    print("|---|---|---|---|---|---|---|---|")
    by = defaultdict(list)
    for g in allg:
        by[(g["dev"], g["variant"], g["backend"], g["prevent"])].append(g)
    for (dev, v, b, pv), gs in sorted(by.items()):
        last = gs[0]["rows"][-1]
        print(f"| {DEVS[dev]} | {VARIANT[v]} | {b.upper()} F32 | {pv} | {len(gs)} | "
              f"{len({g['md5'] for g in gs})} | {len(gs[0]['rows'])} | wave {last['wave']} "
              f"hp {read_bin(gs[0]['trace'])[1][-1]['hp']} |")
    print()
    for v in VARIANT:
        prev = [g for g in allg if g["variant"] == v and g["prevent"] == SPEC]
        ids = sorted({g["md5"] for g in prev})
        T0 = read_bin(prev[0]["trace"])[1]
        o0 = np.concatenate([t["out"] for t in T0 if t["alive"]])
        same_state = same_val = True
        signed = []
        for g in prev[1:]:
            T = read_bin(g["trace"])[1]
            same_state &= len(T) == len(T0) and all(a["state"] == b["state"] and np.array_equal(a["feat"], b["feat"])
                                                   for a, b in zip(T, T0))
            o = np.concatenate([t["out"] for t in T if t["alive"]])
            same_val &= o.shape == o0.shape and bool(np.array_equal(o, o0))
            if o.shape == o0.shape:
                signed.append(int((o.view(np.uint32) != o0.view(np.uint32)).sum()))
        cfgs = sorted({(DEVS[g["dev"]], g["backend"].upper()) for g in prev})
        print(f"- {VARIANT[v]}, prevented: {len(prev)} games over {len(cfgs)} device/backend pairs "
              f"({', '.join(f'{d} {b}' for d, b in cfgs)}): {len(ids)} distinct trace file(s) "
              f"{', '.join(i[:8] for i in ids)}. Inputs and state identical on every tick: {same_state}; "
              f"outputs equal in value on every row: {same_val}; output values differing only in the "
              f"sign of a zero (-0 vs +0), per game against {DEVS[prev[0]['dev']]} {prev[0]['backend'].upper()}: "
              f"{min(signed)} to {max(signed)}.")
    for dev, ref in REF_GAME.items():
        ctrl = [g for g in allg if g["dev"] == dev and g["prevent"] == "off"]
        print(f"- Control ({DEVS[dev]}, XL CPU F32, prevention off): "
              f"{'byte-identical' if all(g['md5'] == md5(ref) for g in ctrl) else 'DIFFERENT'} "
              f"to the study game {os.path.basename(ref)[4:19]} ({len(ctrl)} games).")
    ctrl_all = {g["md5"] for g in allg if g["prevent"] == "off"}
    print(f"- Controls on both tablets: {len(ctrl_all)} distinct trace file(s).\n")

    print("## 2. Mac peers along the device trajectory\n")
    print("Each recorded input row is run through the Mac backend, prevented with the same rule "
          "(portable kernel for guarded rows), and compared with the output the tablet's game "
          "received (Tab S6 Lite CPU game).\n")
    print("| model | Mac peer | rows | rows differing in value | rows differing only in the sign of a zero "
          "| recomputed | largest raw difference vs portable |")
    print("|---|---|---|---|---|---|---|")
    grid_notes = []
    for v in VARIANT:
        g = next(g for g in allg if g["variant"] == v and g["prevent"] == SPEC)
        res, on_grid = trajectory_check(g["trace"], v, ["lite_f32", "np_f16"])
        for name, (n, d, z, rec, mx) in res.items():
            label = {"lite_f32": "LiteRT CPU F32 (XNNPACK)", "np_f16": "emulated fp16 storage"}[name]
            print(f"| {VARIANT[v]} | {label} | {n:,} | {d} | {z} | {rec:.1f} % | {mx:.2g} |")
        grid_notes.append(f"{VARIANT[v]}: {on_grid}")
    print(f"\nDevice outputs all on the 2^-2 grid: {'; '.join(grid_notes)}.\n")

    print("## 3. Bit check of the portable kernel on the S6 Lite\n")
    rp = glob.glob("runs_study3/s6lite_prevent/files/run_*_replay_*_replay.bin")
    for path in rp:
        _, R = read_bin(path)
        _, T = read_bin(REF_GAME["s6lite"])
        assert len(R) == len(T)
        feat = np.concatenate([t["feat"] for t in T if t["alive"] > 0])
        dev = np.concatenate([r["out"] for r in R if r["alive"] > 0])
        mac = pp_outputs("xlarge", feat)
        same = int(np.all(dev.view(np.uint32) == mac.view(np.uint32), axis=1).sum())
        print(f"- {os.path.basename(path)}: {same:,} of {len(feat):,} rows bit-identical to "
              f"pp_cli on the Mac (M5, clang -O2 -ffp-contract=off); largest difference "
              f"{float(np.abs(dev - mac).max()):.3g}.\n")

    print("## 4. Cost (ticks after warmup)\n")
    print("| device | model | backend | prevention | rows recomputed | prevent ms/tick median | p99 | max "
          "| ms per recomputed row | inference ms/tick median | tick ms median | missed deadlines |")
    print("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for (dev, v, b, pv), gs in sorted(by.items()):
        s = [tick_stats(g["rows"]) for g in gs]
        ms = np.concatenate([x["ms"] for x in s])
        rows, rec = sum(x["rows"] for x in s), sum(x["rec"] for x in s)
        per = ms.sum() / rec if rec else float("nan")
        print(f"| {DEVS[dev]} | {VARIANT[v]} | {b.upper()} | {pv} | {100.0 * rec / rows:.1f} % "
              f"| {np.median(ms):.3f} | {np.percentile(ms, 99):.3f} | {ms.max():.2f} | "
              f"{per:.4f} | {np.median(np.concatenate([x['inf'] for x in s])):.2f} | "
              f"{np.median(np.concatenate([x['tot'] for x in s])):.2f} | "
              f"{sum(x['miss'] for x in s)} of {sum(x['ticks'] for x in s)} |")
    print(f"\nMean alive rows per tick: {np.mean([tick_stats(g['rows'])['alive_mean'] for g in allg]):.1f}. "
          "prevent_ms times the whole prevention step (edge tests and grid rounding on every row, "
          "plus the portable recompute of the guarded rows).")

    print("\n## 5. Validation\n")
    first = {}
    for g in allg:
        first.setdefault(g["md5"], g)
    ok = 0
    with tempfile.TemporaryDirectory() as d:
        for h, g in first.items():
            subprocess.run([sys.executable, "tools/hostsim/validate_trace.py", g["trace"], f"{d}/{h}"],
                           check=True, capture_output=True)
            ok += md5(f"{d}/{h}_trace.bin") == h
    print(f"Re-simulated on the Mac (tools/hostsim) from the recorded outputs: {ok} of {len(first)} "
          "distinct trace files reproduced byte for byte.")


if __name__ == "__main__":
    main()
