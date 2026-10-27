"""Study 3 cross-chip summary: every pulled device run in two tables.

1. Free runs: md5 of every game trace per (model, config, device). Traces
   with the same md5 get the same class letter, so a letter shared across
   devices or configs is a whole-game byte identity.
2. Replays (XL): every REPLAY INPUT of the one shared reference trace (the
   Tab S6 Lite CPU F32 game, which the POCO's and the Tab S7+'s CPU F32
   games equal byte for byte), from both phones and the Mac, compared pairwise over identical
   inputs: values and rows that differ, largest difference, fire flips.

  python3 analysis/study3_crosschip.py [OUT_DIR]   (default analysis/out_study3)

Writes crosschip_free.csv, crosschip_replay_pairs.csv and crosschip.md.
"""

import csv
import glob
import hashlib
import os
import re
import sys
from itertools import combinations

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from determinism import FIRE_THRESHOLD, read_bin  # noqa: E402

ROOT = "runs_study3"
DEVICES = [("s6lite", "Tab S6 Lite (Exynos 9611)"),
           ("poco", "POCO X6 Pro (Dimensity 8300)"),
           ("tab", "Tab S7+ (Snapdragon 865+)"),
           ("a56", "Galaxy A56 (Exynos 1580)"),
           ("iphone12", "iPhone 12 (A14, iOS)"),
           ("iphone17pm", "iPhone 17 Pro Max (A19 Pro, iOS)"),
           ("iphone12_lite217", "iPhone 12, LiteRT 2.17.0"),
           ("fold8", "Galaxy Z Fold 8 (Snapdragon 8 Elite Gen 5)")]
# added after the Mac reference in the replay classes, so the earlier class ids stay
LATE = ("iphone17pm", "iphone12_lite217", "fold8")
CONFIGS = ["cpu_f32", "cpu_int8", "gpu_f32", "gpu_int8", "npu_f32", "npu_int8"]
MODELS = ["xlarge", "tiny"]
# the game every device replayed, per model (pushed as replay_input.bin)
REF_TRACES = {"xlarge": "runs_study3/s6lite/files/run_20261013_223444_game_cpu_xlarge_f32_trace.bin",
              "tiny": "runs_study3/s6lite/files/run_20261014_203009_game_cpu_tiny_f32_trace.bin"}
MAC_REPLAYS = {"xlarge": "replay_mac_lite_{prec}.bin", "tiny": "replay_mac_lite_tiny_{prec}.bin"}
NAME = re.compile(r"run_(\d{8}_\d{6,9}(?:_\d{3})?)_(game|replay)_(cpu|gpu|npu)_(\w+?)_(f32|int8)_(trace|replay)\.bin$")


def md5(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def runs(kind):
    """(device, model, config, path) for every pulled file of one kind."""
    out = []
    for dev, _ in DEVICES:
        for p in sorted(glob.glob(f"{ROOT}/{dev}/files/run_*_{kind}.bin")):
            m = NAME.search(os.path.basename(p))
            if m:
                out.append((dev, m.group(4), f"{m.group(3)}_{m.group(5)}", p))
    return out


def free_table():
    rows, letters = [], {}
    for dev, model, cfg, p in runs("trace"):
        if model not in MODELS:
            continue
        h = md5(p)
        letters.setdefault((model, h), chr(ord("A") + sum(k[0] == model for k in letters)))
        _, ticks = read_bin(p)
        rows.append({"model": model, "config": cfg, "device": dev, "file": p,
                     "md5": h, "class": letters[(model, h)],
                     "ticks": len(ticks), "hp": ticks[-1]["hp"], "score": ticks[-1]["score"]})
    return rows


def replay_outputs(model):
    """label -> (ticks, outputs) for every replay of the model's reference trace."""
    numbered = [(f"{dev} {cfg}" + (f" #{i}" if i else ""), p, dev)
                for dev, m, cfg, p, i in _numbered(runs("replay")) if m == model]
    files = [(lab, p) for lab, p, dev in numbered if dev not in LATE]
    files += [(f"mac {prec}", p) for prec in ("f32", "int8")
              for p in [f"{ROOT}/s6lite/mac_replays/" + MAC_REPLAYS[model].format(prec=prec)]
              if os.path.exists(p)]
    files += [(lab, p) for lab, p, dev in numbered if dev in LATE]
    files.insert(0, (f"s6lite cpu_f32 game", REF_TRACES[model]))
    out = {}
    for label, p in files:
        _, ticks = read_bin(p)
        out[label] = ([t["tick"] for t in ticks], np.concatenate([t["out"] for t in ticks]), p)
    return out


def _numbered(items):
    seen = {}
    for dev, model, cfg, p in items:
        k = (dev, model, cfg)
        seen[k] = seen.get(k, 0) + 1
        yield dev, model, cfg, p, seen[k] - 1


def pair_table(outs):
    rows = []
    for (la, (ta, a, _)), (lb, (tb, b, _)) in combinations(outs.items(), 2):
        if ta != tb or a.shape != b.shape:
            raise ValueError(f"{la} and {lb} are not replays of the same inputs")
        d = np.abs(a.astype(np.float64) - b.astype(np.float64))
        flips = (a[:, 2] > FIRE_THRESHOLD) != (b[:, 2] > FIRE_THRESHOLD)
        rows.append({"a": la, "b": lb, "rows": len(a), "values": a.size,
                     "values_differing": int(np.sum(d > 0)),
                     "rows_differing": int(np.sum(np.any(d > 0, axis=1))),
                     "max_abs_diff": float(d.max()), "fire_flips": int(flips.sum())})
    return rows


def replay_classes(outs):
    """Groups of labels whose outputs are byte-identical, in input order."""
    groups = {}
    for label, (_, a, _) in outs.items():
        groups.setdefault(hashlib.md5(a.tobytes()).hexdigest(), []).append(label)
    return list(groups.values())


def write_csv(path, rows):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()), lineterminator="\n")
        w.writeheader()
        w.writerows(rows)


def markdown(free):
    dev_name = dict(DEVICES)
    lines = ["# Study 3 cross-chip summary (generated by `analysis/study3_crosschip.py`)", "",
             "## Free runs: whole-game trace identity", "",
             "Same letter = byte-identical trace (per model). Runs: number of pulled games.", "",
             "| model | config | " + " | ".join(dev_name[d] for d, _ in DEVICES) + " |",
             "|---|---|" + "---|" * len(DEVICES)]
    for model in MODELS:
        for cfg in CONFIGS:
            cells = []
            for dev, _ in DEVICES:
                rs = [r for r in free if (r["model"], r["config"], r["device"]) == (model, cfg, dev)]
                cls = sorted({r["class"] for r in rs})
                cells.append(f"{'/'.join(cls)} ({len(rs)} runs)" if rs else "n/a")
            if any(c != "n/a" for c in cells):
                lines.append(f"| {model} | {cfg} | " + " | ".join(cells) + " |")
    lines += ["", "## Free runs: outcome", "",
              "Game length in ticks and the player's final hp (every game reached wave 10 "
              "with score 440 unless shown). Runs of one cell all ended the same way unless "
              "several outcomes are listed.", "",
              "| model | config | " + " | ".join(dev_name[d] for d, _ in DEVICES) + " |",
              "|---|---|" + "---|" * len(DEVICES)]
    for model in MODELS:
        for cfg in CONFIGS:
            cells = []
            for dev, _ in DEVICES:
                rs = [r for r in free if (r["model"], r["config"], r["device"]) == (model, cfg, dev)]
                outs = sorted({(r["ticks"], r["hp"], r["score"]) for r in rs})
                cells.append(", ".join(f"{t} t, hp {hp}" + ("" if sc == 440 else f", score {sc}")
                                       for t, hp, sc in outs) if rs else "n/a")
            if any(c != "n/a" for c in cells):
                lines.append(f"| {model} | {cfg} | " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def replay_markdown(model, pairs, outs, prefix):
    classes = replay_classes(outs)
    first_out = next(iter(outs.values()))[1]
    lines = ["", f"## Replays ({model}): output classes over identical inputs", "",
             f"All are outputs over the inputs of `{REF_TRACES[model]}`; "
             f"{first_out.shape[0]:,} enemy rows, {first_out.size:,} values. Runs with byte-identical "
             f"outputs form one class; the full pairwise list is in `{pairs_csv(model)}`.", "",
             "| class | members |", "|---|---|"]
    lines += [f"| {prefix}{i + 1} | {', '.join(m)} |" for i, m in enumerate(classes)]
    lines += ["", "| a | b | values differing | rows differing | max abs diff | fire flips |",
              "|---|---|---|---|---|---|"]
    first = {m[0]: f"{prefix}{i + 1}" for i, m in enumerate(classes)}
    for r in pairs:
        if r["a"] in first and r["b"] in first:
            lines.append(f"| {first[r['a']]} | {first[r['b']]} | {r['values_differing']:,} "
                         f"| {r['rows_differing']:,} | {r['max_abs_diff']:.3g} | {r['fire_flips']} |")
    return "\n".join(lines) + "\n"


def pairs_csv(model):
    return "crosschip_replay_pairs.csv" if model == "xlarge" else f"crosschip_replay_pairs_{model}.csv"


def main(argv):
    out_dir = argv[0] if argv else "analysis/out_study3"
    os.makedirs(out_dir, exist_ok=True)
    free = free_table()
    write_csv(os.path.join(out_dir, "crosschip_free.csv"), free)
    md = markdown(free)
    for model, prefix in (("xlarge", "R"), ("tiny", "T")):
        outs = replay_outputs(model)
        pairs = pair_table(outs)
        write_csv(os.path.join(out_dir, pairs_csv(model)), pairs)
        md += replay_markdown(model, pairs, outs, prefix)
    with open(os.path.join(out_dir, "crosschip.md"), "w") as f:
        f.write(md)
    print(md)


if __name__ == "__main__":
    main(sys.argv[1:])
