"""How close the fire output lies to its threshold, per model (Section 4.4).

  python3 analysis/study3_near_threshold.py   -> analysis/out_study3/near_threshold.md

Over the rows of the study game of each model (the Tab S6 Lite CPU float32 trace that
every replay uses as input), the share of rows whose fire value lies within 0.01, 0.001
and 0.02 of the 0.35 threshold, and the shares below -0.9 and above 0.9. The draft
quotes the 0.01 column: tiny 0.29 % against XL 0.03 %.
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from determinism import FIRE_THRESHOLD, read_bin  # noqa: E402

ROOT = os.path.dirname(HERE)
GAMES = {
    "XL": "runs_study3/s6lite/files/run_20261013_223444_game_cpu_xlarge_f32_trace.bin",
    "tiny": "runs_study3/s6lite/files/run_20261014_203009_game_cpu_tiny_f32_trace.bin",
}
WIDTHS = (0.001, 0.01, 0.02)
OUT = os.path.join(ROOT, "analysis", "out_study3", "near_threshold.md")


def main():
    lines = ["# Fire output near its threshold (`analysis/study3_near_threshold.py`)", "",
             f"Rows of each model's study game (Tab S6 Lite CPU float32 trace); fire threshold {FIRE_THRESHOLD}.", "",
             "| model | rows | " + " | ".join(f"within {w}" for w in WIDTHS) + " | below -0.9 | above 0.9 |",
             "|---|---|" + "---|" * (len(WIDTHS) + 2)]
    for model, rel in GAMES.items():
        _, ticks = read_bin(os.path.join(ROOT, rel))
        fire = np.concatenate([t["out"][:, 2] for t in ticks if len(t["out"])])
        near = [100 * np.mean(np.abs(fire - FIRE_THRESHOLD) < w) for w in WIDTHS]
        lines.append(f"| {model} | {len(fire):,} | " + " | ".join(f"{n:.3f} %" for n in near)
                     + f" | {100 * np.mean(fire < -0.9):.1f} % | {100 * np.mean(fire > 0.9):.1f} % |")
    with open(OUT, "w") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
