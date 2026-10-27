"""When each accelerator game leaves the drift band of Figure 2 (Section 4.4).

  python3 analysis/study3_drift_band.py   -> analysis/out_study3/drift_band.md

Same measure as the figure (`study3_figures.drift_per_second`): mean distance between
matching live enemies of the first accelerator game and the first CPU game of the same
device, model and precision, averaged per second of the game. For every drawn path:
the largest per-second drift, and the first second above 0.03 px (the band the text
gives for the GPU games), with the wave starts of the CPU game for reference.
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import study3_figures as sf  # noqa: E402

BAND = 0.03
CFGS = {
    "Mali-G72 GPU F32 (Tab S6 Lite)": ("s6lite", "gpu_xlarge_f32", "cpu_xlarge_f32"),
    "Mali-G615 GPU F32 (POCO)": ("poco", "gpu_xlarge_f32", "cpu_xlarge_f32"),
    "MediaTek NPU F32 (POCO)": ("poco", "npu_xlarge_f32", "cpu_xlarge_f32"),
    "Xclipse 540 GPU F32 (A56)": ("a56", "gpu_xlarge_f32", "cpu_xlarge_f32"),
    "Adreno 650 GPU F32 (Tab S7+)": ("tab", "gpu_xlarge_f32", "cpu_xlarge_f32"),
    "Adreno 840 GPU F32 (Fold 8)": ("fold8", "gpu_xlarge_f32", "cpu_xlarge_f32"),
    "Qualcomm NNAPI F32 (Tab S7+)": ("tab", "npu_xlarge_f32", "cpu_xlarge_f32"),
    "Qualcomm NNAPI INT8 (Tab S7+)": ("tab", "npu_xlarge_int8", "cpu_xlarge_int8"),
    "Metal GPU F32 (iPhone 12)": ("iphone12", "gpu_xlarge_f32", "cpu_xlarge_f32"),
    "Core ML F32 (iPhone 12)": ("iphone12", "npu_xlarge_f32", "cpu_xlarge_f32"),
}
OUT = os.path.join(os.path.dirname(HERE), "analysis", "out_study3", "drift_band.md")


def main():
    lines = ["# Closed-loop drift per path (`analysis/study3_drift_band.py`)", "",
             "First accelerator game against the first CPU game of the same device, XL model, "
             "per-second mean distance between matching live enemies (the Figure 2 measure).", "",
             f"| path | seconds compared | min | max | first second above {BAND} px | seconds above {BAND} px (value) |",
             "|---|---|---|---|---|---|"]
    waves = None
    for label, (dev, acc, cpu) in CFGS.items():
        y, w = sf.drift_per_second(sf._first_trace(dev, acc), sf._first_trace(dev, cpu))
        waves = waves or w
        ok = np.isfinite(y)
        above = np.flatnonzero(ok & (y > BAND))
        first = f"{above[0]} s" if len(above) else "never"
        runs, i = [], 0
        while i < len(above):            # consecutive seconds as one range, with the peak value
            j = i
            while j + 1 < len(above) and above[j + 1] == above[j] + 1:
                j += 1
            span = f"{above[i]}" if i == j else f"{above[i]} to {above[j]}"
            runs.append(f"{span} s ({np.nanmax(y[above[i]:above[j] + 1]):.3g})")
            i = j + 1
        lines.append(f"| {label} | {ok.sum()} | {np.nanmin(y):.4f} | {np.nanmax(y):.3f} | {first} | "
                     f"{'; '.join(runs) or 'none'} |")
    lines += ["", "Wave starts of the CPU game (s): " + ", ".join(f"{t:.1f}" for t in waves) + "."]
    with open(OUT, "w") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
