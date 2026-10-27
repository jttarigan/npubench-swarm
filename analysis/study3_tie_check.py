"""Two claims of Table 4 and Supplement S4 traced to the int8 emulation (audit items C8, C2).

  python3 analysis/study3_tie_check.py   -> analysis/out_study3/tie_check.md

C8  The MediaTek NPU int8 outputs differ from the two-rounding (gemmlowp) emulation,
    which reproduces the Android CPU int8 outputs exactly, in one value. Is that value a
    rounding tie? For its row every requantization is recomputed in exact integers and
    the distance of each pre-rounding value from the nearest rounding boundary is
    reported, and so is the fractional part of the tanh output before its rounding (the
    device's hidden activations are not recorded, so a tie inside a hidden layer can
    only be located, not proven to be what the NPU rounded differently).
C2  How far the fused-tanh emulation's codes lie from the Qualcomm NNAPI int8 codes:
    the share within 0, 1 and 2 quantization steps and the largest distance.
"""
import os
import sys
from fractions import Fraction

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import study3_mechanism as m  # noqa: E402

OUT = os.path.join(m.ROOT, "analysis", "out_study3", "tie_check.md")


def requant_exact(acc, mult):
    """gemmlowp's two roundings for one integer accumulator, with the distance of each
    pre-rounding value from its rounding boundary (0 = tie) in units of the result."""
    q, shift = m.quantize_multiplier(mult)
    left, right = max(shift, 0), max(-shift, 0)
    prod = int(acc) * (1 << left) * q
    first = Fraction(prod, 1 << 31)                  # SaturatingRoundingDoublingHighMul rounds this
    d1 = abs(first - round(first)) if first.denominator == 1 else abs(first - (int(first) + (Fraction(1, 2) if first >= 0 else -Fraction(1, 2))))
    nudge = (1 << 30) if prod >= 0 else 1 - (1 << 30)
    high = int((prod + nudge) / (1 << 31))         # C truncation toward zero
    if right == 0:
        return high, d1, None
    second = Fraction(high, 1 << right)              # RoundingDivideByPOT rounds this
    d2 = abs(second - int(second) - (Fraction(1, 2) if second >= 0 else -Fraction(1, 2)))
    mask = (1 << right) - 1
    rem = high & mask
    thr = (mask >> 1) + (high < 0)
    return (high >> right) + (rem > thr), d1, d2


def main():
    feat = m.inputs()
    layers, tanh = m.model("int8")
    dev = {k: np.rint(m.outputs(v) * 128) for k, v in m.INT8_CLASSES.items()}
    cpu, mtk, qc = (dev[k] for k in ("CPU INT8 (all five Android devices)", "MediaTek NPU INT8",
                                     "Qualcomm NPU INT8 (qti-default)"))
    lines = ["# Int8 claims traced to the emulation (`analysis/study3_tie_check.py`)", ""]

    # C8: the one MediaTek value
    em = m.run_int8(feat, layers, tanh, "gemmlowp", "rne")
    lines += ["## C8: the one MediaTek NPU int8 value that the two-rounding emulation misses", "",
              f"Emulation (gemmlowp requantization, round-to-nearest-even tanh) against the Android CPU int8 "
              f"codes: {100 * np.mean(em == cpu):.4f} % equal; against the MediaTek NPU: "
              f"{100 * np.mean(em == mtk):.4f} % equal, {int(np.sum(em != mtk))} value(s) differ."]
    for r, c in np.argwhere(em != mtk):
        lines.append(f"- row {r}, output {c}: emulation and CPU code {int(em[r, c])}, MediaTek code "
                     f"{int(mtk[r, c])} (CPU code {int(cpu[r, c])}).")
        # recompute the row layer by layer in exact integers
        s_in, zp_in = layers[0]["in"]
        x = np.clip(m.away(feat[r].astype(np.float32) / np.float32(s_in)) + zp_in, -128, 127).astype(np.int64)
        for li, L in enumerate(layers):
            s_x, zp_x = L["in"]
            s_w, _ = L["wq"]
            s_o, zp_o = L["out"]
            acc = ((x - zp_x) @ L["w"].astype(np.int64) + L["b"].astype(np.int64))
            codes, ties1, ties2 = [], [], []
            for u, a in enumerate(acc):
                y, d1, d2 = requant_exact(int(a), s_x * s_w / s_o)
                codes.append(int(np.clip(y + zp_o, -128, 127)))
                if d1 == 0:
                    ties1.append(u)
                if d2 is not None and d2 == 0:
                    ties2.append(u)
            x = np.array(codes, dtype=np.int64)
            lines.append(f"  - layer {li + 1} ({len(acc)} units): exact ties in the first rounding: "
                         f"{len(ties1)} {ties1[:8]}; in the second rounding: {len(ties2)} {ties2[:8]}.")
        s_t, zp_t = tanh["in"]
        s_u, zp_u = tanh["out"]
        v = np.tanh(s_t * (x[c] - zp_t)) / s_u
        frac = v - np.floor(v)
        lines.append(f"  - tanh input code {int(x[c])}, tanh output before rounding {v:.6f} (fraction "
                     f"{frac:.6f}, distance from the .5 tie {abs(frac - 0.5):.6f}); round-to-nearest gives "
                     f"{int(np.rint(v)) + zp_u}, the MediaTek code is {int(mtk[r, c])}.")

    # every tie of the last layer's second rounding, all rows: which way does the NPU round?
    s_in, zp_in = layers[0]["in"]
    x = np.clip(m.away(feat.astype(np.float32) / np.float32(s_in)) + zp_in, -128, 127)
    for L in layers[:-1]:
        s_x, zp_x = L["in"]
        s_w, _ = L["wq"]
        s_o, zp_o = L["out"]
        x = np.clip(m.requant((x - zp_x) @ L["w"] + L["b"], s_x * s_w / s_o, "gemmlowp") + zp_o, -128, 127)
    L = layers[-1]
    s_x, zp_x = L["in"]
    s_w, _ = L["wq"]
    s_o, zp_o = L["out"]
    acc = ((x - zp_x) @ L["w"] + L["b"]).astype(np.int64)
    q, shift = m.quantize_multiplier(s_x * s_w / s_o)
    left, right = max(shift, 0), max(-shift, 0)
    prod = acc * (1 << left) * q
    tie1 = (prod % (1 << 31)) == (1 << 30)
    nudge = np.where(prod >= 0, 1 << 30, 1 - (1 << 30))
    high = (prod + nudge) // (1 << 31)
    high = np.where((prod + nudge) < 0, -((-(prod + nudge)) // (1 << 31)), high)
    mask = (1 << right) - 1
    rem = high & mask
    y = (high >> right) + (rem > (mask >> 1) + (high < 0))     # gemmlowp: ties away from zero
    tie2 = rem == (1 << (right - 1))
    y_other = y - np.where(high >= 0, 1, -1)                   # the other side of the tie
    y_even = np.where(y % 2 == 0, y, y_other)                  # round half to even
    s_t, zp_t = tanh["in"]
    s_u, zp_u = tanh["out"]

    def out_codes(yy):
        code = np.clip(yy + zp_o, -128, 127)
        return np.clip(np.rint(np.tanh(s_t * (code - zp_t)) / s_u) + zp_u, -128, 127)

    o, o_other, o_even = out_codes(y), out_codes(y_other), out_codes(y_even)
    change = tie2 & (o != o_other)
    lines += ["", f"Last layer (shift {shift}: left {left}, right {right}), all {acc.shape[0]:,} rows x 3 units: "
              f"{int(tie1.sum())} exact ties in the first rounding, {int(tie2.sum())} in the second "
              f"(RoundingDivideByPOT). Rounding the second-rounding ties the other way changes the output code "
              f"at {int(change.sum())} of them; there the MediaTek codes equal the emulation (ties away from zero) "
              f"at {int(np.sum(mtk[change] == o[change]))} and the other side at "
              f"{int(np.sum(mtk[change] == o_other[change]))}; round half to even would give the MediaTek "
              f"code at {int(np.sum(mtk[change] == o_even[change]))} of the {int(change.sum())}."]

    # C2: the Qualcomm fit
    lines += ["", "## C2: distance of the fused-tanh emulation from the Qualcomm NNAPI int8 codes", "",
              "| hidden requantization | equal | within 1 step | within 2 steps | largest distance (steps) |",
              "|---|---|---|---|---|"]
    for mode in ("gemmlowp", "fp32_rne"):
        d = np.abs(m.run_int8_fused(feat, layers, mode) - qc)
        lines.append(f"| {mode} | {100 * np.mean(d == 0):.2f} % | {100 * np.mean(d <= 1):.2f} % | "
                     f"{100 * np.mean(d <= 2):.3f} % | {int(d.max())} |")
    with open(OUT, "w") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
