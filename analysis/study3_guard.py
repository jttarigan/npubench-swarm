"""Study 3: symmetric prevention with a common recompute kernel.

Every peer applies the same rule to its own outputs: rows with any output within D
of a rounding edge or of the fire threshold are recomputed with the portable kernel
(tools/portable_policy, the same float32 bits on every platform), then all outputs are
rounded to a grid of 2^-K. A peer pair stays identical when both obtain the same
values on every row. Unlike analysis/prevention.py, no peer is assumed to reproduce
the Android CPU's bits; every pair of output classes can be evaluated.

  python3 analysis/study3_guard.py PP_OUT.f32 > analysis/out_study3/guard.md
PP_OUT.f32: tools/portable_policy/pp_cli outputs over the reference XL game's rows.
"""
import sys
import numpy as np
sys.path.insert(0, "analysis")
from cosim import Prevent  # noqa: E402
from study3_crosschip import replay_outputs  # noqa: E402

PEERS = [  # (label, replay label in crosschip, class)
    ("Mac CPU F32, XNNPACK", "mac f32", "R16"),
    ("iPhone CPU F32 (LiteRT 2.20-dev)", "iphone12 cpu_f32", "R11"),
    ("iPhone CPU F32 (LiteRT 2.17.0)", "iphone12_lite217 cpu_f32", "R19"),
    ("Xclipse 540 GPU F32", "a56 gpu_f32", "R10"),
    ("Mali-G72 GPU F32", "s6lite gpu_f32", "R3"),
    ("Mali-G615 GPU F32", "poco gpu_f32", "R4"),
    ("MediaTek NPU F32", "poco npu_f32", "R5"),
    ("Core ML F32", "iphone12 npu_f32", "R15"),
    ("Metal GPU F32 (A14)", "iphone12 gpu_f32", "R13"),
    ("Metal GPU F32 (A19 Pro)", "iphone17pm gpu_f32", "R17"),
    ("Adreno 650 GPU F32", "tab gpu_f32", "R9"),
    ("Qualcomm NPU F32", "tab npu_f32", "R8"),
    ("Adreno 840 GPU F32", "fold8 gpu_f32", "R20"),
]
INT8 = [("Apple CPU INT8 vs Android CPU INT8", "iphone12 cpu_int8", "s6lite cpu_int8")]
METHODS = ["grid:2", "grid:3", "grid:4", "grid:2+fallback:0.0001", "grid:3+fallback:0.0001",
           "grid:2+fallback:0.01", "grid:3+fallback:0.01", "grid:2+fallback:0.02", "grid:2+fallback:0.12"]


def pair(a, b, common, spec):
    p = Prevent(spec)
    ao, na = p.apply(a, common)
    bo, nb = p.apply(b, common)
    diff = np.any(ao != bo, axis=1)
    return int(diff.sum()), 100.0 * max(na, nb) / len(a)


def cell(n, cost, rows, guarded):
    if not guarded:
        return f"{100.0 * n / rows:.3g} %" if n else "0"
    bound = f" (<{300.0 / rows:.1g} %)" if n == 0 else ""
    return (f"{100.0 * n / rows:.3g} %" if n else "0" + bound) + f" at {cost:.2g} %"


def main(argv):
    outs = replay_outputs("xlarge")
    ref = outs["s6lite cpu_f32"][1].reshape(-1, 3)
    common = np.fromfile(argv[0], "<f4").reshape(-1, 3)
    assert common.shape == ref.shape
    rows = len(ref)
    print("# Symmetric prevention with a common recompute kernel (`analysis/study3_guard.py`)\n")
    print(f"XL replay, {rows:,} rows. Each cell: share of rows where the two peers still differ "
          "after prevention, and (with a guard) the larger of the two peers' shares of rows "
          "recomputed with the portable kernel. '0 (<x %)': no differing row; x is the 95 % "
          "upper bound on the per-row rate (rule of three).\n")
    print(f"Portable kernel vs Android CPU F32 (R1): max |diff| "
          f"{np.abs(common.astype(np.float64) - ref).max():.2g}.\n")
    hdr = ["peer (vs Android CPU F32, R1)", "max abs diff"] + [m.replace("grid:", "2^-").replace("+fallback:", ", guard ") for m in METHODS]
    print("| " + " | ".join(hdr) + " |"); print("|" + "---|" * len(hdr))
    for label, key, cls in PEERS:
        b = outs[key][1].reshape(-1, 3)
        cells = [cell(*pair(ref, b, common, m), rows, "fallback" in m) for m in METHODS]
        print(f"| {label} ({cls}) | {np.abs(ref.astype(np.float64) - b).max():.2g} | " + " | ".join(cells) + " |")
    for label, ka, kb in INT8:
        a, b = outs[kb][1].reshape(-1, 3), outs[ka][1].reshape(-1, 3)
        cells = [cell(*pair(a, b, common, m), rows, "fallback" in m) for m in METHODS]
        print(f"| {label} | {np.abs(a.astype(np.float64) - b).max():.2g} | " + " | ".join(cells) + " |")


if __name__ == "__main__":
    main(sys.argv[1:])
