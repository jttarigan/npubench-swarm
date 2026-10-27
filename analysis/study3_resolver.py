"""Study 3: Mac LiteRT (TF 2.20) with three op resolvers vs every CPU replay class.

Replays the inputs of the reference XL game through the Mac's LiteRT (TensorFlow
2.20 Python) with three op resolvers: AUTO (XNNPACK delegate, the Python
default), BUILTIN_WITHOUT_DEFAULT_DELEGATES (the built-in optimized kernels,
which is what the TFLite C API uses on the phones: no XNNPACK delegate is
created by default) and BUILTIN_REF (reference kernels). Compares each with the
CPU replay classes of the phones.

  python3 analysis/study3_resolver.py > analysis/out_study3/resolver.txt
"""
import sys, numpy as np
sys.path.insert(0, "analysis")
import tensorflow as tf
import policy_backends as pb
from determinism import read_bin
REF = "runs_study3/s6lite/files/run_20261013_223444_game_cpu_xlarge_f32_trace.bin"
F = "runs_study3/{}/files/run_{}_replay.bin"
PEERS = {"f32": {"Android CPU (R1)": F.format("s6lite", "20261013_234430_replay_cpu_xlarge_f32"),
                 "iPhone CPU 2.20 (R11)": F.format("iphone12", "20261016_130318732_004_replay_cpu_xlarge_f32"),
                 "iPhone CPU 2.17 (R19)": F.format("iphone12_lite217", "20261017_120624178_004_replay_cpu_xlarge_f32")},
         "int8": {"Android CPU (R2)": F.format("s6lite", "20261013_234500_replay_cpu_xlarge_int8"),
                  "iPhone CPU (R12)": F.format("iphone12", "20261016_130321484_008_replay_cpu_xlarge_int8")}}
R = tf.lite.experimental.OpResolverType
MODES = {"XNNPACK (default)": R.AUTO, "built-in, no XNNPACK": R.BUILTIN_WITHOUT_DEFAULT_DELEGATES, "reference kernels": R.BUILTIN_REF}
_, ticks = read_bin(REF)
print("Mac M5, TensorFlow", tf.__version__, "| values differing over 370,062 (max abs diff)")
for prec, peers in PEERS.items():
    ref = {k: np.concatenate([t["out"] for t in read_bin(p)[1]]) for k, p in peers.items()}
    for name, mode in MODES.items():
        pb._interpreter = lambda path, m=mode: tf.lite.Interpreter(model_path=path, num_threads=4, experimental_op_resolver_type=m)
        b = pb.LiteBackend("xlarge", prec)
        o = np.concatenate([b.run(t["feat"], t["batch"]) for t in ticks])
        cells = [f"{k}: {(o != v).sum()} ({np.abs(o - v).max():.2g})" for k, v in ref.items()]
        print(f"{prec:4s} | {name:20s} | " + " | ".join(cells), flush=True)
