"""Export a float32 policy model (app/src/main/assets/models/policy_<v>_f32.tflite)
to the flat little-endian file read by policy_portable.c:

  u32 magic 'NPPW', u32 n_layers, then per layer: u32 n_in, u32 n_out,
  f32 W[n_out][n_in] (row j = weights of output j), f32 b[n_out].
Hidden layers use ReLU, the last tanh.

  python3 tools/portable_policy/export_weights.py xlarge OUT.bin
"""
import os, struct, sys
import numpy as np
import tensorflow as tf

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
variant, out = sys.argv[1], sys.argv[2]
it = tf.lite.Interpreter(model_path=os.path.join(ROOT, "app/src/main/assets/models", f"policy_{variant}_f32.tflite"))
it.allocate_tensors()
layers = []
for op in it._get_ops_details():  # noqa: SLF001
    if op["op_name"] == "CONV_2D":
        _, w, b = op["inputs"][:3]
        W = it.get_tensor(w)[:, 0, 0, :].astype(np.float32)      # [out, in]
        layers.append((W, it.get_tensor(b).astype(np.float32)))
with open(out, "wb") as f:
    f.write(b"NPPW" + struct.pack("<I", len(layers)))
    for W, b in layers:
        f.write(struct.pack("<II", W.shape[1], W.shape[0]))
        f.write(np.ascontiguousarray(W, "<f4").tobytes()); f.write(np.ascontiguousarray(b, "<f4").tobytes())
print(out, [W.shape for W, _ in layers])
