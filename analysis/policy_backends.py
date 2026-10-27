"""Study 3: run the enemy-AI policy model on the Mac in several numeric
"backends", so prevention methods can be compared before (and alongside)
the device data.

The device's behaviour is modelled, not measured, here: LiteRT's CPU
kernels on an Apple M-series chip are not the Poco's, and fp16 is emulated
in numpy. The point is realistic MAGNITUDES of cross-backend output
differences for the same inputs, with the real trained weights
(app/src/main/assets/models, the files inside the APK).

Backends (names used on the command line and in result tables)
  lite_f32    LiteRT float32 interpreter (XNNPACK), the reference, same
              batch-shape handling as the app (resize to the wave's batch,
              zero padding rows, alive rows used)
  lite_int8   LiteRT full-integer int8 model with the app's quantize /
              dequantize code path (nn.cpp: lroundf, clamp, (q-zp)*scale)
  np_f32      numpy float32 MLP from the same weights (Accelerate BLAS):
              a different CPU kernel, so ULP-level differences; proxy for
              "another chip's CPU F32"
  np_f16      fp16 weights and fp16 activations between layers, fp32
              accumulation (what a GPU/NPU running the f32 model at
              reduced precision does at best)
  np_f16acc   as np_f16 but the dot products also accumulate in fp16
              (chunked emulation; the pessimistic fp16 case)
  np_f64      float64 MLP: the "mathematically exact" reference used to
              split each backend's error into its own rounding vs. the
              reference's

Commands
  info  [--variant xlarge]
  replay TRACE.bin BACKEND OUT.bin [--variant xlarge]
      Replays an NPBT trace's inputs through BACKEND at the trace's batch
      shapes and writes an NPBR file, exactly like the app's REPLAY INPUT
      (so analysis/determinism.py and prevention.py read it unchanged).
  probe TRACE.bin [--variant xlarge] [--ticks N]
      Quick table: every backend against lite_f32 over the trace's inputs.
"""

import os
import struct
import sys
import warnings

import numpy as np

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
warnings.filterwarnings("ignore")

HERE = os.path.dirname(os.path.abspath(__file__))
MODELS = os.path.normpath(os.path.join(HERE, "..", "app", "src", "main", "assets", "models"))
FEAT, OUT = 14, 3
BACKENDS = ("lite_f32", "lite_int8", "lite_int8_ref", "np_f32", "np_f16", "np_f16acc", "np_f64", "pp")


def _interpreter(path, ref_kernels=False):
    import tensorflow as tf  # noqa: E402  (late import: slow)
    if ref_kernels:  # TFLite reference kernels: the Android CPU int8 class (two roundings)
        return tf.lite.Interpreter(model_path=path, num_threads=4,
                                   experimental_op_resolver_type=tf.lite.experimental.OpResolverType.BUILTIN_REF)
    return tf.lite.Interpreter(model_path=path, num_threads=4)


def lround(x):
    """C lroundf: round half away from zero (numpy rounds half to even)."""
    return np.sign(x) * np.floor(np.abs(x) + 0.5)


class LiteBackend:
    """LiteRT interpreter with the app's batch handling and int8 I/O path."""

    def __init__(self, variant, prec, ref_kernels=False):
        self.path = os.path.join(MODELS, f"policy_{variant}_{prec}.tflite")
        self.it = _interpreter(self.path, ref_kernels)
        self.inp = self.it.get_input_details()[0]
        self.out = self.it.get_output_details()[0]
        self.int8 = self.inp["dtype"] == np.int8
        self.in_scale, self.in_zero = self.inp["quantization"]
        self.out_scale, self.out_zero = self.out["quantization"]
        self.batch = -1

    def set_batch(self, n):
        if n == self.batch:
            return
        self.it.resize_tensor_input(self.inp["index"], [1, n, 1, FEAT])
        self.it.allocate_tensors()
        self.batch = n

    def run(self, feat, batch):
        """feat: (alive x 14) float32. Returns (alive x 3) float32."""
        alive = feat.shape[0]
        self.set_batch(batch)
        x = np.zeros((1, batch, 1, FEAT), np.float32)
        x[0, :alive, 0, :] = feat
        if self.int8:
            q = lround(x / np.float32(self.in_scale)) + self.in_zero
            x = np.clip(q, -128, 127).astype(np.int8)
        self.it.set_tensor(self.inp["index"], x)
        self.it.invoke()
        y = self.it.get_tensor(self.out["index"])
        if self.int8:
            y = ((y.astype(np.int32) - self.out_zero) * np.float32(self.out_scale)).astype(np.float32)
        return np.ascontiguousarray(y[0, :alive, 0, :]).astype(np.float32)


class NumpyBackend:
    """The f32 model's weights re-run as a plain MLP in the chosen arithmetic."""

    def __init__(self, variant, mode):
        it = _interpreter(os.path.join(MODELS, f"policy_{variant}_f32.tflite"))
        it.allocate_tensors()
        self.layers = []
        for op in it._get_ops_details():  # noqa: SLF001
            if op["op_name"] != "CONV_2D":
                continue
            _, w_idx, b_idx = op["inputs"][:3]
            w = it.get_tensor(w_idx)            # [out, 1, 1, in]
            b = it.get_tensor(b_idx)
            self.layers.append((w[:, 0, 0, :].T.copy(), b.copy()))   # (in x out), (out)
        self.mode = mode
        if mode == "np_f64":
            self.dt = np.float64
        elif mode == "np_f32":
            self.dt = np.float32
        else:
            self.dt = np.float16
        self.W = [(w.astype(self.dt), b.astype(self.dt)) for w, b in self.layers]

    def _dot(self, x, w):
        if self.mode == "np_f16acc":
            # fp16 accumulation emulated: partial sums over chunks of 8 inputs
            # rounded to fp16 after every chunk (hardware FMA units differ; this
            # is a pessimistic, documented approximation)
            acc = np.zeros((x.shape[0], w.shape[1]), np.float16)
            x32, w32 = x.astype(np.float32), w.astype(np.float32)
            for k in range(0, w.shape[0], 8):
                part = x32[:, k:k + 8] @ w32[k:k + 8, :]
                acc = (acc.astype(np.float32) + part.astype(np.float16).astype(np.float32)).astype(np.float16)
            return acc
        if self.dt == np.float16:
            return (x.astype(np.float32) @ w.astype(np.float32)).astype(np.float16)
        return x @ w

    def run(self, feat, batch):
        x = feat.astype(self.dt)
        n = len(self.W)
        for i, (w, b) in enumerate(self.W):
            z = self._dot(x, w)
            if self.dt == np.float16:
                z = (z.astype(np.float32) + b.astype(np.float32)).astype(np.float16)
            else:
                z = z + b
            if i < n - 1:
                x = np.maximum(z, 0)
            else:
                x = np.tanh(z.astype(np.float32)).astype(self.dt) if self.dt == np.float16 else np.tanh(z)
        return np.ascontiguousarray(x).astype(np.float32)


class PortableBackend:
    """tools/portable_policy through ctypes (build_lib.sh): the kernel every peer
    ships for the guarded recompute, bit-identical to the app's copy."""

    def __init__(self, variant):
        import ctypes
        lib = ctypes.CDLL(os.path.join(HERE, "..", "tools", "portable_policy", "libpp.dylib"))
        lib.pp_load.restype = ctypes.c_void_p
        lib.pp_load.argtypes = [ctypes.c_char_p, ctypes.c_ulong]
        lib.pp_forward.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]
        self._blob = open(os.path.join(MODELS, f"policy_{variant}_f32.nppw"), "rb").read()
        self.lib, self.p = lib, lib.pp_load(self._blob, len(self._blob))

    def run(self, feat, batch):
        x = np.ascontiguousarray(feat, np.float32)
        y = np.zeros((len(x), OUT), np.float32)
        for i in range(len(x)):
            self.lib.pp_forward(self.p, x[i].ctypes.data, y[i].ctypes.data)
        return y


def make_backend(name, variant="xlarge"):
    if name == "lite_f32":
        return LiteBackend(variant, "f32")
    if name == "lite_int8":
        return LiteBackend(variant, "int8")
    if name == "lite_int8_ref":
        return LiteBackend(variant, "int8", ref_kernels=True)
    if name == "pp":
        return PortableBackend(variant)
    if name in ("np_f32", "np_f16", "np_f16acc", "np_f64"):
        return NumpyBackend(variant, name)
    raise ValueError(f"unknown backend {name}; one of {BACKENDS}")


# ------------------------------------------------------------------ traces

def read_trace(path):
    sys.path.insert(0, HERE)
    from determinism import read_bin
    kind, ticks = read_bin(path)
    if kind != "NPBT":
        sys.exit(f"{path}: need an NPBT trace (with inputs)")
    return ticks


def write_replay(path, ticks, outs):
    with open(path, "wb") as f:
        f.write(b"NPBR" + struct.pack("<III", 1, FEAT, OUT))
        for t, o in zip(ticks, outs):
            f.write(struct.pack("<IHHHH", t["tick"], t["wave"], t["batch"], t["alive"], 0))
            f.write(np.ascontiguousarray(o, np.float32).tobytes())


def replay(ticks, backend):
    return [backend.run(t["feat"], t["batch"]) if t["alive"] > 0
            else np.zeros((0, OUT), np.float32) for t in ticks]


# ------------------------------------------------------------------ commands

def cmd_info(variant):
    for prec in ("f32", "int8"):
        b = LiteBackend(variant, prec)
        b.set_batch(80)
        print(f"{b.path}: input {b.inp['dtype'].__name__} q={b.inp['quantization']} "
              f"output {b.out['dtype'].__name__} q={b.out['quantization']}")
    nb = NumpyBackend(variant, "np_f32")
    for i, (w, b) in enumerate(nb.layers):
        print(f"  layer {i}: {w.shape[0]} -> {w.shape[1]}  |w|max {np.abs(w).max():.4f}  |b|max {np.abs(b).max():.4f}")


def cmd_replay(trace, backend, out, variant):
    ticks = read_trace(trace)
    outs = replay(ticks, make_backend(backend, variant))
    write_replay(out, ticks, outs)
    rows = sum(len(o) for o in outs)
    print(f"{out}: {len(ticks)} ticks, {rows} rows from {backend} ({variant})")


def cmd_probe(trace, variant, nticks):
    from determinism import output_diff
    ticks = read_trace(trace)
    if nticks:
        ticks = ticks[:nticks]
    ref = replay(ticks, make_backend("lite_f32", variant))
    a = np.concatenate([o for o in ref if len(o)])
    print(f"{trace}: {len(ticks)} ticks, {len(a)} rows, variant {variant}")
    print(f"{'backend':10s} {'bitexact%':>9s} {'max_abs':>9s} {'mean_abs':>9s} {'max_ulp':>8s} "
          f"{'fire_flips':>10s} {'flip/1k':>8s}")
    for name in BACKENDS:
        if name == "lite_f32":
            continue
        outs = replay(ticks, make_backend(name, variant))
        b = np.concatenate([o for o in outs if len(o)])
        d, ulp, flip = output_diff(a, b)
        print(f"{name:10s} {100 * np.mean(d == 0):9.3f} {d.max():9.3g} {d.mean():9.3g} {ulp.max():8d} "
              f"{int(flip.sum()):10d} {1000 * flip.mean():8.2f}")


def main(argv):
    variant = "xlarge"
    if "--variant" in argv:
        i = argv.index("--variant")
        variant = argv[i + 1]
        del argv[i:i + 2]
    nticks = 0
    if "--ticks" in argv:
        i = argv.index("--ticks")
        nticks = int(argv[i + 1])
        del argv[i:i + 2]
    if argv and argv[0] == "info":
        cmd_info(variant)
    elif len(argv) == 4 and argv[0] == "replay":
        cmd_replay(argv[1], argv[2], argv[3], variant)
    elif len(argv) == 2 and argv[0] == "probe":
        cmd_probe(argv[1], variant, nticks)
    else:
        sys.exit(__doc__)


if __name__ == "__main__":
    main(sys.argv[1:])
