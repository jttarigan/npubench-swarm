"""Study 3: explain the device differences by reproducing the devices' outputs
on the Mac with candidate arithmetic.

All device replays share the inputs of one XL game (the Tab S6 Lite CPU F32
trace), so every candidate is run over those 123,354 rows and compared with
each device output class value by value.

int8: the full-integer model is 4 CONV_2D (1x1, fused ReLU, per-tensor
scales) and an int8 TANH. Accumulation is exact integer arithmetic (done in
float64, exact below 2^53), so the candidates differ only in how the int32
accumulator is requantized to int8 and how TANH is evaluated:
  fp32_rne    acc * float32(multiplier), round half to even (XNNPACK "fp32")
  fp32_away   acc * float32(multiplier), round half away from zero
  gemmlowp    TFLite reference: QuantizeMultiplier + SaturatingRounding-
              DoublingHighMul + RoundingDivideByPOT (two roundings)
  rndnu       one fixed-point rounding, ties toward +inf ((acc*m + 2^(s-1)) >> s)
  f64_rne     exact real multiplier, round half to even
TANH: q_out = round(tanh(s_in * (q_in - zp_in)) / s_out) + zp_out with half
to even or half away.

Qualcomm int8: the same model with the last CONV_2D fused into TANH (tanh on
the dequantized int32 accumulator, no int8 step in between).

float: fp16 weights and activations with fp32 accumulation or fp16
accumulation every 64, 16, 4 or 1 inputs, output kept in fp32 or rounded to
fp16, on 20,000 sampled rows, compared by error size against float64.

  python3 analysis/study3_mechanism.py [OUT_DIR]   (default analysis/out_study3)
"""

import glob
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
from determinism import read_bin  # noqa: E402

ROOT = os.path.normpath(os.path.join(HERE, ".."))
MODELS = os.path.join(ROOT, "app", "src", "main", "assets", "models")
REF = os.path.join(ROOT, "runs_study3/s6lite/files/run_20261013_223444_game_cpu_xlarge_f32_trace.bin")
INT8_CLASSES = {
    "CPU INT8 (all five Android devices)": "runs_study3/s6lite/files/*replay_cpu_xlarge_int8_replay.bin",
    "MediaTek NPU INT8": "runs_study3/poco/files/*replay_npu_xlarge_int8_replay.bin",
    "Qualcomm NPU INT8 (qti-default)": "runs_study3/tab/files/*replay_npu_xlarge_int8_replay.bin",
    "Mac LiteRT int8": "runs_study3/s6lite/mac_replays/replay_mac_lite_int8.bin",
}
F32_CLASSES = {
    "CPU F32 (all five Android devices)": "runs_study3/s6lite/files/*replay_cpu_xlarge_f32_replay.bin",
    "Mali-G72 GPU": "runs_study3/s6lite/files/*replay_gpu_xlarge_f32_replay.bin",
    "Mali-G615 GPU": "runs_study3/poco/files/*replay_gpu_xlarge_f32_replay.bin",
    "MediaTek NPU F32": "runs_study3/poco/files/*replay_npu_xlarge_f32_replay.bin",
    "Adreno 650 GPU": "runs_study3/tab/files/*replay_gpu_xlarge_f32_replay.bin",
    "Qualcomm NPU F32 (qti-default)": "runs_study3/tab/files/*replay_npu_xlarge_f32_replay.bin",
    "Mac LiteRT f32": "runs_study3/s6lite/mac_replays/replay_mac_lite_f32.bin",
    "Xclipse 540 GPU": "runs_study3/a56/files/*replay_gpu_xlarge_f32_replay.bin",
    "Metal GPU (A14)": "runs_study3/iphone12/files/*replay_gpu_xlarge_f32_replay.bin",
    "Metal GPU (A19 Pro)": "runs_study3/iphone17pm/files/*replay_gpu_xlarge_f32_replay.bin",
    "Core ML F32": "runs_study3/iphone12/files/*replay_npu_xlarge_f32_replay.bin",
    "Adreno 840 GPU": "runs_study3/fold8/files/*replay_gpu_xlarge_f32_replay.bin",
}


def outputs(path):
    p = sorted(glob.glob(os.path.join(ROOT, path)))[0]
    return np.concatenate([t["out"] for t in read_bin(p)[1] if t["alive"] > 0])


def inputs():
    return np.concatenate([t["feat"] for t in read_bin(REF)[1] if t["alive"] > 0])


def model(prec):
    import tensorflow as tf
    it = tf.lite.Interpreter(model_path=os.path.join(MODELS, f"policy_xlarge_{prec}.tflite"))
    it.allocate_tensors()
    td = {t["index"]: t for t in it.get_tensor_details()}

    def q(i):
        p = td[i]["quantization_parameters"]
        return (float(p["scales"][0]), int(p["zero_points"][0])) if len(p["scales"]) else (None, None)

    layers, tanh = [], None
    for op in it._get_ops_details():  # noqa: SLF001
        if op["op_name"] == "CONV_2D":
            x, w, b = op["inputs"][:3]
            layers.append({"w": it.get_tensor(w)[:, 0, 0, :].T.astype(np.float64),
                           "b": it.get_tensor(b).astype(np.float64),
                           "in": q(x), "wq": q(w), "out": q(op["outputs"][0])})
        elif op["op_name"] == "TANH":
            tanh = {"in": q(op["inputs"][0]), "out": q(op["outputs"][0])}
    return layers, tanh


# ------------------------------------------------------------------ int8

def rne(x):
    return np.rint(x)


def away(x):
    return np.sign(x) * np.floor(np.abs(x) + 0.5)


def quantize_multiplier(m):
    """TFLite QuantizeMultiplier: m = q * 2^shift, q in [2^30, 2^31)."""
    mant, shift = np.frexp(m)
    q = int(round(mant * (1 << 31)))
    if q == (1 << 31):
        q //= 2
        shift += 1
    return q, int(shift)


def gemmlowp(acc, m):
    q, shift = quantize_multiplier(m)
    left, right = max(shift, 0), max(-shift, 0)
    a = acc.astype(np.int64) * (1 << left)
    prod = a * q                                         # SaturatingRoundingDoublingHighMul
    nudge = np.where(prod >= 0, 1 << 30, 1 - (1 << 30))
    high = (prod + nudge) // (1 << 31)
    high = np.where((prod + nudge) < 0, -((-(prod + nudge)) // (1 << 31)), high)  # C truncation
    if right == 0:
        return high
    mask = (1 << right) - 1                              # RoundingDivideByPOT
    rem = high & mask
    thr = (mask >> 1) + (high < 0)
    return (high >> right) + (rem > thr)


def rndnu(acc, m):
    q, shift = quantize_multiplier(m)
    s = 31 - shift
    return (acc.astype(np.int64) * q + (1 << (s - 1))) >> s


def requant(acc, m, mode):
    if mode == "fp32_rne":
        return rne(acc.astype(np.float32) * np.float32(m))
    if mode == "fp32_away":
        return away(acc.astype(np.float32) * np.float32(m))
    if mode == "f64_rne":
        return rne(acc * m)
    if mode == "gemmlowp":
        return gemmlowp(acc.astype(np.int64), m).astype(np.float64)
    if mode == "rndnu":
        return rndnu(acc.astype(np.int64), m).astype(np.float64)
    raise ValueError(mode)


def run_int8(feat, layers, tanh, mode, tanh_round):
    s_in, zp_in = layers[0]["in"]
    x = np.clip(away(feat.astype(np.float32) / np.float32(s_in)) + zp_in, -128, 127)  # app: lroundf
    for L in layers:
        s_x, zp_x = L["in"]
        s_w, _ = L["wq"]
        s_o, zp_o = L["out"]
        acc = (x - zp_x) @ L["w"] + L["b"]                # exact integers in float64
        y = requant(acc, s_x * s_w / s_o, mode) + zp_o
        x = np.clip(y, -128, 127)                          # fused ReLU = clamp at zp_o (-128)
    s_t, zp_t = tanh["in"]
    s_u, zp_u = tanh["out"]
    v = np.tanh(s_t * (x - zp_t)) / s_u
    out = np.clip((rne(v) if tanh_round == "rne" else away(v)) + zp_u, -128, 127)
    return out                                             # int8 codes (output scale 1/128, zp 0)


def run_int8_fused(feat, layers, mode):
    s_in, zp_in = layers[0]["in"]
    x = np.clip(away(feat.astype(np.float32) / np.float32(s_in)) + zp_in, -128, 127)
    for L in layers[:-1]:
        s_x, zp_x = L["in"]
        s_w, _ = L["wq"]
        s_o, zp_o = L["out"]
        x = np.clip(requant((x - zp_x) @ L["w"] + L["b"], s_x * s_w / s_o, mode) + zp_o, -128, 127)
    L = layers[-1]
    real = ((x - L["in"][1]) @ L["w"] + L["b"]) * L["in"][0] * L["wq"][0]
    return np.clip(rne(np.tanh(real) * 128), -128, 127)


# ------------------------------------------------------------------ float

def run_float(feat, layers, store, acc, chunk=8):
    """store: 'f32' or 'f16' for weights/activations; acc: 'f32', 'f16'."""
    st = np.float16 if store == "f16" else np.float32
    x = feat.astype(st)
    n = len(layers)
    for i, L in enumerate(layers):
        w = L["w"].astype(st)
        if acc == "f16":
            a = np.zeros((x.shape[0], w.shape[1]), np.float16)
            for k in range(0, w.shape[0], chunk):
                part = (x[:, k:k + chunk].astype(np.float32) @ w[k:k + chunk].astype(np.float32))
                a = (a.astype(np.float32) + part.astype(np.float16).astype(np.float32)).astype(np.float16)
            z = (a.astype(np.float32) + L["b"].astype(st).astype(np.float32)).astype(st)
        else:
            z = (x.astype(np.float32) @ w.astype(np.float32) + L["b"].astype(st).astype(np.float32)).astype(st)
        x = np.maximum(z, 0) if i < n - 1 else np.tanh(z.astype(np.float32)).astype(st)
    return x.astype(np.float32)


def emulate(f, layers, chunk, final):
    """fp16 weights, inputs and activations; accumulation in fp32 (chunk 0) or
    rounded to fp16 after every `chunk` inputs; tanh in fp32 on the fp16
    pre-activation; output kept in fp32 or rounded to fp16."""
    x = f.astype(np.float16)
    for i, L in enumerate(layers):
        w = L["w"].astype(np.float16).astype(np.float32)
        xs = x.astype(np.float32)
        if chunk:
            a = np.zeros((len(xs), w.shape[1]), np.float16)
            for k in range(0, w.shape[0], chunk):
                part = (xs[:, k:k + chunk] @ w[k:k + chunk]).astype(np.float16).astype(np.float32)
                a = (a.astype(np.float32) + part).astype(np.float16)
            z = a.astype(np.float32) + L["b"].astype(np.float16).astype(np.float32)
        else:
            z = xs @ w + L["b"].astype(np.float16).astype(np.float32)
        z = z.astype(np.float16)
        if i < len(layers) - 1:
            x = np.maximum(z, 0)
        else:
            y = np.tanh(z.astype(np.float32))
            x = y.astype(np.float16).astype(np.float32) if final == "f16" else y
    return x.astype(np.float32)


def run_f64(feat, layers):
    x = feat.astype(np.float64)
    for i, L in enumerate(layers):
        z = x @ L["w"] + L["b"]
        x = np.maximum(z, 0) if i < len(layers) - 1 else np.tanh(z)
    return x


def main(argv):
    out_dir = argv[0] if argv else os.path.join(ROOT, "analysis/out_study3")
    feat = inputs()
    lines = ["# Study 3: reproducing the device outputs on the Mac (`analysis/study3_mechanism.py`)", "",
             f"Inputs: the {feat.shape[0]:,} enemy rows of the shared XL replay; "
             f"{feat.shape[0] * 3:,} output values per run.", ""]

    # int8
    layers, tanh = model("int8")
    dev = {k: np.rint(outputs(v) * 128) for k, v in INT8_CLASSES.items()}
    lines += ["## int8: share of output codes reproduced exactly", "",
              "| requantization | tanh rounding | " + " | ".join(dev) + " |",
              "|---|---|" + "---|" * len(dev)]
    for mode in ("fp32_rne", "fp32_away", "gemmlowp", "rndnu", "f64_rne"):
        for tr in ("rne", "away"):
            em = run_int8(feat, layers, tanh, mode, tr)
            cells = [f"{100 * np.mean(em == d):.4f} %" for d in dev.values()]
            lines.append(f"| {mode} | {tr} | " + " | ".join(cells) + " |")
            print(mode, tr, cells, flush=True)

    # Qualcomm: the last CONV_2D fused with TANH, tanh on the dequantized int32
    # accumulator (no int8 requantization in between), hidden layers as above
    lines += ["", "Fused last layer (tanh applied to the dequantized accumulator, no int8 step "
              "before it), hidden layers requantized as named:", "",
              "| hidden requantization | " + " | ".join(dev) + " |", "|---|" + "---|" * len(dev)]
    for mode in ("gemmlowp", "fp32_rne"):
        em = run_int8_fused(feat, layers, mode)
        lines.append(f"| {mode} | " + " | ".join(f"{100 * np.mean(em == d):.4f} %" for d in dev.values()) + " |")
        print("fused", mode, flush=True)

    # float
    flayers, _ = model("f32")
    rows = np.sort(np.random.default_rng(0).choice(len(feat), 20000, replace=False))
    f = feat[rows]
    devf = {k: outputs(v)[rows] for k, v in F32_CLASSES.items()}
    exact = run_f64(f, flayers)
    lines += ["", "## float32 model: error against exact (float64) arithmetic", "",
              "20,000 rows sampled from the replay (seed 0). An emulation explains a device when "
              "its error has the same size; summation order inside a GPU or NPU kernel is unknown, "
              "so bit-exact matches are not expected except for the MediaTek NPU. 'on fp16 grid' is "
              "the share of output values exactly representable in fp16 (1.000 = final tensor "
              "stored in fp16).", "",
              "| run | mean abs err | p99 abs err | max abs err | on fp16 grid | bit-exact with |",
              "|---|---|---|---|---|---|"]

    def row(name, y, match=""):
        e = np.abs(y.astype(np.float64) - exact)
        grid = np.mean(y.astype(np.float16).astype(np.float32) == y)
        lines.append(f"| {name} | {e.mean():.2e} | {np.quantile(e, 0.99):.2e} | {e.max():.2e} | {grid:.3f} | {match} |")

    for k, y in devf.items():
        row(k, y)
    for chunk, final in ((0, "f32"), (0, "f16"), (64, "f32"), (16, "f32"), (4, "f32"), (1, "f16")):
        y = emulate(f, flayers, chunk, final)
        best = max(devf, key=lambda k: np.mean(devf[k] == y))
        acc = "f32 accumulation" if chunk == 0 else f"f16 accumulation every {chunk} inputs"
        row(f"emulation: f16 weights and activations, {acc}, {final} output", y,
            f"{100 * np.mean(devf[best] == y):.2f} % ({best})")
        print("float", chunk, final, flush=True)

    md = "\n".join(lines) + "\n"
    with open(os.path.join(out_dir, "mechanism.md"), "w") as f:
        f.write(md)
    print(md)


if __name__ == "__main__":
    main(sys.argv[1:])
