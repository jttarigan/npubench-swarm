"""Trains and exports all benchmark models (see PROJECT_BRIEF).

Policy net (feature spec PF1, mirrors app/src/main/cpp/src/common.h):
  input  [N x 14] : posx_n, posy_n, velx_n, vely_n, dpx, dpy, dist,
                    3x nearest-neighbour deltas (dx,dy)/0.25 clamped, t_sin
  output [N x 3]  : tanh steer_x, steer_y, fire
  Trained by imitation of a scripted heuristic (behaviour quality is NOT the
  research question; latency is). 5 width/depth variants for the CPU-vs-NPU
  crossover sweep.

Gesture CNN (spec GS1, mirrors gesture.cpp):
  input  [64 x 2] : stroke resampled to 64 equidistant points, centroid at
                    origin, scaled by half max bbox dim
  output [4]      : softmax over {none, circle, slash, zigzag}
  Trained on synthetic strokes with heavy augmentation.

Exports every model as float32 and full-integer int8 .tflite into
app/src/main/assets/models/ and verifies each with the TFLite interpreter
(including a batch-resize check for the policy nets).
Ops are kept to FULLY_CONNECTED / CONV_2D / POOL / MEAN / TANH / SOFTMAX so the
NNAPI delegate can take whole models (see brief: avoid exotic ops).
"""

import json
import os
import shutil
import time

import numpy as np

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
import tensorflow as tf  # noqa: E402

SEED = 7
np.random.seed(SEED)
tf.random.set_seed(SEED)

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "out")
ASSETS = os.path.normpath(os.path.join(HERE, "..", "app", "src", "main", "assets", "models"))
# scratch outside OneDrive (sync locks/readonly-flags files under the repo)
TMP = os.path.join(os.path.expanduser("~"), ".npubench-build", "tmp_export")
os.makedirs(OUT, exist_ok=True)
os.makedirs(ASSETS, exist_ok=True)

FEATURE_DIM = 14
POLICY_OUT = 3
GESTURE_PTS = 64
GESTURE_CLASSES = 4  # none, circle, slash, zigzag

POLICY_VARIANTS = {
    "tiny": (1, 16),
    "small": (2, 32),
    "medium": (2, 64),
    "large": (3, 128),
    "l192": (3, 192),   # ~78k params  } bracket the CPU->NPU crossover
    "l256": (3, 256),   # ~136k params } (bench predicted ~130k at batch 32)
    "xlarge": (3, 512),
}

# ---------------------------------------------------------------------------
# Policy: synthetic states + scripted-heuristic teacher (all normalized space)
# ---------------------------------------------------------------------------

def gen_policy_data(n=250_000):
    # enemies spawn offscreen and converge -> positions slightly beyond [-1,1]
    posx = np.random.uniform(-1.25, 1.25, n)
    posy = np.random.uniform(-1.25, 1.25, n)
    ang = np.random.uniform(0, 2 * np.pi, n)
    spd = np.random.uniform(0, 1, n)
    velx = np.cos(ang) * spd
    vely = np.sin(ang) * spd
    # player sits (nearly) at screen centre
    px = np.random.uniform(-0.15, 0.15, n)
    py = np.random.uniform(-0.15, 0.15, n)
    dpx = (px - posx) / 2.0
    dpy = (py - posy) / 2.0
    dist = np.hypot(dpx, dpy)

    # 3 nearest neighbours: radii sorted ascending, some slots absent
    present = np.random.uniform(0, 1, (n, 3)) < np.array([0.92, 0.85, 0.78])
    radius = np.sort(np.random.uniform(0.02, 0.35, (n, 3)), axis=1)
    nang = np.random.uniform(0, 2 * np.pi, (n, 3))
    ndx_raw = np.cos(nang) * radius * present  # in W-fractions
    ndy_raw = np.sin(nang) * radius * present

    t_sin = np.sin(np.random.uniform(0, 2 * np.pi, n))

    feat = np.stack(
        [posx, posy, velx, vely, dpx, dpy, dist]
        + [np.clip(ndx_raw[:, k] / 0.25, -1, 1) if i == 0 else
           np.clip(ndy_raw[:, k] / 0.25, -1, 1)
           for k in range(3) for i in range(2)]
        + [t_sin],
        axis=1,
    ).astype(np.float32)

    # ---- teacher ----
    seek_len = np.maximum(dist, 1e-6)
    seekx, seeky = dpx / seek_len, dpy / seek_len
    sepx = np.zeros(n)
    sepy = np.zeros(n)
    for k in range(3):
        d2 = ndx_raw[:, k] ** 2 + ndy_raw[:, k] ** 2 + 1e-4
        w = present[:, k] * 0.004 / d2
        sepx -= ndx_raw[:, k] * w
        sepy -= ndy_raw[:, k] * w
    swirlx, swirly = -seeky * 0.55 * t_sin, seekx * 0.55 * t_sin
    wallx = -np.sign(posx) * np.maximum(0, np.abs(posx) - 0.8) * 2.0
    wally = -np.sign(posy) * np.maximum(0, np.abs(posy) - 0.8) * 2.0
    steerx = np.clip(seekx * 1.15 + sepx + swirlx + wallx, -1, 1)
    steery = np.clip(seeky * 1.15 + sepy + swirly + wally, -1, 1)
    # stationary player + autofire: enemies must threaten from range, in bursts
    fire = np.where((dist < 0.85) & (t_sin > 0.35), 0.85, -0.85)

    y = np.stack([steerx, steery, fire], axis=1).astype(np.float32)
    return feat, y


def build_policy(depth, width):
    # 1x1-conv formulation: the swarm rides the spatial H dimension
    # ([1, N, 1, feat]) instead of the batch dimension. Identical math and
    # memory layout to a per-enemy MLP, but supported by ALL delegates —
    # the GPU delegate's OpenGL backend rejects batch != 1 outright
    # (observed on the Poco X6 Pro, where broken libOpenCL forces GL).
    m = tf.keras.Sequential([tf.keras.layers.Input(shape=(None, 1, FEATURE_DIM))])
    for _ in range(depth):
        m.add(tf.keras.layers.Conv2D(width, 1, activation="relu"))
    m.add(tf.keras.layers.Conv2D(POLICY_OUT, 1, activation="tanh"))
    m.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss="mse")
    return m


# ---------------------------------------------------------------------------
# Gesture: synthetic strokes -> GS1 resample/normalize (mirrors gesture.cpp)
# ---------------------------------------------------------------------------

def resample_gs1(pts):
    pts = np.asarray(pts, dtype=np.float64)
    seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    total = seg.sum()
    if total < 1e-6:
        return None
    cum = np.concatenate([[0], np.cumsum(seg)])
    targets = np.linspace(0, total, GESTURE_PTS)
    out = np.empty((GESTURE_PTS, 2))
    out[0] = pts[0]
    out[-1] = pts[-1]
    idx = np.searchsorted(cum, targets[1:-1], side="right")
    idx = np.clip(idx, 1, len(pts) - 1)
    f = (targets[1:-1] - cum[idx - 1]) / np.maximum(cum[idx] - cum[idx - 1], 1e-9)
    out[1:-1] = pts[idx - 1] + (pts[idx] - pts[idx - 1]) * f[:, None]
    c = out.mean(axis=0)
    out -= c
    scale = max(out[:, 0].max() - out[:, 0].min(), out[:, 1].max() - out[:, 1].min()) * 0.5
    if scale < 1e-6:
        return None
    return (out / scale).astype(np.float32)


def _noise(pts, s):
    return pts + np.random.normal(0, s, pts.shape)


def gen_circle():
    extent = np.random.uniform(0.85, 1.2) * 2 * np.pi
    t = np.linspace(0, extent, np.random.randint(40, 90))
    if np.random.rand() < 0.5:
        t = t[::-1]
    t = t + np.random.uniform(0, 2 * np.pi)
    rx = np.random.uniform(0.5, 1.0)
    ry = rx * np.random.uniform(0.6, 1.4)
    pts = np.stack([np.cos(t) * rx, np.sin(t) * ry], axis=1)
    rot = np.random.uniform(0, np.pi)
    R = np.array([[np.cos(rot), -np.sin(rot)], [np.sin(rot), np.cos(rot)]])
    return _noise(pts @ R.T, np.random.uniform(0.015, 0.05))


def gen_slash():
    ang = np.random.uniform(0, 2 * np.pi)
    L = np.random.uniform(0.8, 2.0)
    d = np.array([np.cos(ang), np.sin(ang)])
    perp = np.array([-d[1], d[0]])
    bow = np.random.uniform(-0.10, 0.10) * L
    t = np.linspace(0, 1, np.random.randint(25, 60))
    pts = np.outer(t, d * L) + np.outer(4 * t * (1 - t), perp * bow)
    return _noise(pts, np.random.uniform(0.01, 0.04))


def gen_zigzag():
    k = np.random.randint(3, 7)
    base = np.random.uniform(0, 2 * np.pi)
    turn = np.random.uniform(np.deg2rad(50), np.deg2rad(110))
    pts = [np.zeros(2)]
    for i in range(k):
        ang = base + (turn if i % 2 == 0 else -turn) * np.random.uniform(0.8, 1.2)
        seg = np.random.uniform(0.25, 0.55)
        n = np.random.randint(6, 14)
        step = np.array([np.cos(ang), np.sin(ang)]) * (seg / n)
        for _ in range(n):
            pts.append(pts[-1] + step)
    pts = np.array(pts)
    return _noise(pts, np.random.uniform(0.01, 0.035))


def gen_none():
    kind = np.random.randint(0, 4)
    if kind == 0:  # smooth random walk
        steps = np.random.randint(30, 80)
        v = np.random.normal(0, 1, (steps, 2))
        for i in range(1, steps):
            v[i] = 0.9 * v[i - 1] + 0.4 * v[i]
        return np.cumsum(v * 0.05, axis=0)
    if kind == 1:  # partial arc (too short to be a circle)
        extent = np.random.uniform(np.deg2rad(40), np.deg2rad(200))
        t = np.linspace(0, extent, np.random.randint(20, 50)) + np.random.uniform(0, 2 * np.pi)
        return _noise(np.stack([np.cos(t), np.sin(t)], axis=1), 0.03)
    if kind == 2:  # single V (one turn — not a zigzag)
        a1 = np.random.uniform(0, 2 * np.pi)
        a2 = a1 + np.random.uniform(np.deg2rad(60), np.deg2rad(140)) * np.random.choice([-1, 1])
        p1 = np.outer(np.linspace(0, 1, 20), [np.cos(a1), np.sin(a1)])
        p2 = p1[-1] + np.outer(np.linspace(0, 1, 20), [np.cos(a2), np.sin(a2)])
        return _noise(np.concatenate([p1, p2]), 0.03)
    # hook / short scribble
    t = np.linspace(0, np.random.uniform(1, 3), 30)
    return _noise(np.stack([t * 0.3, np.sin(t * 3) * 0.1], axis=1), 0.05)


def gen_gesture_data(per_class=8000):
    gens = [gen_none, gen_circle, gen_slash, gen_zigzag]
    X, y = [], []
    for cls, g in enumerate(gens):
        made = 0
        while made < per_class:
            r = resample_gs1(g())
            if r is None:
                continue
            X.append(r)
            y.append(cls)
            made += 1
    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.int32)
    idx = np.random.permutation(len(X))
    return X[idx], y[idx]


def build_gesture():
    m = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=(GESTURE_PTS, 2)),
            tf.keras.layers.Conv1D(16, 5, padding="same", activation="relu"),
            tf.keras.layers.MaxPooling1D(2),
            tf.keras.layers.Conv1D(32, 5, padding="same", activation="relu"),
            tf.keras.layers.MaxPooling1D(2),
            tf.keras.layers.Conv1D(64, 3, padding="same", activation="relu"),
            tf.keras.layers.GlobalAveragePooling1D(),
            tf.keras.layers.Dense(32, activation="relu"),
            tf.keras.layers.Dense(GESTURE_CLASSES, activation="softmax"),
        ]
    )
    m.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return m


# ---------------------------------------------------------------------------
# Export + verify
# ---------------------------------------------------------------------------

def export_tflite(model, name, rep_data):
    saved = os.path.join(TMP, name)
    if os.path.exists(saved):
        shutil.rmtree(saved)
    model.export(saved)

    results = {}
    for prec in ("f32", "int8"):
        conv = tf.lite.TFLiteConverter.from_saved_model(saved)
        if prec == "int8":
            conv.optimizations = [tf.lite.Optimize.DEFAULT]

            def rep():
                for i in range(min(300, len(rep_data))):
                    yield [rep_data[i : i + 1].astype(np.float32)]

            conv.representative_dataset = rep
            conv.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
            conv.inference_input_type = tf.int8
            conv.inference_output_type = tf.int8
            # Per-tensor quant required: mtk-mdla_shim rejects per-channel weights
            # (ANEURALNETWORKS_BAD_DATA while adding operation), and NNAPI never
            # supported per-channel FULLY_CONNECTED at all.
            conv._experimental_disable_per_channel = True
        blob = conv.convert()
        fname = f"{name}_{prec}.tflite"
        for d in (OUT, ASSETS):
            with open(os.path.join(d, fname), "wb") as f:
                f.write(blob)
        results[prec] = len(blob)
        print(f"  exported {fname}: {len(blob)/1024:.1f} KiB")
    return results


def verify_tflite(path, batch=None):
    """Load, (optionally resize batch), invoke once, report ops + latency."""
    interp = tf.lite.Interpreter(model_path=path)
    inp = interp.get_input_details()[0]
    if batch is not None:
        shape = list(inp["shape"])
        # policy nets are [1, N, 1, feat] (N on the spatial axis); rank-2 would be axis 0
        shape[1 if len(shape) == 4 else 0] = batch
        interp.resize_tensor_input(inp["index"], shape)
    interp.allocate_tensors()
    inp = interp.get_input_details()[0]
    out = interp.get_output_details()[0]
    x = np.random.uniform(-1, 1, inp["shape"]).astype(np.float32)
    if inp["dtype"] == np.int8:
        s, zp = inp["quantization"]
        x = np.clip(np.round(x / s) + zp, -128, 127).astype(np.int8)
    interp.set_tensor(inp["index"], x)
    t0 = time.perf_counter()
    interp.invoke()
    ms = (time.perf_counter() - t0) * 1000
    ops = sorted({d["op_name"] for d in interp._get_ops_details()})  # noqa: SLF001
    print(f"  verify {os.path.basename(path)} in={list(inp['shape'])} "
          f"out={list(out['shape'])} {ms:.2f}ms host  ops={ops}")
    return ops


# ---------------------------------------------------------------------------

def main():
    t_start = time.time()
    manifest = {"feature_spec": "PF1", "gesture_spec": "GS1", "models": {}}

    print("=== policy data ===")
    X, y = gen_policy_data()
    print(f"  {X.shape} -> {y.shape}")

    Xc = X.reshape(-1, 1, 1, FEATURE_DIM)   # H=1 per training sample
    yc = y.reshape(-1, 1, 1, POLICY_OUT)
    for name, (depth, width) in POLICY_VARIANTS.items():
        print(f"=== policy_{name} ({depth}x{width}) ===")
        m = build_policy(depth, width)
        h = m.fit(Xc, yc, epochs=3, batch_size=1024, verbose=0,
                  validation_split=0.05)
        loss = h.history["val_loss"][-1]
        params = m.count_params()
        print(f"  params={params}  val_mse={loss:.4f}")
        sizes = export_tflite(m, f"policy_{name}", Xc)
        manifest["models"][f"policy_{name}"] = {
            "depth": depth, "width": width, "params": int(params),
            "val_mse": float(loss), "bytes": sizes,
        }

    print("=== gesture data ===")
    Xg, yg = gen_gesture_data()
    n_val = 4000
    print(f"  {Xg.shape}")
    print("=== gesture CNN ===")
    g = build_gesture()
    h = g.fit(Xg[:-n_val], yg[:-n_val], epochs=8, batch_size=256, verbose=0,
              validation_data=(Xg[-n_val:], yg[-n_val:]))
    acc = h.history["val_accuracy"][-1]
    print(f"  params={g.count_params()}  val_acc={acc:.4f}")
    sizes = export_tflite(g, "gesture", Xg)
    manifest["models"]["gesture"] = {
        "params": int(g.count_params()), "val_acc": float(acc), "bytes": sizes,
    }

    print("=== verification (host TFLite interpreter) ===")
    for name in POLICY_VARIANTS:
        for prec in ("f32", "int8"):
            verify_tflite(os.path.join(ASSETS, f"policy_{name}_{prec}.tflite"), batch=32)
    for prec in ("f32", "int8"):
        verify_tflite(os.path.join(ASSETS, f"gesture_{prec}.tflite"))

    with open(os.path.join(OUT, "model_manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"=== done in {time.time()-t_start:.0f}s -> {ASSETS} ===")


if __name__ == "__main__":
    main()
