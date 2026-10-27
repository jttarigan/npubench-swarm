"""x86 kit: does this machine's CPU compute the same bits as the phones and the Mac?

Replays the two recorded reference games through LiteRT (the TensorFlow Python
package) with three kernel sets, compares every output value with the reference
outputs shipped in reference/, checks the portable kernel bit for bit, re-simulates
the recorded games with the host build of the game, and plays a few seeds in closed
loop. Everything is written to --out; nothing outside this folder is touched.

  python x86_replay.py [--out results/<name>] [--seeds 20] [--threads 4]

Needs numpy and tensorflow (2.20.0). The compiled helpers are optional: pp_cli
(portable kernel) and hostsim_strict / hostsim_precise / hostsim_contract (the game),
built by run.ps1 (Windows) or build_mac.sh (macOS) next to this file.
"""
import argparse
import csv
import hashlib
import json
import os
import platform
import struct
import subprocess
import sys
import tempfile
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
FEAT, OUT = 14, 3
FIRE = 0.35
TICK_HDR = struct.Struct("<IHHHH")
PIPE_HDR = struct.Struct("<iiii")
MODELS = ("xlarge", "tiny")
PRECS = ("f32", "int8")
MODES = ("xnnpack", "builtin", "ref")
EXE = ".exe" if os.name == "nt" else ""


# ------------------------------------------------------------------ files

def read_bin(path):
    """(kind, ticks): NPBT traces (features, outputs, state) or NPBR replays (outputs)."""
    data = open(path, "rb").read()
    kind = data[:4].decode("ascii", "replace")
    if kind not in ("NPBT", "NPBR"):
        raise ValueError(f"{path}: not an NPBT/NPBR file")
    ver, fd, od = struct.unpack_from("<III", data, 4)
    off, ticks = 16, []
    while off < len(data):
        tick, wave, batch, alive, _ = TICK_HDR.unpack_from(data, off)
        off += TICK_HDR.size
        t = {"tick": tick, "wave": wave, "batch": batch, "alive": alive}
        if kind == "NPBT":
            t["feat"] = np.frombuffer(data, "<f4", alive * fd, off).reshape(alive, fd)
            off += 4 * alive * fd
        t["out"] = np.frombuffer(data, "<f4", alive * od, off).reshape(alive, od)
        off += 4 * alive * od
        if kind == "NPBT":
            off += 2 * alive                      # alive indices
            s0 = off
            (E,) = struct.unpack_from("<H", data, off)
            off += 2 + 9 * E + 12                 # enemies, hp/score/bullets/rng
            t["state"] = data[s0:off]
        ticks.append(t)
    return kind, ticks


def write_replay(path, ticks, outs):
    with open(path, "wb") as f:
        f.write(b"NPBR" + struct.pack("<III", 1, FEAT, OUT))
        for t, o in zip(ticks, outs):
            f.write(TICK_HDR.pack(t["tick"], t["wave"], t["batch"], t["alive"], 0))
            f.write(np.ascontiguousarray(o, np.float32).tobytes())


def outputs_of(path):
    return np.concatenate([t["out"] for t in read_bin(path)[1] if t["alive"] > 0])


def md5(path):
    return hashlib.md5(open(path, "rb").read()).hexdigest()


# ------------------------------------------------------------------ LiteRT

def lround(x):
    """C lroundf: half away from zero (the app's int8 input quantization)."""
    return np.sign(x) * np.floor(np.abs(x) + 0.5)


class LiteBackend:
    def __init__(self, tf, model, prec, mode, threads):
        R = tf.lite.experimental.OpResolverType
        resolver = {"xnnpack": R.AUTO, "builtin": R.BUILTIN_WITHOUT_DEFAULT_DELEGATES, "ref": R.BUILTIN_REF}[mode]
        path = os.path.join(HERE, "models", f"policy_{model}_{prec}.tflite")
        self.it = tf.lite.Interpreter(model_path=path, num_threads=threads, experimental_op_resolver_type=resolver)
        self.inp = self.it.get_input_details()[0]
        self.out = self.it.get_output_details()[0]
        self.int8 = self.inp["dtype"] == np.int8
        self.in_scale, self.in_zero = self.inp["quantization"]
        self.out_scale, self.out_zero = self.out["quantization"]
        self.batch = -1

    def run(self, feat, batch):
        alive = feat.shape[0]
        if batch != self.batch:
            self.it.resize_tensor_input(self.inp["index"], [1, batch, 1, FEAT])
            self.it.allocate_tensors()
            self.batch = batch
        x = np.zeros((1, batch, 1, FEAT), np.float32)
        x[0, :alive, 0, :] = feat
        if self.int8:
            x = np.clip(lround(x / np.float32(self.in_scale)) + self.in_zero, -128, 127).astype(np.int8)
        self.it.set_tensor(self.inp["index"], x)
        self.it.invoke()
        y = self.it.get_tensor(self.out["index"])
        if self.int8:
            y = ((y.astype(np.int32) - self.out_zero) * np.float32(self.out_scale)).astype(np.float32)
        return np.ascontiguousarray(y[0, :alive, 0, :]).astype(np.float32)


# ------------------------------------------------------------------ comparison

def compare(a, b):
    """Two output arrays over the same rows: bit identity and the size of the gap."""
    a = a.reshape(-1, OUT); b = b.reshape(-1, OUT)
    if a.shape != b.shape:
        return {"identical": False, "note": f"shape {a.shape} vs {b.shape}"}
    diff = a != b
    d = np.abs(a.astype(np.float64) - b.astype(np.float64))
    same_sign = np.sign(a) == np.sign(b)
    ulp = np.abs(a.view(np.int32).astype(np.int64) - b.view(np.int32).astype(np.int64))
    ulp_med = int(np.median(ulp[diff & same_sign])) if (diff & same_sign).any() else 0
    flips = int(((a[:, 2] > FIRE) != (b[:, 2] > FIRE)).sum())
    return {"identical": bool(not diff.any()), "values": int(a.size), "values_differing": int(diff.sum()),
            "rows_differing": int(diff.any(axis=1).sum()), "max_abs_diff": float(d.max()),
            "median_ulp": ulp_med, "fire_flips": flips}


# ------------------------------------------------------------------ host simulation

def exe(name):
    p = os.path.join(HERE, name + EXE)
    return p if os.path.exists(p) else None


def hostsim_variants():
    return [(n, exe(n)) for n in ("hostsim_strict", "hostsim_precise", "hostsim_contract", "hostsim") if exe(n)]


def pipe_game(hs, prefix, policy, seed=None):
    """Play one game: policy(feat, batch) -> outputs per tick. Returns (ticks, wave, score, hp)."""
    env = dict(os.environ)
    if seed is not None:
        env["HOSTSIM_SEED"] = str(seed)
    p = subprocess.Popen([hs, prefix, "--pipe"], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                         stderr=subprocess.PIPE, env=env)
    n = 0
    while True:
        hdr = p.stdout.read(PIPE_HDR.size)
        if len(hdr) < PIPE_HDR.size:
            raise RuntimeError("hostsim ended early: " + p.stderr.read().decode(errors="replace"))
        tick, wave, batch, alive = PIPE_HDR.unpack(hdr)
        if tick < 0:
            break
        feat = np.frombuffer(p.stdout.read(4 * alive * FEAT), "<f4").reshape(alive, FEAT) if alive else np.zeros((0, FEAT), np.float32)
        out = policy(feat, batch, n) if alive else np.zeros((0, OUT), np.float32)
        p.stdin.write(np.ascontiguousarray(out, np.float32).tobytes())
        p.stdin.flush()
        n += 1
    p.stdin.close()
    p.wait()
    return n, wave, batch, alive       # after the end header: wave, score, hp


def state_hashes(prefix):
    with open(prefix + "_ticks.csv") as f:
        return [r["state_hash"] for r in csv.DictReader(f)]


def validate(hs, trace_path, tmp):
    """Feed the recorded outputs to the host game; the state must follow the recording."""
    _, T = read_bin(trace_path)
    prefix = os.path.join(tmp, "val")
    mism = [0]

    def policy(feat, batch, i):
        t = T[i]
        if t["alive"] != feat.shape[0] or not np.array_equal(t["feat"], feat):
            mism[0] += 1
        return t["out"]

    try:
        n, *_ = pipe_game(hs, prefix, policy)
    except RuntimeError as e:
        return {"ok": False, "note": str(e)[:200]}
    _, H = read_bin(prefix + "_trace.bin")
    same = sum(a["state"] == b["state"] for a, b in zip(T, H))
    first = next((k for k, (a, b) in enumerate(zip(T, H)) if a["state"] != b["state"]), None)
    equal = open(prefix + "_trace.bin", "rb").read() == open(trace_path, "rb").read()
    return {"ok": bool(equal), "ticks": len(T), "state_identical_ticks": same, "first_state_diff_tick": first,
            "feature_mismatches": mism[0], "file_equal": bool(equal)}


def seed_of(index):
    return None if index == 0 else ((index * 2654435761) & 0xFFFFFFFF) | 1


# ------------------------------------------------------------------ main

def env_info(tf):
    info = {"platform": platform.platform(), "machine": platform.machine(), "processor": platform.processor(),
            "cpu_count": os.cpu_count(), "python": sys.version.split()[0], "numpy": np.__version__,
            "tensorflow": tf.__version__, "hostname": platform.node()}
    try:
        if os.name == "nt":
            import winreg
            k = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"HARDWARE\DESCRIPTION\System\CentralProcessor\0")
            info["cpu_name"] = winreg.QueryValueEx(k, "ProcessorNameString")[0].strip()
        elif sys.platform == "darwin":
            info["cpu_name"] = subprocess.check_output(["sysctl", "-n", "machdep.cpu.brand_string"]).decode().strip()
        else:
            info["cpu_name"] = next(l.split(":", 1)[1].strip() for l in open("/proc/cpuinfo") if l.startswith("model name"))
    except Exception:
        info["cpu_name"] = info["processor"]
    return info


def main(argv):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=None)
    ap.add_argument("--seeds", type=int, default=20)
    ap.add_argument("--threads", type=int, default=4)
    a = ap.parse_args(argv)
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    import tensorflow as tf
    info = env_info(tf)
    out = a.out or os.path.join(HERE, "results", platform.node().split(".")[0] or "host")
    os.makedirs(os.path.join(out, "seeds"), exist_ok=True)
    res = {"env": info, "started": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "replays": {}, "portable": {},
           "validate": {}, "seeds": {}, "compare": {}}
    print("machine:", info["cpu_name"], "|", info["platform"], "| TensorFlow", info["tensorflow"], flush=True)

    traces = {m: read_bin(os.path.join(HERE, "traces", f"{m}_trace.bin"))[1] for m in MODELS}
    feats = {m: np.concatenate([t["feat"] for t in traces[m] if t["alive"] > 0]) for m in MODELS}

    # 1. replays through LiteRT, three kernel sets
    for m in MODELS:
        for prec in PRECS:
            for mode in MODES:
                name = f"replay_{m}_{prec}_{mode}"
                t0 = time.time()
                b = LiteBackend(tf, m, prec, mode, a.threads)
                outs = [b.run(t["feat"], t["batch"]) for t in traces[m]]
                path = os.path.join(out, name + ".bin")
                write_replay(path, traces[m], outs)
                res["replays"][name] = {"md5": md5(path), "seconds": round(time.time() - t0, 1)}
                print(f"  {name}: {res['replays'][name]['seconds']} s", flush=True)

    # 2. portable kernel
    pp = exe("pp_cli")
    if pp:
        for m in MODELS:
            with tempfile.TemporaryDirectory() as d:
                np.ascontiguousarray(feats[m], "<f4").tofile(os.path.join(d, "in.f32"))
                subprocess.run([pp, os.path.join(HERE, "models", f"policy_{m}_f32.nppw"), os.path.join(d, "in.f32"),
                                os.path.join(out, f"pp_{m}.f32")], check=True, capture_output=True)
            res["portable"][m] = {"md5": md5(os.path.join(out, f"pp_{m}.f32"))}
        print("  portable kernel done", flush=True)
    else:
        print("  no pp_cli: portable kernel skipped", flush=True)

    # 3. re-simulation of the recorded games, 4. closed loop over seeds
    variants = hostsim_variants()
    if not variants:
        print("  no hostsim: re-simulation and seeds skipped", flush=True)
    for vname, hs in variants:
        with tempfile.TemporaryDirectory() as d:
            for m in MODELS:
                r = validate(hs, os.path.join(HERE, "traces", f"{m}_trace.bin"), d)
                res["validate"][f"{vname}/{m}"] = r
                print(f"  {vname} re-simulates {m}: {'byte-identical' if r['ok'] else r}", flush=True)
    if variants and a.seeds > 0:
        vname, hs = variants[0]
        backends = {f"{m}_{mode}": LiteBackend(tf, m, "f32", mode, a.threads) for m in MODELS for mode in ("xnnpack", "builtin")}
        with tempfile.TemporaryDirectory() as d:
            for m in MODELS:
                for mode in ("xnnpack", "builtin"):
                    b = backends[f"{m}_{mode}"]
                    for i in range(a.seeds):
                        prefix = os.path.join(d, f"{m}_{mode}_{i}")
                        n, wave, score, hp = pipe_game(hs, prefix, lambda f, batch, k: b.run(f, batch), seed_of(i))
                        hashes = state_hashes(prefix)
                        with open(os.path.join(out, "seeds", f"{m}_{mode}_{i}.txt"), "w") as f:
                            f.write("\n".join(hashes) + "\n")
                        res["seeds"][f"{m}_{mode}_{i}"] = {"hostsim": vname, "ticks": n, "wave": wave, "score": score, "hp": hp}
                    print(f"  seeds {m} {mode}: {a.seeds} games", flush=True)

    # 5. compare with the shipped references
    refdir = os.path.join(HERE, "reference")
    devices = {"R1_android_cpu_xlarge_f32": np.concatenate([t["out"] for t in traces["xlarge"] if t["alive"] > 0]),
               "T1_android_cpu_tiny_f32": np.concatenate([t["out"] for t in traces["tiny"] if t["alive"] > 0])}
    for fn in sorted(os.listdir(os.path.join(refdir, "device"))):
        if fn.endswith(".bin"):
            devices[fn[:-4]] = outputs_of(os.path.join(refdir, "device", fn))
    bundles = [b for b in sorted(os.listdir(refdir)) if b != "device" and os.path.isdir(os.path.join(refdir, b))
               and os.path.abspath(os.path.join(refdir, b)) != os.path.abspath(out)]
    for name in res["replays"]:
        here = outputs_of(os.path.join(out, name + ".bin"))
        m, prec = name.split("_")[1], name.split("_")[2]
        cmp = {}
        for dn, dv in devices.items():
            if f"_{m}_{prec}" in dn:
                cmp["device:" + dn] = compare(here, dv)
        for b in bundles:
            p = os.path.join(refdir, b, name + ".bin")
            if os.path.exists(p):
                cmp[b + ":" + name] = compare(here, outputs_of(p))
        res["compare"][name] = cmp
    for m in MODELS:
        p = os.path.join(out, f"pp_{m}.f32")
        if os.path.exists(p):
            here = np.fromfile(p, "<f4")
            for b in bundles:
                q = os.path.join(refdir, b, f"pp_{m}.f32")
                if os.path.exists(q):
                    res["compare"][f"pp_{m}"] = {b: compare(here, np.fromfile(q, "<f4"))}
    for key, g in res["seeds"].items():
        for b in bundles:
            q = os.path.join(refdir, b, "seeds", key + ".txt")
            if os.path.exists(q):
                mine = open(os.path.join(out, "seeds", key + ".txt")).read().split()
                theirs = open(q).read().split()
                first = next((k for k, (x, y) in enumerate(zip(mine, theirs)) if x != y), None)
                if first is None and len(mine) != len(theirs):
                    first = min(len(mine), len(theirs))
                g[b] = {"identical": first is None, "first_state_diff_tick": first}
                rj = json.load(open(os.path.join(refdir, b, "results.json")))["seeds"].get(key)
                if rj:
                    g[b]["outcome_equal"] = (rj["ticks"], rj["score"], rj["hp"]) == (g["ticks"], g["score"], g["hp"])

    res["finished"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    json.dump(res, open(os.path.join(out, "results.json"), "w"), indent=1)
    write_md(res, out, bundles)
    print("wrote", out, flush=True)


def write_md(res, out, bundles):
    L = [f"# x86 kit results: {res['env']['cpu_name']}", "",
         f"{res['env']['platform']}, Python {res['env']['python']}, TensorFlow {res['env']['tensorflow']}, "
         f"numpy {res['env']['numpy']}, {res['env']['cpu_count']} CPUs; {res['started']} to {res['finished']}.", "",
         "## Replays over the recorded inputs (values differing of all; max abs diff; median ULP; fire flips)", "",
         "| replay | reference | identical | differing | max abs diff | median ULP | flips |", "|---|---|---|---|---|---|---|"]
    for name, cmp in res["compare"].items():
        for ref, c in cmp.items():
            if "values" in c:
                L.append(f"| {name} | {ref} | {'YES' if c['identical'] else 'no'} | {c['values_differing']:,} of {c['values']:,} "
                         f"| {c['max_abs_diff']:.3g} | {c['median_ulp']} | {c['fire_flips']} |")
            else:
                L.append(f"| {name} | {ref} | no | {c.get('note', '')} | | | |")
    L += ["", "## Re-simulation of the recorded games (host build of the game, recorded outputs fed in)", ""]
    for k, r in res["validate"].items():
        L.append(f"- {k}: " + ("byte-identical trace" if r.get("ok") else f"differs: {r}"))
    if res["seeds"]:
        L += ["", f"## Closed loop over seeds ({len(res['seeds'])} games)", ""]
        for b in bundles:
            rows = [(k, g[b]) for k, g in res["seeds"].items() if b in g]
            if rows:
                same = sum(1 for _, x in rows if x["identical"])
                outc = sum(1 for _, x in rows if x.get("outcome_equal"))
                L.append(f"- against {b}: {same} of {len(rows)} games identical on every tick; outcome equal in {outc}")
    with open(os.path.join(out, "RESULTS.md"), "w") as f:
        f.write("\n".join(L) + "\n")


if __name__ == "__main__":
    main(sys.argv[1:])
