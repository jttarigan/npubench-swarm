"""Study 3 validation: re-simulate a recorded device game on the Mac.

  python3 tools/hostsim/validate_trace.py RUN_trace.bin OUT_PREFIX

Runs tools/hostsim/hostsim (the game's C++ simulation built for the Mac) in pipe
mode and feeds it the model outputs recorded in the trace, tick by tick. Prints
whether the features match and whether the Mac's trace file equals the device's.
Results for all study 3 games: analysis/out_study3/validation.txt.
"""
import subprocess, struct, sys, numpy as np
sys.path.insert(0, "analysis")
from determinism import read_bin
ref = sys.argv[1]; prefix = sys.argv[2]
_, T = read_bin(ref)
p = subprocess.Popen([__import__("os").environ.get("HOSTSIM","tools/hostsim/hostsim"), prefix, "--pipe"], stdin=subprocess.PIPE, stdout=subprocess.PIPE)
i = 0; feat_mismatch = 0
while True:
    tick, wave, batch, alive = struct.unpack("<4i", p.stdout.read(16))
    if tick < 0: print("end: wave", wave, "score", batch, "hp", alive); break
    f = np.frombuffer(p.stdout.read(alive * 14 * 4), "<f4").reshape(alive, 14)
    t = T[i] if i < len(T) else None
    if t is None or t["alive"] != alive:
        print("diverged structurally at tick", tick); p.kill(); break
    if t["tick"] != tick or not np.array_equal(t["feat"], f):
        feat_mismatch += 1
        if feat_mismatch == 1: print("first feature mismatch at tick", tick)
    p.stdin.write(np.ascontiguousarray(t["out"], "<f4").tobytes()); p.stdin.flush()
    i += 1
p.wait()
print("ticks", i, "of", len(T), "feature mismatches", feat_mismatch)
try:
    _, H = read_bin(prefix + "_trace.bin")
except Exception:
    H = []
same = sum(a["state"] == b["state"] for a, b in zip(T, H))
first = next((k for k, (a, b) in enumerate(zip(T, H)) if a["state"] != b["state"]), None)
print("first state difference at tick", first)
print("state identical on", same, "of", len(T), "ticks; files equal:", open(prefix + "_trace.bin","rb").read() == open(ref,"rb").read())
