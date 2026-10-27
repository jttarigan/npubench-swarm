"""Study 3: the guard width tested on rows it was not sized on.

The guard D in study3_guard.py is sized from the replay of one XL game. Every
free game records its own feature rows and the outputs its backend computed
along its own trajectory, so the portable kernel (tools/portable_policy) can be
run on those rows on the Mac: the largest difference per distinct float32 game
is what D has to exceed. Byte-identical games are counted once; the prevented
games of the on-device experiment are left out (their outputs are rounded).

  python3 analysis/study3_guard_oos.py > analysis/out_study3/guard_oos.csv
"""
import csv
import glob
import hashlib
import os
import subprocess
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from determinism import read_bin  # noqa: E402

PP_CLI = "tools/portable_policy/pp_cli"
WEIGHTS = {"xlarge": "app/src/main/assets/models/policy_xlarge_f32.nppw",
           "tiny": "app/src/main/assets/models/policy_tiny_f32.nppw"}


def portable(model, feat):
    with tempfile.TemporaryDirectory() as d:
        np.ascontiguousarray(feat, "<f4").tofile(f"{d}/in.f32")
        subprocess.run([PP_CLI, WEIGHTS[model], f"{d}/in.f32", f"{d}/out.f32"],
                       check=True, capture_output=True)
        return np.fromfile(f"{d}/out.f32", "<f4").reshape(-1, 3)


def main(argv):
    seen = set()
    w = csv.writer(sys.stdout, lineterminator="\n")
    w.writerow(["model", "device", "backend", "rows", "max_abs_diff_vs_portable", "trace"])
    for f in sorted(glob.glob("runs_study3/*/files/run_*_game_*_f32_trace.bin")):
        dev = f.split("/")[1]
        back, model = os.path.basename(f).split("_game_")[1].split("_")[:2]
        if dev.endswith("_prevent") or model not in WEIGHTS:
            continue
        h = hashlib.md5(open(f, "rb").read()).hexdigest()
        if h in seen:
            continue
        seen.add(h)
        _, ticks = read_bin(f)
        ticks = [t for t in ticks if t["alive"] > 0]
        feat = np.concatenate([t["feat"] for t in ticks])
        out = np.concatenate([t["out"] for t in ticks]).astype(np.float64)
        d = np.abs(out - portable(model, feat).astype(np.float64)).max()
        w.writerow([model, dev, back, len(out), f"{d:.4g}", os.path.basename(f)])


if __name__ == "__main__":
    main(sys.argv[1:])
