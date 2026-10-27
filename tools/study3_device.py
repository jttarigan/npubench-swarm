#!/usr/bin/env python3
"""Study 3 device session helper (Mac side). Does every adb step of a
device session so the attended part is only the menu taps on the phone.

  python3 tools/study3_device.py <device> [--serial S] <command> [args]

devices: poco, tab, a56, s6lite, fold8 are found by model in `adb devices -l`; any
other name (e.g. redmi17) needs --serial. Data goes to runs_study3/<device>/.

Commands, in session order:
  status              model, Android, battery, display settings, app
                      installed?, files on the device; saved to
                      device_info_<utc>.txt
  backup              pull the app's files dir to preinstall_backup_<utc>/
                      and verify every file's size (needed before `install`
                      may uninstall)
  install [APK]       adb install -r; on a keystore mismatch, uninstall (only
                      if a backup matches the files now on the device), then
                      install fresh. Xiaomi: the user taps Install on the phone
  logcat start|stop   background capture of NPUBench/tflite/crash lines to
                      logcat_<utc>.txt; stop prints the key lines
  push-replay TRACE   push a run_*_trace.bin as replay_input.bin, verify it,
                      print the source tag the replay CSVs must carry. TRACE =
                      a path, or first:<device>:<variant> = that device's
                      first CPU F32 game trace of the variant (e.g.
                      first:poco:xlarge)
  pull                copy new or grown files to files/ (md5-verified)
  check               per config: runs, ticks, outcome, repeat identity, first
                      split from CPU F32; replays: which pushed trace they read
                      (source tag), bit-exact share against that trace

Every command appends a line to runs_study3/<device>/session_log.txt. Only
com.npubench.swarm and its files are touched; no device setting is changed.
Set ADB=/path/to/adb to override the SDK adb, STUDY3_RUNS=dir to move the data
(tests).
"""

import argparse
import csv
import datetime as dt
import hashlib
import json
import os
import re
import signal
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "analysis"))

PKG = "com.npubench.swarm"
REMOTE = f"/sdcard/Android/data/{PKG}/files"
RUNS = Path(os.environ.get("STUDY3_RUNS", ROOT / "runs_study3"))
APK = Path.home() / ".npubench-build/app/outputs/apk/debug/app-debug.apk"
ADB = os.environ.get("ADB", str(Path.home() / "Library/Android/sdk/platform-tools/adb"))
# model prefixes as `adb devices -l` prints them (it turns "-" into "_")
MODELS = {"poco": "2311DRK48G", "tab": ("SM_T97", "SM_T87"), "a56": "SM_A566",
          "s6lite": "SM_P615",   # Galaxy Tab S6 Lite LTE (Exynos 9611, no NPU)
          "fold8": "SM_F971"}    # Galaxy Z Fold 8 (SD 8 Elite Gen 5, borrowed; no NNAPI driver)
# "tab": Galaxy Tab S7+ (SM-T97x) or Tab S7 (SM-T87x), both Snapdragon 865+
WIB = dt.timezone(dt.timedelta(hours=7))
RUN_RE = re.compile(r"run_(\d{8}_\d{6})_(game|replay)_([a-z]+)_([a-z0-9]+)_(f32|int8)_"
                    r"([a-z]+)\.(csv|bin)$")
LOGCAT_FILTER = ["NPUBench:V", "tflite:V", "AndroidRuntime:E", "DEBUG:V", "libc:F", "*:S"]
TAP_NOTE = "If the phone shows an install prompt, tap Install (Xiaomi asks on every install)."


def stamps():
    u = dt.datetime.now(dt.timezone.utc)
    return u.strftime("%Y%m%dT%H%M%SZ"), u.astimezone(WIB).strftime("%Y-%m-%d %H:%M:%S WIB")


def run(args, check=True, timeout=900):
    p = subprocess.run(args, capture_output=True, text=True, timeout=timeout)
    if check and p.returncode != 0:
        sys.exit(f"FAILED ({p.returncode}): {' '.join(args)}\n{p.stdout}{p.stderr}")
    return p


def md5_file(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def fnv1a64_file(path):
    """Same hash as fnv1a64() in trace.cpp over the whole file (the app's
    replay source tag)."""
    h = 0xCBF29CE484222325
    with open(path, "rb") as f:
        data = f.read()
    for b in data:
        h = ((h ^ b) * 0x100000001B3) & 0xFFFFFFFFFFFFFFFF
    return h


# ------------------------------------------------------------------ device

class Device:
    def __init__(self, name, serial):
        self.name, self.serial = name, serial
        self.dir = RUNS / name
        self.dir.mkdir(parents=True, exist_ok=True)

    def adb(self, *args, check=True, timeout=900):
        return run([ADB, "-s", self.serial, *args], check, timeout)

    def sh(self, cmd, check=True):
        return self.adb("shell", cmd, check=check).stdout

    def listing(self):
        """{relative path: size} of every file under the app's files dir."""
        out = self.sh(f"if [ -d {REMOTE} ]; then find {REMOTE} -type f "
                      f"-exec stat -c '%s %n' {{}} +; fi")
        files = {}
        for line in out.splitlines():
            size, _, path = line.strip().partition(" ")
            if path.startswith(REMOTE + "/"):
                files[path[len(REMOTE) + 1:]] = int(size)
        return files

    def remote_md5(self, rel):
        out = self.sh(f"md5sum '{REMOTE}/{rel}'", check=False).split()
        return out[0] if out else None

    def installed(self):
        return "package:" in self.sh(f"pm path {PKG}", check=False)

    def log(self, cmd, note):
        utc, wib = stamps()
        with open(self.dir / "session_log.txt", "a") as f:
            f.write(f"{utc} | {wib} | {self.serial} | {cmd} | {note}\n")


def find_device(name, serial):
    rows = []
    # the serial may contain a space: a second mDNS transport to the same tablet is
    # listed as "adb-SERIAL-xxxx (2)._adb-tls-connect._tcp" (S6 Lite, 2026-10-13)
    for line in run([ADB, "devices", "-l"]).stdout.splitlines()[1:]:
        m = re.match(r"^(\S.*?)\s+(device|offline|unauthorized|authorizing|no permissions)\s*(.*)$",
                     line)
        if m:
            model = next((p[6:] for p in m.group(3).split() if p.startswith("model:")), "")
            rows.append((m.group(1), m.group(2), model))
    if serial:
        cands = [r for r in rows if r[0] == serial]
    elif name in MODELS:
        cands = [r for r in rows if r[2].startswith(MODELS[name])]  # str or tuple of prefixes
    else:
        sys.exit(f"no model known for '{name}': pass --serial (adb devices -l lists {rows})")
    ready = [r for r in cands if r[1] == "device"]
    if not ready:
        seen = "\n  ".join(" ".join(r) for r in rows) or "(none)"
        hint = (f"found it in state '{cands[0][1]}'" if cands else
                "not listed: is wireless debugging on? (`adb mdns services` shows paired phones)")
        sys.exit(f"{name}: {hint}\nadb devices:\n  {seen}")
    # the same phone can appear twice (mDNS name and ip:port); refuse two phones
    ids = {}
    for r in ready:
        sn = run([ADB, "-s", r[0], "shell", "getprop ro.serialno"]).stdout.strip()
        ids.setdefault(sn, r[0])
    if len(ids) > 1:
        sys.exit(f"{name}: several matching devices {list(ids.values())}; pass --serial")
    return Device(name, ready[0][0])


# ------------------------------------------------------------------ commands

def cmd_status(dev):
    props = ["ro.product.manufacturer", "ro.product.model", "ro.serialno",
             "ro.build.version.release", "ro.build.version.sdk", "ro.soc.manufacturer",
             "ro.soc.model", "ro.build.fingerprint"]
    out = [f"device {dev.name} serial {dev.serial}"]
    out += dev.sh("; ".join(f'echo "{p}=$(getprop {p})"' for p in props)).splitlines()
    batt = dev.sh("dumpsys battery", check=False)
    keep = ("AC powered", "USB powered", "Wireless powered", "status", "level", "temperature")
    out += ["battery " + ", ".join(l.strip() for l in batt.splitlines()
                                   if l.strip().split(":")[0] in keep)]
    # read-only: refresh-rate settings differ per vendor, unknown keys print null
    settings = [("system", "screen_off_timeout"), ("secure", "refresh_rate_mode"),
                ("system", "peak_refresh_rate"), ("system", "min_refresh_rate"),
                ("system", "user_refresh_rate"), ("global", "stay_on_while_plugged_in")]
    vals = dev.sh("; ".join(f"settings get {ns} {k}" for ns, k in settings)).splitlines()
    out += [f"setting {ns}/{k} = {v.strip()}" for (ns, k), v in zip(settings, vals)]
    if dev.installed():
        pkg = dev.sh(f"dumpsys package {PKG}", check=False)
        info = sorted({l.strip() for l in pkg.splitlines()
                       if l.strip().startswith(("versionName", "firstInstallTime",
                                                "lastUpdateTime"))})
        out.append("app installed: " + "; ".join(info))
    else:
        out.append("app NOT installed")
    files = dev.listing()
    out.append(f"files on device: {len(files)}, {sum(files.values()) / 1e6:.1f} MB")
    if "replay_input.bin" in files:
        out.append(f"replay_input.bin: {files['replay_input.bin']} bytes, "
                   f"md5 {dev.remote_md5('replay_input.bin')}")
    utc, wib = stamps()
    text = "\n".join([f"status at {utc} ({wib})"] + out) + "\n"
    (dev.dir / f"device_info_{utc}.txt").write_text(text)
    print(text, end="")
    dev.log("status", f"{len(files)} files; installed={dev.installed()}")


def backup_matches(dev, files):
    for marker in sorted(dev.dir.glob("preinstall_backup_*/BACKUP_OK.json")):
        if json.loads(marker.read_text())["files"] == files:
            return marker.parent
    return None


def cmd_backup(dev):
    files = dev.listing()
    if not files:
        print("nothing on the device to back up")
        dev.log("backup", "nothing to back up")
        return
    utc, _ = stamps()
    dest = dev.dir / f"preinstall_backup_{utc}"
    print(f"pulling {len(files)} files ({sum(files.values()) / 1e6:.1f} MB) to {dest}")
    dev.adb("pull", REMOTE, str(dest), timeout=7200)
    base = dest / "files" if (dest / "files").is_dir() else dest
    bad = [r for r, s in files.items()
           if not (base / r).is_file() or (base / r).stat().st_size != s]
    if bad:
        sys.exit(f"backup INCOMPLETE, {len(bad)} files missing or wrong size, e.g. {bad[:5]}")
    if dev.listing() != files:
        sys.exit("files on the device changed during the backup: run backup again")
    (dest / "BACKUP_OK.json").write_text(json.dumps(
        {"serial": dev.serial, "utc": utc, "base": base.name, "files": files}, indent=1))
    print(f"backup verified: {len(files)} files")
    dev.log("backup", f"{len(files)} files -> {dest.name}")


def cmd_install(dev, apk):
    apk = Path(apk) if apk else APK
    head = run(["git", "-C", str(ROOT), "rev-parse", "--short", "HEAD"]).stdout.strip()
    dirty = run(["git", "-C", str(ROOT), "status", "--porcelain", "app"]).stdout.strip()
    note = f"apk md5 {md5_file(apk)} built {dt.datetime.fromtimestamp(apk.stat().st_mtime):%Y-%m-%d %H:%M}, " \
           f"git {head}{' +uncommitted app/ changes' if dirty else ''}"
    print(note)
    print(TAP_NOTE)
    p = dev.adb("install", "-r", str(apk), check=False)
    msg = p.stdout + p.stderr
    if p.returncode != 0 and ("UPDATE_INCOMPATIBLE" in msg or "signatures do not match" in msg):
        files = dev.listing()
        if files and not backup_matches(dev, files):
            sys.exit("installed app has a different keystore and its files are not backed "
                     "up (or changed since the backup): run `backup` first")
        print("keystore differs: uninstalling (files are backed up), then installing fresh")
        dev.adb("uninstall", PKG)
        print(TAP_NOTE)
        p = dev.adb("install", str(apk), check=False)
        msg = p.stdout + p.stderr
        note += "; uninstalled first"
    if p.returncode != 0:
        if "USER_RESTRICTED" in msg:
            sys.exit("the phone's install prompt was declined or timed out: run install again "
                     "and tap Install")
        sys.exit(f"install FAILED:\n{msg}")
    if not dev.installed():
        sys.exit("install reported success but pm path finds no package")
    print("installed. Open the app once on the phone before push-replay (it creates its "
          "files dir).")
    dev.log("install", note)


def cmd_logcat(dev, action):
    pidfile = dev.dir / ".logcat.json"
    if action == "start":
        if pidfile.exists():
            sys.exit(f"capture already running ({json.loads(pidfile.read_text())}); stop it first")
        utc, _ = stamps()
        path = dev.dir / f"logcat_{utc}.txt"
        with open(path, "a") as f:
            p = subprocess.Popen([ADB, "-s", dev.serial, "logcat", "-v", "threadtime",
                                  *LOGCAT_FILTER], stdout=f, stderr=subprocess.STDOUT,
                                 stdin=subprocess.DEVNULL, start_new_session=True)
        pidfile.write_text(json.dumps({"pid": p.pid, "file": str(path)}))
        print(f"logcat capture pid {p.pid} -> {path}")
        dev.log("logcat start", path.name)
        return
    if not pidfile.exists():
        sys.exit("no capture running")
    info = json.loads(pidfile.read_text())
    try:
        os.kill(info["pid"], signal.SIGTERM)
    except ProcessLookupError:
        print("capture process had already ended (connection dropped?)")
    pidfile.unlink()
    text = Path(info["file"]).read_text(errors="replace")
    keys = ("screen ", "NNAPI devices", " loaded on ", "replay ", "FATAL", "Abort", "failed")
    print(f"{info['file']}: {len(text.splitlines())} lines; key lines:")
    for line in text.splitlines():
        if any(k in line for k in keys):
            print("  " + line[:200])
    dev.log("logcat stop", Path(info["file"]).name)


def resolve_trace(arg):
    if arg.startswith("first:"):
        _, src, variant = arg.split(":")
        hits = sorted((RUNS / src / "files").rglob(
            f"run_*_game_cpu_{variant}_f32_trace.bin"))
        if not hits:
            sys.exit(f"no CPU F32 {variant} game trace pulled from {src} yet")
        return hits[0]
    return Path(arg)


def cmd_push_replay(dev, arg):
    trace = resolve_trace(arg).resolve()
    with open(trace, "rb") as f:
        if f.read(4) != b"NPBT":
            sys.exit(f"{trace} is not an NPBT trace")
    if dev.sh(f"[ -d {REMOTE} ] && echo yes", check=False).strip() != "yes":
        sys.exit("the app's files dir does not exist yet: open the app once on the phone")
    tag = f"replay_input.bin#{fnv1a64_file(trace):016x}"
    md5 = md5_file(trace)
    dev.adb("push", str(trace), f"{REMOTE}/replay_input.bin")
    if dev.remote_md5("replay_input.bin") != md5:
        sys.exit("pushed replay_input.bin does not match (md5): push again")
    rel = os.path.relpath(trace, ROOT)
    pushed = dev.dir / "replay_inputs.json"     # every trace pushed, by source tag
    known = json.loads(pushed.read_text()) if pushed.exists() else {}
    known[tag] = {"source": rel, "md5": md5, "utc": stamps()[0]}
    pushed.write_text(json.dumps(known, indent=1))
    print(f"pushed {rel}\nreplay CSVs must show source {tag}")
    dev.log("push-replay", f"{rel} {tag}")


def cmd_pull(dev):
    base = dev.dir / "files"
    base.mkdir(exist_ok=True)
    started = stamps()[1]                # logged so peers on the same Wi-Fi can tag their runs
    new, grown, changed, nbytes = [], [], [], 0
    for rel, size in sorted(dev.listing().items()):
        dst = base / rel
        if dst.is_file() and dst.stat().st_size == size:
            continue
        (grown if dst.exists() else new).append(rel)
        dst.parent.mkdir(parents=True, exist_ok=True)
        part = dst.with_name(dst.name + ".part")
        dev.adb("pull", f"{REMOTE}/{rel}", str(part), timeout=3600)
        if md5_file(part) != dev.remote_md5(rel):
            changed.append(rel)          # still being written: keep the old copy
            part.unlink()
            continue
        part.replace(dst)
        nbytes += size
    print(f"pulled {len(new)} new + {len(grown)} grown files, {nbytes / 1e6:.1f} MB -> {base}")
    for rel in new:
        print("  + " + rel)
    for rel in changed:
        print("  ! changed while pulling (run still going?), pull again later: " + rel)
    dev.log("pull", f"{len(new)} new, {len(grown)} grown, {len(changed)} changing, "
                    f"{nbytes / 1e6:.1f} MB, started {started}")


def load_runs(base):
    runs = {}
    for p in sorted(base.rglob("run_*")):
        m = RUN_RE.match(p.name)
        if m:
            stamp, kind, backend, variant, prec, part, _ = m.groups()
            r = runs.setdefault(f"{stamp}_{kind}_{backend}_{variant}_{prec}",
                                {"kind": kind, "cfg": (backend, variant, prec), "files": {}})
            r["files"][part + "." + m.group(7)] = p
    return runs


def csv_rows(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def first_split(ta, tb):
    """First tick where outputs, then game state, differ between two traces."""
    out_t = st_t = None
    for a, b in zip(ta, tb):
        if out_t is None and not (a["out"].shape == b["out"].shape
                                  and (a["out"] == b["out"]).all()):
            out_t = a["tick"]
        if st_t is None and a["state"] != b["state"]:
            st_t = a["tick"]
        if out_t is not None and st_t is not None:
            break
    return out_t, st_t


def cmd_check(dev):
    import determinism as det
    runs = load_runs(dev.dir / "files")
    if not runs:
        sys.exit("no runs pulled yet")
    games = {}
    for rid, r in runs.items():
        if r["kind"] == "game":
            games.setdefault(r["cfg"], []).append((rid, r))
    refs = {}
    print("GAME runs (free-running)")
    for cfg in sorted(games):
        print(f"  {' '.join(cfg)}: {len(games[cfg])} runs")
        hashes = []
        for rid, r in games[cfg]:
            ticks = len(csv_rows(r["files"]["ticks.csv"])) if "ticks.csv" in r["files"] else 0
            tp = r["files"].get("trace.bin")
            if tp is None:
                print(f"    {rid}: tick rows {ticks}, NO TRACE (TRACE toggle off?)")
                continue
            _, tt = det.read_bin(tp)
            last = tt[-1] if tt else {}
            hashes.append(md5_file(tp))
            print(f"    {rid}: tick rows {ticks}, trace ticks {len(tt)}, "
                  f"{tp.stat().st_size / 1e6:.1f} MB, end wave {last.get('wave')} "
                  f"score {last.get('score')} hp {last.get('hp')}")
            if cfg[0] == "cpu" and cfg[2] == "f32":
                refs.setdefault(cfg[1], (rid, tt))
        if len(hashes) > 1:
            same = len(set(hashes)) == 1
            print(f"    repeats byte-identical: {'YES' if same else 'NO'} "
                  f"({len(set(hashes))} distinct of {len(hashes)})")
    print("\nFIRST SPLIT from the CPU F32 run of the same model (first runs)")
    for cfg in sorted(games):
        ref = refs.get(cfg[1])
        tp = games[cfg][0][1]["files"].get("trace.bin")
        if ref is None or tp is None or (cfg[0], cfg[2]) == ("cpu", "f32"):
            continue
        out_t, st_t = first_split(ref[1], det.read_bin(tp)[1])
        print(f"  {' '.join(cfg)}: outputs differ from tick {out_t}, state from tick "
              f"{st_t}" if out_t is not None or st_t is not None else
              f"  {' '.join(cfg)}: identical to CPU F32 over the shorter run")
    pushed = dev.dir / "replay_inputs.json"
    known = json.loads(pushed.read_text()) if pushed.exists() else {}
    reps = sorted((rid, r) for rid, r in runs.items() if r["kind"] == "replay")
    if reps:
        print("\nREPLAYS (bit-exact = outputs equal to the pushed trace's, value by value)")
    for rid, r in reps:
        rows = csv_rows(r["files"]["replay.csv"]) if "replay.csv" in r["files"] else []
        tags = sorted({row.get("source", "") for row in rows})
        src = known.get(tags[0]) if len(tags) == 1 else None
        line = f"  {rid}: rows {len(rows)}, " + (
            f"input {Path(src['source']).name}" if src else f"UNKNOWN source tags {tags}")
        rp = r["files"].get("replay.bin")
        if src and rp is not None:
            pairs, skipped = det.same_input_pairs(ROOT / src["source"], rp)
            s = det.summarize_outputs(pairs) or {}
            line += (f", ticks {s.get('ticks', 0)} (+{skipped} skipped), bit-exact values "
                     f"{s.get('bitexact_value_pct', float('nan')):.3f}%, max ulp "
                     f"{s.get('max_ulp', '-')}, fire flips {s.get('fire_flips', '-')}")
        print(line)
    dev.log("check", f"{len(runs)} runs")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("device")
    ap.add_argument("--serial")
    ap.add_argument("command", choices=["status", "backup", "install", "logcat",
                                        "push-replay", "pull", "check"])
    ap.add_argument("arg", nargs="?")
    a = ap.parse_args()
    if a.command == "check":            # offline: no adb needed
        dev = Device(a.device, a.serial or "offline")
        return cmd_check(dev)
    dev = find_device(a.device, a.serial)
    if a.command == "status":
        cmd_status(dev)
    elif a.command == "backup":
        cmd_backup(dev)
    elif a.command == "install":
        cmd_install(dev, a.arg)
    elif a.command == "logcat":
        if a.arg not in ("start", "stop"):
            sys.exit("logcat start|stop")
        cmd_logcat(dev, a.arg)
    elif a.command == "push-replay":
        if not a.arg:
            sys.exit("push-replay TRACE (path or first:<device>:<variant>)")
        cmd_push_replay(dev, a.arg)
    elif a.command == "pull":
        cmd_pull(dev)


if __name__ == "__main__":
    main()
