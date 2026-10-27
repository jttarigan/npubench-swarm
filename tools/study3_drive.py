"""Study 3: drive the NPUBench menu over adb on a device that accepts
injected taps (Samsung: Tab S6 Lite 2026-10-13; NOT the POCO, which refuses
INJECT_EVENTS). Button centres come from Engine::renderMenu (internal
240x534); the screen mapping is the app's own "screen WxH internal ... scale S"
logcat line: sx = offX + S*ix, sy = offY + S*iy. Defaults are the S6 Lite's
1200x2000 (scale 3, offset 240,199); pass --screen W H to recompute.
A bare tap during play is rejected as a gesture (stroke < 8 points), so the
loop-stop taps cannot change a run.

  python3 tools/study3_drive.py [--model SM_P615] [--screen 1200 2000] CMD
    taps NAME [NAME ...]   tap menu buttons by name (cpu gpu npu t xl f32 int8
                           loop trace replay start neutral ...)
    loop3                  START RUN, wait for 3 looped runs (logcat "wave 1:"
                           / "wave 10:" lines), tap the loop away, confirm menu
    replay                 tap REPLAY INPUT, wait for "replay ... done", tap for menu
    tomenu                 stop a loop left running (neutral taps until the menu shows)
    shot OUT.png           screenshot;  ismenu  print whether the menu is showing
    refshot                save the menu reference for --model (menu must be showing)
  The menu check compares the START/BENCH button band with the reference shot
  next to this script (take a new one with `shot` for another screen size).
  Session 2026-10-13 on the S6 Lite: 9 runs driven this way, 3 configs x 3.
"""
import subprocess, sys, time, io, re, os
import numpy as np
from PIL import Image

ADB = os.path.expanduser("~/Library/Android/sdk/platform-tools/adb")
def _opt(name, n, default):
    """--name v1 .. vn from argv (removed), else default."""
    if name in sys.argv:
        i = sys.argv.index(name)
        vals = sys.argv[i + 1:i + 1 + n]
        del sys.argv[i:i + 1 + n]
        return vals
    return default


MODEL = _opt("--model", 1, ["SM_P615"])[0]     # as `adb devices -l` prints it
_W, _H = (int(v) for v in _opt("--screen", 2, ["1200", "2000"]))
SC = max(1, min(_W // 240, _H // 534))
OFFX, OFFY = (_W - 240 * SC) // 2, (_H - 534 * SC) // 2
# a letterboxed app window starts at --origin X Y on the display (Fold 8 inner
# screen: window [639,0]-[1809,1848], `dumpsys window` Frames); --display picks
# the screencap display id when the device has several
_OX, _OY = (int(v) for v in _opt("--origin", 2, ["0", "0"]))
OFFX, OFFY = OFFX + _OX, OFFY + _OY
DISPLAY = _opt("--display", 1, [""])[0]


def _serial():
    out = subprocess.run([ADB, "devices", "-l"], capture_output=True, text=True, timeout=20).stdout
    for line in out.splitlines()[1:]:
        m = re.match(r"^(.*?)\s+device\s+product:.*model:" + re.escape(MODEL), line)
        if m:
            return m.group(1)
    raise SystemExit(f"no device with model {MODEL} on adb")
S = _serial()
import os
# menu reference shot per model (take one with `refshot` while the menu shows)
REF = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                   "study3_menu_s6lite.png" if MODEL == "SM_P615" else f"study3_menu_{MODEL}.png")

# internal-coordinate button centres from Engine::renderMenu
BTN = {
    "cpu": (94, 117), "gpu": (150, 117), "npu": (206, 117),
    "t": (79, 145), "xl": (223, 145),   # MODEL row: T = variant 0, XL = 6
    "f32": (94, 173), "int8": (150, 173),
    "glow_on": (94, 201), "glow_off": (150, 201), "loop": (208, 201),
    "trace": (94, 229), "replay": (178, 229),
    "start": (120, 265),
    "neutral": (120, 450),   # bottom text area, no button
}


def adb(*a, timeout=30):
    return subprocess.run([ADB, "-s", S, *a], capture_output=True, text=True, timeout=timeout).stdout


def tap(name):
    ix, iy = BTN[name]
    adb("shell", "input", "tap", str(OFFX + SC * ix), str(OFFY + SC * iy))


def shot(path=None):
    disp = ["-d", DISPLAY] if DISPLAY else []
    png = subprocess.run([ADB, "-s", S, "exec-out", "screencap", "-p", *disp],
                         capture_output=True, timeout=60).stdout
    if path:
        open(path, "wb").write(png)
    return np.asarray(Image.open(io.BytesIO(png)).convert("RGB"))


_ref = None
def is_menu(img=None):
    """Compare the START RUN / BENCH button band with the reference menu shot."""
    global _ref
    if _ref is None:
        _ref = np.asarray(Image.open(REF).convert("RGB"))
    if img is None:
        img = shot()
    # internal rows 250..310, columns 20..220 (on the S6 Lite: y 950..1130, x 300..900)
    ys, xs = slice(OFFY + SC * 250, OFFY + SC * 310), slice(OFFX + SC * 20, OFFX + SC * 220)
    a = img[ys, xs].astype(int)
    b = _ref[ys, xs].astype(int)
    return np.abs(a - b).max() < 24


def logcount(pattern):
    out = adb("shell", f"logcat -d -s NPUBench:V | grep -c '{pattern}'")
    try:
        return int(out.strip().splitlines()[-1])
    except Exception:
        return 0


def wait_count(pattern, target, every=3.0, limit=900):
    t0 = time.time()
    while time.time() - t0 < limit:
        n = logcount(pattern)
        if n >= target:
            return n
        time.sleep(every)
    raise SystemExit(f"timeout waiting for {pattern} >= {target}")


def ts():
    return time.strftime("%H:%M:%S")


def loop3():
    w1 = logcount("wave 1:")
    w10 = logcount("wave 10:")
    print(ts(), "baseline wave1", w1, "wave10", w10)
    tap("start")
    print(ts(), "START tapped")
    for k in range(1, 4):
        wait_count("wave 1:", w1 + k)
        print(ts(), f"run {k} started")
    wait_count("wave 10:", w10 + 3)
    print(ts(), "run 3 in wave 10; tapping the neutral spot once a second until the menu shows")
    t0 = time.time()
    while time.time() - t0 < 120:
        for _ in range(3):
            tap("neutral")
            time.sleep(1.0)
        if is_menu():
            print(ts(), "menu is back")
            n = logcount("wave 1:")
            print(ts(), "runs started for this config:", n - w1)
            return
    raise SystemExit("menu did not come back")


def tomenu(limit=200):
    """Back to the menu from a loop left running: taps during play are rejected
    as gestures, the one in the between-games window stops the loop."""
    t0 = time.time()
    while time.time() - t0 < limit:
        if is_menu():
            print(ts(), "menu is showing")
            return
        for _ in range(3):
            tap("neutral")
            time.sleep(1.0)
    raise SystemExit("menu did not come back")


def replay():
    d = logcount("replay .* done")
    tap("replay")
    print(ts(), "REPLAY INPUT tapped")
    wait_count("replay .* done", d + 1, every=2.0, limit=600)
    print(ts(), "replay done")
    time.sleep(1.0)
    tap("neutral")
    time.sleep(1.5)
    print(ts(), "menu:", is_menu())


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "taps":
        for n in sys.argv[2:]:
            tap(n)
            time.sleep(0.6)
        print(ts(), "tapped", sys.argv[2:])
    elif cmd == "loop3":
        loop3()
    elif cmd == "replay":
        replay()
    elif cmd == "tomenu":
        tomenu()
    elif cmd == "refshot":
        shot(REF); print("wrote menu reference", REF)
    elif cmd == "shot":
        shot(sys.argv[2]); print("wrote", sys.argv[2])
    elif cmd == "ismenu":
        print("menu:", is_menu())
