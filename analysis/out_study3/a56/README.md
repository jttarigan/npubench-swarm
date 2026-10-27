# Study 3, Galaxy A56 (2026-10-16, 18:57 to 19:21 WIB)

Device: SM-A566B, Samsung Exynos 1580 (s5e8855), Xclipse 540 GPU (AMD RDNA-based),
Android 15 (A566BXXU1AYB7). Borrowed; one-hour slot from Main, user away during the
run. NNAPI lists only `nnapi-reference` (the app's picker took none), so there is no
NPU path: CPU F32, CPU INT8 and GPU F32 per model, as on the Tab S6 Lite.
Screen 1080x2340, internal 240x534 at scale 4; menu reference
`tools/study3_menu_SM_A566B.png`.

App: the study 3 APK (md5 cd4d9034...), fresh install, uninstalled afterwards (data
pulled and md5-verified first, 67 files). Unattended over wireless adb with
`tools/study3_drive.py --model SM_A566B --screen 1080 2340`; one script ran XL free
runs, XL replays, then tiny free runs and tiny replays (`runs_study3/a56/seq_20261016.txt`,
logcat `logcat_20261016T115259Z.txt`, `session_log.txt`). The user unlocked the phone
once; nothing else by hand. Settings: stay_on set to 7 for the run and restored to
0; screen_off_timeout 30000 and refresh_rate_mode 0 untouched. On AC and fast
charging: battery 69 % to 87 %, 35.6 to 38.8 C (rising through the run).

Delegation: GPU through OpenGL (OpenCL library lacks
`clEnqueueReleaseEGLObjectsKHR`, as on every other Android device); the policy model
fully delegated (5 of 5 nodes XL, 3 of 3 tiny); the gesture model fell back to the CPU.

| model | config | games | trace class (`../crosschip.md`) | ticks, hp |
|---|---|---|---|---|
| XL | CPU F32 | 3 | A (= S6 Lite, POCO, Tab S7+) | 3724, 6 |
| XL | CPU INT8 | 3 | B (= S6 Lite, POCO, Tab S7+) | 3702, 6 |
| XL | GPU F32 | 3 | I | 3724, 6 |
| tiny | CPU F32 | 3 | A (= S6 Lite, Tab S7+) | 4407, 5 |
| tiny | CPU INT8 | 3 | B (= S6 Lite, Tab S7+) | 3956, 7 |
| tiny | GPU F32 | 3 | G | 4407, 5 |

Replays (of the S6 Lite CPU F32 games, XL tag 1c255b7430926177, tiny b707785ae20c8a26):
- CPU F32 and CPU INT8 replays are bit-exact with the other Android CPUs (classes R1,
  R2, T1, T2) for both models.
- Xclipse 540 F32, XL: 21 % of values bit-exact, max 2.6e-3, 0 fire flips (class
  R10); against the Mali-G615 max 6.1e-3. Tiny: max 1.8e-3, 18 fire flips (T7).

Reading: a fourth chip, a second Exynos generation and a new GPU vendor. The CPU
games are byte-identical to the three other Android chips for both models. The
Xclipse is the closest accelerator to the CPU in the study and, with Core ML, the only
one that keeps the tiny model's CPU outcome (4407 ticks, hp 5), though its game trace
differs from tick 0.
