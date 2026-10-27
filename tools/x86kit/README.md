# x86 kit (study 3)

Sources of the self-contained kit on the GitHub branch `x86-kit`: `x86_replay.py`
(replays, comparison, portable-kernel check, re-simulation, seeds), `run.ps1` (Windows),
`build_mac.sh`, `README-x86.md` (the user's steps). The branch also carries the models,
the two reference traces, the device reference outputs and the Mac's own run of the kit
(`reference/mac_m5`, 2026-10-21, TensorFlow 2.20). Rebuild the branch from this folder
plus those data files; the Mac run reproduced the paper's classes byte for byte
(XNNPACK vs R1 230,645 values; built-in == R11; reference kernels == R2).

Finding from the Mac self-test (2026-10-21): the host build of the game reproduces the
recorded traces only with the compiler's default FMA contraction (clang
`-ffp-contract=on`); built with `-ffp-contract=off` the state leaves the recording at
tick 17 (XL) and 13 (tiny). Both ARM compilers in the study contract, so the paper's
"identical across compilers" holds for them; x86 without FMA is the open case.
