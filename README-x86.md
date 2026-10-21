# x86 kit: does an Intel PC compute the same bits as the phones and the Mac?

This folder replays two recorded games of the enemy-AI model on this PC's CPU through
LiteRT (the TensorFlow 2.20 Python package, the same version as on the Mac) with three
kernel sets, and compares every output value with the Android phones, the iPhone and
the Mac. If Visual Studio is installed it also compiles the game and checks that the
recorded games re-simulate byte for byte on x86, then plays 20 seeds in closed loop.
No timing is measured. Everything lives in this folder: a Python venv, the pip
packages, the compiled helpers and the results. Nothing is installed on the PC.

## Steps (about 20 to 40 minutes, mostly unattended)

1. Check the PC is idle: no Lightbound run, no NeonSwarm training, and no
   `C:\dev\GPU_HOLD.txt` naming another project. If anything is busy, stop and say so.
2. Open **Developer PowerShell for VS 2022** (Start menu) if it exists; otherwise a
   normal PowerShell. Then:
   ```
   cd C:\dev
   git clone --branch x86-kit --single-branch https://github.com/jttarigan/npubench-swarm.git npubench-x86
   cd npubench-x86
   powershell -ExecutionPolicy Bypass -File run.ps1
   ```
   The first run downloads TensorFlow (about 400 MB). Replays take 5 to 10 minutes,
   the seeds another 5 to 10. The last lines show where the results are.
3. Read `results\<pc-name>\RESULTS.md` if you like, then push the results back:
   ```
   git add results
   git commit -m "x86 results"
   git push
   ```
   If git asks you to sign in, use the browser sign-in it offers.
4. Delete the folder and say "done":
   ```
   cd C:\dev
   Remove-Item -Recurse -Force npubench-x86
   ```

## If something goes wrong

- "No Python 3.9-3.13 on this PC": the script unpacks the python.org embeddable zip
  into this folder (not an installer; nothing outside the folder changes) and goes on.
- A compile error from `cl`: the replay part still runs and is the main result; the
  re-simulation and seeds are then skipped. Push whatever `results\` holds.
- pip or download errors: run the same `run.ps1` command again; it resumes.
- `git push` rejected: zip `results\` and copy it to the Mac instead.

## What the results mean

`RESULTS.md` lists, for each replay (model x precision x kernel set), whether the
outputs are bit-identical to each reference ("identical: YES") and otherwise how many
of the 370,062 (XL) or 389,232 (tiny) values differ and by how much. References:
R1/T1 the Android phones' CPU (LiteRT 1.4.2), R2/T2 their int8, R11 the iPhone and Mac
built-in kernels of LiteRT 2.20, R12 the Apple int8 class, R19 LiteRT 2.17, and
`mac_m5`, the Mac's own run of this kit. The question is whether the same LiteRT
version gives the same bits on x86 as on ARM.
