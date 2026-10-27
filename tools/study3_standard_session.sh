#!/bin/zsh
# Study 3 standard session on one device: free runs (3 looped games per config) and
# replays of the S6 Lite CPU F32 trace, for the XL and tiny models, as on the Tab S7+
# (runs_study3/tab/seq_*_20261016.txt). Run it under caffeinate -is.
# usage: study3_standard_session.sh DEV MODEL W H SERIAL "cpu:f32 cpu:int8 gpu:f32" ["xl t"]
# (the last argument picks the models; DRIVE_EXTRA passes --origin/--display to the driver)
# Needs the study APK installed and the app showing its menu (fresh launch defaults:
# CPU, M, F32, TRACE on, LOOP off), and a menu reference shot for MODEL (`refshot`).
# Ends with the app force-stopped; uninstalling is a separate step.
set -e
cd "${0:A:h}/.."
DEV=$1; M=$2; W=$3; H=$4; S=$5; CONFIGS=(${=6}); MODELS=(${=${7:-xl t}})
A=~/Library/Android/sdk/platform-tools/adb
D() { python3 -u tools/study3_drive.py --model $M --screen $W $H ${=DRIVE_EXTRA} "$@"; }
T() { python3 -u tools/study3_device.py $DEV --serial $S "$@"; }
batt() { $A -s $S shell dumpsys battery | grep -E "AC powered|level|temperature" | tr -s ' ' | tr '\n' ' '; echo; }
macload() { echo "mac load $(sysctl -n vm.loadavg | tr -d '{}')"; }
cfg() { echo "=== $1 $(TZ=Asia/Jakarta date +%H:%M:%S) WIB; $(batt) $(macload)"; }
typeset -A TRACE
TRACE[xl]=runs_study3/s6lite/files/run_20261013_223444_game_cpu_xlarge_f32_trace.bin
TRACE[t]=runs_study3/s6lite/files/run_20261014_203009_game_cpu_tiny_f32_trace.bin
echo "=== start $(date -u +%Y%m%d_%H%M%S) UTC / $(TZ=Asia/Jakarta date +%H:%M) WIB; $(batt) $(macload)"
D ismenu | grep -q "True" || { echo "menu not showing: stop"; exit 1; }
D taps loop
for model in $MODELS; do
  T push-replay $TRACE[$model] | tail -1
  D taps $model
  for c in $CONFIGS; do
    D taps ${c%%:*} ${c##*:}; cfg "FREE $model ${c:u}"; D loop3; T pull | tail -1
  done
  cfg "REPLAYS $model"
  for c in $CONFIGS; do
    D taps ${c%%:*} ${c##*:}; echo "--- replay $model ${c:u}"; D replay
  done
  T pull | tail -1
  D taps cpu f32
done
T pull | tail -1
T check | tail -20
$A -s $S shell am force-stop com.npubench.swarm
$A -s $S shell pidof com.npubench.swarm || echo "app not running"
echo "=== all done $(TZ=Asia/Jakarta date +%H:%M:%S) WIB; $(batt) $(macload)"
