#!/bin/zsh
# Study 3 experiment C: on-device prevention (grid 2^-2 + guard 0.02, portable recompute).
# usage: prevent_campaign.sh DEV MODEL W H SERIAL "cpu gpu [npu]" [bitcheck]
# Starts from the app freshly launched by this script (menu defaults: CPU, M, F32, TRACE on).
set -e
cd "${0:A:h}/.."
DEV=$1; M=$2; W=$3; H=$4; S=$5; BACKENDS=(${=6}); BIT=$7
APK=${PREVENT_APK:-$HOME/.npubench-build/apk-archive/study3-prevent-f391f612.apk}   # archived: the build path is shared with study 2
STUDY_APK=~/.npubench-build/apk-archive/study3-cd4d9034.apk
SPEC="grid:2+fallback:0.02"
A=~/Library/Android/sdk/platform-tools/adb
D() { python3 -u tools/study3_drive.py --model $M --screen $W $H ${=DRIVE_EXTRA} "$@"; }
T() { python3 -u tools/study3_device.py $DEV --serial $S "$@"; }
batt() { $A -s $S shell dumpsys battery | grep -E "AC powered|level|temperature" | tr -s ' ' | tr '\n' ' '; echo; }
cfg() { echo "=== $1 $(TZ=Asia/Jakarta date +%H:%M:%S) WIB; prevent='$($A -s $S shell getprop debug.npubench.prevent)'; $(batt)"; }
prop() { $A -s $S shell setprop debug.npubench.prevent "${1:-''}"; }
echo "=== start $(date -u +%Y%m%d_%H%M%S) UTC / $(TZ=Asia/Jakarta date +%H:%M) WIB; apk $(md5 -q $APK)"
T status | tail -4
T pull | tail -1
T install $APK | tail -1
T logcat start | tail -1
$A -s $S shell monkey -p com.npubench.swarm -c android.intent.category.LAUNCHER 1 >/dev/null 2>&1; sleep 5
T push-replay runs_study3/s6lite/files/run_20261013_223444_game_cpu_xlarge_f32_trace.bin | tail -1
D taps xl
D ismenu
if [ "$BIT" = "bitcheck" ]; then
  prop portable; cfg "BITCHECK XL replay, every row portable"; D replay
fi
D taps loop
prop; cfg "CONTROL XL CPU F32, prevention off"; D loop3; T pull | tail -1
prop $SPEC
for model in xl t; do
  D taps $model
  for b in $BACKENDS; do D taps $b f32; cfg "PREVENT $model ${b:u} F32"; D loop3; T pull | tail -1; done
  D taps cpu
done
prop
T pull | tail -1
T logcat stop | tail -2
T install $STUDY_APK | tail -1
$A -s $S shell am force-stop com.npubench.swarm
$A -s $S shell pidof com.npubench.swarm || echo "app not running"
echo "=== all done $(TZ=Asia/Jakarta date +%H:%M:%S) WIB; $(batt)"
