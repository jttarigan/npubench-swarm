#!/bin/zsh
# Study 3, Galaxy Z Fold 8 evening session (planned 2026-10-21): two tests, no new build.
#   A  LiteRT 1.0.1 repeatability on a second chip with two core types (Oryon 6 + 2):
#      the repacked APK of 2026-10-17 (md5 5a20931b...), XL and tiny, CPU F32 and CPU
#      INT8 x3 games each plus CPU F32 replays, as on the Tab S6 Lite (study3_standard_session.sh).
#   B  On-device prevention (grid 2^-2 + guard 0.02, portable recompute), the prevention
#      APK (md5 f391f612...): control, then XL and tiny on CPU and GPU x3 (study3_prevent_campaign.sh).
# Fold rules (borrowed phone, 2026-10-21): pull after every config (the called scripts do);
# if wireless debugging drops, stop without retrying; uninstall the app at the end; put
# Stay awake and the screen timeout back as found; the user switches Developer options
# and wireless debugging off afterwards. Never touch the owner's files.
# usage: caffeinate -is tools/study3_fold8_night.sh <serial> [A|B|AB]      (default AB)
set -e
cd "${0:A:h}/.."
S=$1; PART=${2:-AB}
A=~/Library/Android/sdk/platform-tools/adb
PKG=com.npubench.swarm
LITERT101_APK=${LITERT101_APK:-$HOME/.npubench-build/apk-archive/study3-litert101.apk}   # tools/study3_litert101_repack.sh
PREVENT_APK=$HOME/.npubench-build/apk-archive/study3-prevent-f391f612.apk   # archived: the build path is shared with study 2
export DRIVE_EXTRA="--origin 639 0 --display 4630947004648141459"   # letterboxed on the inner screen
GEOM=(SM_F971B 1170 1848)
T() { python3 -u tools/study3_device.py fold8 --serial $S "$@"; }
say() { echo "=== $(TZ=Asia/Jakarta date +%H:%M:%S) WIB: $*"; }
launch() { $A -s $S shell monkey -p $PKG -c android.intent.category.LAUNCHER 1 >/dev/null 2>&1; sleep 6; }
stopapp() { $A -s $S shell am force-stop $PKG; sleep 1; $A -s $S shell pidof $PKG && { echo "app still running"; exit 1; } || true; }

[ -n "$S" ] || { echo "usage: $0 <serial> [A|B|AB]"; exit 1; }
$A -s $S get-state >/dev/null 2>&1 || { echo "device $S not reachable over adb: stop"; exit 1; }
STAY=$($A -s $S shell settings get global stay_on_while_plugged_in | tr -d '\r')
TIMEOUT=$($A -s $S shell settings get system screen_off_timeout | tr -d '\r')
say "start; stay_on_while_plugged_in=$STAY screen_off_timeout=$TIMEOUT; $(T status | tail -2 | tr '\n' ' ')"

if [[ $PART == *A* ]]; then
  [ -f $LITERT101_APK ] && [ "$(unzip -p $LITERT101_APK lib/arm64-v8a/libtensorflowlite_jni.so | shasum -a 256 | cut -d' ' -f1)" = "$(cat $LITERT101_APK.libsha256)" ] \
    || { echo "LiteRT 1.0.1 APK missing or not carrying the 1.0.1 library ($LITERT101_APK): run tools/study3_litert101_repack.sh"; exit 1; }
  say "A: LiteRT 1.0.1 repeatability"
  $A -s $S shell setprop debug.npubench.prevent "''"
  T install $LITERT101_APK | tail -1
  T logcat start | tail -1
  launch
  tools/study3_standard_session.sh fold8 $GEOM $S "cpu:f32 cpu:int8" "xl t"
  T pull | tail -1
  T logcat stop | tail -1
  stopapp
  $A -s $S uninstall $PKG | tail -1
  say "A done"
fi

if [[ $PART == *B* ]]; then
  [ "$(md5 -q $PREVENT_APK | cut -c1-8)" = "f391f612" ] || { echo "prevention APK md5 changed: stop"; exit 1; }
  say "B: on-device prevention"
  tools/study3_prevent_campaign.sh fold8 $GEOM $S "cpu gpu"
  stopapp
  $A -s $S uninstall $PKG | tail -1      # the campaign script reinstalls the study APK; the Fold 8 keeps nothing
  say "B done"
fi

# KEEP_STAY=1: another project takes the phone next and needs Stay awake on (Main, 2026-10-21)
[ -n "$KEEP_STAY" ] || $A -s $S shell settings put global stay_on_while_plugged_in "$STAY" || true
$A -s $S shell settings put system screen_off_timeout "$TIMEOUT" || true
say "end; app installed: $($A -s $S shell pm path $PKG | tr -d '\r' | head -1)"; $A -s $S shell dumpsys battery | grep -E 'level|temperature' | tr -s ' ' | tr '\n' ' '; echo
