# NPUBench determinism harness for iOS (study 3)

A headless iOS app that runs the game's own simulation (`app/src/main/cpp/src/game.cpp`,
`trace.cpp`, compiled unchanged) with the enemy-AI model in LiteRT on the CPU (XNNPACK),
the GPU (Metal delegate) or Core ML (Neural Engine), and writes the same NPBT/NPBR trace
files as the Android app. It mirrors `Engine::doTick` and `Engine::replayStep`
(and `tools/hostsim`, which reproduces the phones' traces byte for byte).

## Setup

1. LiteRT for iOS (not in git): TensorFlowLiteC CocoaPods release `0.0.1-nightly.20250619`
   (reports `TfLiteVersion()` = 2.20.0-dev0), the closest published build to the Android
   app's LiteRT 1.4.2:
   ```
   D=~/.npubench-build/litert-ios/nightly-20250619; mkdir -p $D && cd $D
   curl -sLO https://dl.google.com/tflite-nightly/ios/prod/tensorflow/lite/release/ios/nightly/1673/20250619-035012/TensorFlowLiteC/0.0.1-nightly.20250619/TensorFlowLiteC-0.0.1-nightly.20250619.tar.gz
   # sha256 25f4474c65b80aa33e4ad4b98ea2b60c67b70ce6b64c88e9018a1bfdf7bb5e44
   tar xzf TensorFlowLiteC-*.tar.gz
   ln -sfn $D/TensorFlowLiteC-0.0.1-nightly.20250619/Frameworks <repo>/ios/third_party/litert-ios
   ```
2. Replay input (not in git): `cp runs_study3/s6lite/files/run_20261013_223444_game_cpu_xlarge_f32_trace.bin ios/replay/replay_input.bin`
3. Build (team YOUR_TEAM_ID, automatic signing):
   `xcodebuild -project ios/NPUBenchDet.xcodeproj -scheme NPUBenchDet -configuration Release -destination 'generic/platform=iOS' -derivedDataPath ios/build -allowProvisioningUpdates build`

## Run from the Mac (no taps needed; the device must be unlocked)

```
U=<device UDID>
xcrun devicectl device install app --device $U ios/build/Build/Products/Release-iphoneos/NPUBenchDet.app
xcrun devicectl device process launch --terminate-existing --device $U com.josta.npubench.det -- --autorun=all
xcrun devicectl device copy from --device $U --domain-type appDataContainer \
    --domain-identifier com.josta.npubench.det --source Documents --destination runs_study3/<device>/files
# the app returns to its menu after an autorun and KEEPS RUNNING: always end with
xcrun devicectl device info processes --device $U | grep NPUBenchDet   # note the pid
xcrun devicectl device process terminate --device $U --pid <pid>       # then check again: gone
```
Wait for the end of a run by the log line `=== ALL DONE (tiny)` in the copied
`session_log.txt` (or the file count), not by the process exiting.
`--autorun=` `quick` (one CPU F32 XL game), `xl`, `tiny`, `all` (XL then tiny: CPU F32/INT8,
GPU F32/INT8, NPU F32, 3 games each, plus a replay of the bundled trace for XL), or
`repeat:<cpu|gpu|npu>:<variant>:<f32|int8>:<games>`. `--legacy-delegate-order` restores the
Android app's order (delegate added at interpreter creation, input resized afterwards).

## Known issue found with this harness (2026-10-16)

With the delegate added at creation and the input resized afterwards, the Metal delegate
running the int8 model wrote fp16-precision values into freed host memory on its first
invoke in some launches; the game's enemy array had reused that memory (enemy 5's
velocity and hp were overwritten before tick 0). The harness now resizes first and applies
the delegate with `TfLiteInterpreterModifyGraphWithDelegate`; six separate launches and a
full run then repeated byte for byte. Every game also self-checks its first three ticks
(the same step on a copy of the game) and logs the thread's FPCR.

## Second runtime: LiteRT 2.17.0 (iPhone 12, 2026-10-17)

To separate the runtime build from the OS, the same app was rebuilt against the
TensorFlowLiteC 2.17.0 CocoaPods release (podspec source
`https://dl.google.com/tflite-release/ios/prod/tensorflow/lite/release/ios/release/32/20240729-115310/TensorFlowLiteC/2.17.0/0c10b3543e01f547/TensorFlowLiteC-2.17.0.tar.gz`,
sha256 9667b476015f136e5b332ce040e12822c4ac6d5c58947882ddc809cdff0fb99e), unpacked in
`~/.npubench-build/litert-ios/release-2.17.0/`, by pointing `ios/third_party/litert-ios` at its
`Frameworks` for the build (`-derivedDataPath ios/build217`) and back afterwards. The app logs
`LiteRT 2.17.0`. Results: `runs_study3/iphone12_lite217/README.md`.

## Build provenance (iPhone 12 runs, 2026-10-16)

Xcode 27.1 (27A9275), iPhoneOS 27.1 SDK (Xcode auto-updated at 10:14 WIB that day;
all our iOS builds came after it), Apple clang from that Xcode, deployment target
iOS 17.0, -O2. Device: iPhone 12 (iPhone13,2), iOS 26.6.1. No CMake involved.
