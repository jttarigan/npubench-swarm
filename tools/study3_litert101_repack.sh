#!/bin/zsh
# Rebuild the LiteRT 1.0.1 test APK of 2026-10-17 (runs_study3/s6lite_litert101/README.md):
# the study APK (md5 cd4d9034...) with only its two LiteRT native libraries replaced by
# those of com.google.ai.edge.litert:litert:1.0.1 and litert-gpu:1.0.1 (the Maven
# relocation of TensorFlow Lite 2.17.0; the library reports 2.18.0). The .so entries are
# stored uncompressed, the APK zipaligned (-P 16) and signed with the debug key. The
# signature time makes the md5 differ from the original (5a20931b...), so the sidecar
# .libsha256 (sha256 of the jni .so) is what study3_fold8_night.sh verifies.
#   tools/study3_litert101_repack.sh   -> ~/.npubench-build/apk-archive/study3-litert101.apk
set -e
OUT=~/.npubench-build/apk-archive/study3-litert101.apk
STUDY=~/.npubench-build/apk-archive/study3-cd4d9034.apk
BT=$(ls -d ~/Library/Android/sdk/build-tools/*/ | sort -V | tail -1)
W=$(mktemp -d /tmp/litert101.XXXX)
cd $W
M=https://dl.google.com/dl/android/maven2/com/google/ai/edge/litert
curl -sSL -o litert.aar $M/litert/1.0.1/litert-1.0.1.aar
curl -sSL -o litert-gpu.aar $M/litert-gpu/1.0.1/litert-gpu-1.0.1.aar
[ "$(shasum -a 256 litert.aar | cut -c1-16)" = "467679820f836fe7" ] || { echo "litert-1.0.1.aar sha256 differs from the 2026-10-17 note"; exit 1; }
[ "$(shasum -a 256 litert-gpu.aar | cut -c1-16)" = "b757c4ddec610fc5" ] || { echo "litert-gpu-1.0.1.aar sha256 differs from the 2026-10-17 note"; exit 1; }
unzip -q -o litert.aar jni/arm64-v8a/libtensorflowlite_jni.so
unzip -q -o litert-gpu.aar jni/arm64-v8a/libtensorflowlite_gpu_jni.so
[ "$(md5 -q $STUDY | cut -c1-8)" = "cd4d9034" ] || { echo "study APK md5 changed"; exit 1; }
python3 - "$STUDY" "$W" <<'EOF'
import sys, zipfile
src, w = sys.argv[1], sys.argv[2]
rep = {"lib/arm64-v8a/libtensorflowlite_jni.so": f"{w}/jni/arm64-v8a/libtensorflowlite_jni.so",
       "lib/arm64-v8a/libtensorflowlite_gpu_jni.so": f"{w}/jni/arm64-v8a/libtensorflowlite_gpu_jni.so"}
with zipfile.ZipFile(src) as zin, zipfile.ZipFile(f"{w}/unaligned.apk", "w") as zout:
    n = 0
    for item in zin.infolist():
        if item.filename.startswith("META-INF/") and item.filename.split(".")[-1] in ("RSA", "SF", "MF", "EC", "DSA"):
            continue
        if item.filename in rep:
            zi = zipfile.ZipInfo(item.filename, item.date_time)
            zi.compress_type = zipfile.ZIP_STORED
            zout.writestr(zi, open(rep[item.filename], "rb").read())
            n += 1
        else:
            zi = zipfile.ZipInfo(item.filename, item.date_time)
            zi.compress_type = item.compress_type
            zout.writestr(zi, zin.read(item.filename))
    assert n == 2, n
print("repacked: 2 libraries replaced")
EOF
${BT}zipalign -P 16 -f 4 unaligned.apk aligned.apk
${BT}apksigner sign --ks ~/.android/debug.keystore --ks-pass pass:android --key-pass pass:android --out $OUT aligned.apk
${BT}apksigner verify $OUT
shasum -a 256 jni/arm64-v8a/libtensorflowlite_jni.so | cut -d' ' -f1 > $OUT.libsha256
echo "wrote $OUT (md5 $(md5 -q $OUT | cut -c1-8); jni lib sha256 $(cut -c1-16 $OUT.libsha256)...)"
unzip -p $OUT lib/arm64-v8a/libtensorflowlite_jni.so | shasum -a 256 | cut -d' ' -f1 | diff - $OUT.libsha256 && echo "APK carries the 1.0.1 library"
rm -rf $W
