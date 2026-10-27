import AVFoundation
import ImageIO
import UniformTypeIdentifiers
let args = CommandLine.arguments
let url = URL(fileURLWithPath: args[1]); let outdir = args[2]
let times = args[3...].compactMap { Double($0) }
let asset = AVURLAsset(url: url)
let gen = AVAssetImageGenerator(asset: asset)
gen.appliesPreferredTrackTransform = true
gen.requestedTimeToleranceBefore = .zero; gen.requestedTimeToleranceAfter = .zero
let dur = CMTimeGetSeconds(asset.duration)
print("duration \(dur) s")
for t in times {
    let cm = CMTime(seconds: t, preferredTimescale: 600)
    do {
        var actual = CMTime.zero
        let img = try gen.copyCGImage(at: cm, actualTime: &actual)
        let path = "\(outdir)/frame_\(String(format: "%05.1f", t))s.png"
        let dest = CGImageDestinationCreateWithURL(URL(fileURLWithPath: path) as CFURL, UTType.png.identifier as CFString, 1, nil)!
        CGImageDestinationAddImage(dest, img, nil); CGImageDestinationFinalize(dest)
        print("wrote \(path) (\(img.width)x\(img.height), actual \(CMTimeGetSeconds(actual)) s)")
    } catch { print("t=\(t): \(error)") }
}
