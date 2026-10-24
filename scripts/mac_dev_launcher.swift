// macOS dev launcher, built by scripts/mac_dev_launcher.sh. Recent macOS refuses the camera to a command-line
// Python without ever showing the question; an app bundle with usage strings gets the question, and the assistant
// it starts as a child inherits the answer. Runs <checkout>/.venv/bin/python -m biobench_sar, where <checkout> is
// the first argument (`open "SAR Launcher.app" --args /absolute/path/to/checkout`, while it is not running) or the
// SARCheckout key the build wrote.
import AVFoundation
import Foundation

let logURL = Bundle.main.bundleURL.deletingLastPathComponent().appendingPathComponent("launcher.log")
if !FileManager.default.fileExists(atPath: logURL.path) {
    FileManager.default.createFile(atPath: logURL.path, contents: nil)
}
guard let log = FileHandle(forWritingAtPath: logURL.path) else { exit(1) }
log.seekToEndOfFile()
func say(_ line: String) { log.write(Data((line + "\n").utf8)) }
say("---- SAR Launcher \(Date())")

func ask(_ media: AVMediaType, _ name: String, _ pane: String) {
    if AVCaptureDevice.authorizationStatus(for: media) == .notDetermined {
        let answered = DispatchSemaphore(value: 0)
        AVCaptureDevice.requestAccess(for: media) { _ in answered.signal() }
        _ = answered.wait(timeout: .now() + 120)
    }
    let status = AVCaptureDevice.authorizationStatus(for: media)
    say("\(name): \(status == .authorized ? "allowed" : "NOT allowed (status \(status.rawValue))")")
    if status != .authorized {
        say("  allow SAR Launcher in System Settings > Privacy & Security > \(pane), then open it again")
    }
}

ask(.video, "camera", "Camera")
ask(.audio, "microphone", "Microphone")

let args = CommandLine.arguments.dropFirst().filter { !$0.hasPrefix("-psn_") }
guard let checkout = args.first ?? Bundle.main.object(forInfoDictionaryKey: "SARCheckout") as? String else {
    say("no checkout: rebuild with scripts/mac_dev_launcher.sh")
    exit(1)
}
guard checkout.hasPrefix("/") else {
    say("\(checkout) is not an absolute path")
    exit(1)
}
let python = URL(fileURLWithPath: checkout).appendingPathComponent(".venv/bin/python")
guard FileManager.default.isExecutableFile(atPath: python.path) else {
    say("\(python.path) not found: run bash scripts/setup_mac.sh in \(checkout) first")
    exit(1)
}

// Opened from Finder or `open`, the PATH is launchd's; add Homebrew's so tesseract and friends are found.
var env = ProcessInfo.processInfo.environment
env["PATH"] = "/opt/homebrew/bin:/usr/local/bin:" + (env["PATH"] ?? "/usr/bin:/bin:/usr/sbin:/sbin")
env["PYTHONUNBUFFERED"] = "1"

let assistant = Process()
assistant.executableURL = python
assistant.arguments = ["-m", "biobench_sar"]
assistant.currentDirectoryURL = URL(fileURLWithPath: checkout)
assistant.environment = env
assistant.standardOutput = log
assistant.standardError = log

do {
    try assistant.run()
} catch {
    say("could not start the assistant: \(error)")
    exit(1)
}
say("checkout \(checkout), assistant pid \(assistant.processIdentifier)")

// `pkill -f SARLauncher` (or quitting from Activity Monitor) stops the assistant too.
var stoppers: [DispatchSourceSignal] = []
for sig in [SIGTERM, SIGINT, SIGHUP] {
    signal(sig, SIG_IGN)
    let source = DispatchSource.makeSignalSource(signal: sig, queue: .global())
    source.setEventHandler { if assistant.isRunning { assistant.terminate() } }
    source.resume()
    stoppers.append(source)
}
assistant.waitUntilExit()
say("assistant exited with \(assistant.terminationStatus)")
exit(assistant.terminationStatus)
