import ApplicationServices
import AppKit
import Carbon
import Foundation

let logPath = "@@LOG_PATH@@"
let wrapperPath = "@@WRAPPER_PATH@@"
let statePath = "@@STATE_PATH@@"
let spaceKeyCode: Int64 = 49

func log(_ message: String) {
    let formatter = ISO8601DateFormatter()
    let line = "\(formatter.string(from: Date())) \(message)\n"
    guard let data = line.data(using: .utf8) else { return }

    if FileManager.default.fileExists(atPath: logPath),
       let handle = try? FileHandle(forWritingTo: URL(fileURLWithPath: logPath)) {
        defer { try? handle.close() }
        _ = try? handle.seekToEnd()
        _ = try? handle.write(contentsOf: data)
    } else {
        try? data.write(to: URL(fileURLWithPath: logPath))
    }
}

func writeStateFile(_ state: String, _ detail: String) {
    let formatter = ISO8601DateFormatter()
    let line = "\(formatter.string(from: Date()))\t\(state)\t\(detail)\n"
    try? line.write(to: URL(fileURLWithPath: statePath), atomically: true, encoding: .utf8)
}

func postCommandV() {
    let source = CGEventSource(stateID: .hidSystemState)
    let commandKeyCode: CGKeyCode = 55
    let vKeyCode: CGKeyCode = 9

    let commandDown = CGEvent(keyboardEventSource: source, virtualKey: commandKeyCode, keyDown: true)
    commandDown?.flags = .maskCommand
    commandDown?.post(tap: .cghidEventTap)
    usleep(20_000)

    let vDown = CGEvent(keyboardEventSource: source, virtualKey: vKeyCode, keyDown: true)
    vDown?.flags = .maskCommand
    vDown?.post(tap: .cghidEventTap)
    usleep(20_000)

    let vUp = CGEvent(keyboardEventSource: source, virtualKey: vKeyCode, keyDown: false)
    vUp?.flags = .maskCommand
    vUp?.post(tap: .cghidEventTap)
    usleep(20_000)

    let commandUp = CGEvent(keyboardEventSource: source, virtualKey: commandKeyCode, keyDown: false)
    commandUp?.post(tap: .cghidEventTap)
}

class HotkeyStatus: NSObject {
    let statusItem: NSStatusItem
    let menu = NSMenu()
    let statusMenuItem: NSMenuItem
    var lastStateLine = ""
    var readyResetTimer: Timer?

    override init() {
        statusItem = NSStatusBar.system.statusItem(withLength: NSStatusItem.variableLength)
        statusMenuItem = NSMenuItem(title: "TransClip: Ready", action: nil, keyEquivalent: "")
        super.init()

        statusMenuItem.isEnabled = false
        menu.addItem(statusMenuItem)
        menu.addItem(NSMenuItem.separator())

        let openToggleLog = NSMenuItem(
            title: "Open toggle log",
            action: #selector(openToggleLog(_:)),
            keyEquivalent: ""
        )
        openToggleLog.target = self
        menu.addItem(openToggleLog)

        let openHotkeyLog = NSMenuItem(
            title: "Open hotkey log",
            action: #selector(openHotkeyLog(_:)),
            keyEquivalent: ""
        )
        openHotkeyLog.target = self
        menu.addItem(openHotkeyLog)

        menu.addItem(NSMenuItem.separator())
        let quit = NSMenuItem(title: "Quit TransClip Hotkey", action: #selector(quit(_:)), keyEquivalent: "q")
        quit.target = self
        menu.addItem(quit)

        statusItem.menu = menu
        setStatus("ready", "Ready")
    }

    func setStatus(_ state: String, _ detail: String) {
        if Thread.isMainThread {
            applyStatus(state, detail)
        } else {
            DispatchQueue.main.async {
                self.applyStatus(state, detail)
            }
        }
    }

    private func applyStatus(_ state: String, _ detail: String) {
        readyResetTimer?.invalidate()
        readyResetTimer = nil

        let title: String
        let fallback: String

        switch state {
        case "shortcut":
            title = "TC..."
            fallback = "Shortcut received"
        case "busy":
            title = "TC..."
            fallback = "Already working"
        case "recovering":
            title = "TC..."
            fallback = "Recovering"
        case "listening":
            title = "REC"
            fallback = "Recording"
        case "transcribing":
            title = "TXT..."
            fallback = "Transcribing"
        case "pasting":
            title = "PST..."
            fallback = "Pasting transcript"
        case "paste_requested":
            title = "PST..."
            fallback = "Paste transcript"
        case "finished":
            title = "OK"
            fallback = "Finished"
        case "ready":
            title = "TC"
            fallback = "Ready"
        case "error":
            title = "TC!"
            fallback = "Error"
        default:
            title = "TC"
            fallback = "Ready"
        }

        let message = detail.isEmpty ? fallback : detail
        statusItem.button?.attributedTitle = NSAttributedString(
            string: title,
            attributes: [
                .foregroundColor: color(for: state),
                .font: NSFont.monospacedSystemFont(ofSize: NSFont.systemFontSize, weight: .semibold),
            ]
        )
        statusItem.button?.toolTip = "TransClip: \(message)"
        statusMenuItem.title = "TransClip: \(message)"

        if state == "finished" {
            readyResetTimer = Timer.scheduledTimer(withTimeInterval: 2.5, repeats: false) { [weak self] _ in
                self?.setStatus("ready", "Ready")
            }
        }
    }

    private func color(for state: String) -> NSColor {
        switch state {
        case "shortcut", "busy", "recovering":
            return .systemYellow
        case "listening":
            return .systemOrange
        case "transcribing":
            return .systemPurple
        case "pasting", "paste_requested":
            return .systemTeal
        case "finished":
            return .systemGreen
        case "ready":
            return .labelColor
        case "error":
            return .systemRed
        default:
            return .labelColor
        }
    }

    @objc func pollState(_ timer: Timer) {
        guard let line = try? String(contentsOfFile: statePath, encoding: .utf8)
            .trimmingCharacters(in: .whitespacesAndNewlines),
              !line.isEmpty,
              line != lastStateLine else {
            return
        }

        lastStateLine = line
        let parts = line.components(separatedBy: "\t")
        let state = parts.count > 1 ? parts[1] : "ready"
        let detail = parts.count > 2 ? parts[2] : state
        if state == "paste_requested" {
            performPasteRequest(detail)
            return
        }
        setStatus(state, detail)
    }

    func performPasteRequest(_ detail: String) {
        setStatus("pasting", detail.isEmpty ? "Pasting transcript" : detail)
        log("paste requested by wrapper")
        DispatchQueue.main.asyncAfter(deadline: .now() + 0.15) {
            postCommandV()
            log("posted Command+V")
            writeStateFile("finished", "Pasted")
            self.setStatus("finished", "Pasted")
        }
    }

    @objc func openToggleLog(_ sender: Any?) {
        let toggleLogPath = logPath.replacingOccurrences(
            of: "hotkey.log",
            with: "toggle-record.log"
        )
        NSWorkspace.shared.open(URL(fileURLWithPath: toggleLogPath))
    }

    @objc func openHotkeyLog(_ sender: Any?) {
        NSWorkspace.shared.open(URL(fileURLWithPath: logPath))
    }

    @objc func quit(_ sender: Any?) {
        NSApplication.shared.terminate(nil)
    }
}

let app = NSApplication.shared
app.setActivationPolicy(.accessory)
let hotkeyStatus = HotkeyStatus()
Timer.scheduledTimer(
    timeInterval: 0.25,
    target: hotkeyStatus,
    selector: #selector(HotkeyStatus.pollState(_:)),
    userInfo: nil,
    repeats: true
)

func runWrapper() {
    hotkeyStatus.setStatus("shortcut", "Shortcut received")
    let process = Process()
    process.executableURL = URL(fileURLWithPath: wrapperPath)
    do {
        try process.run()
        log("launched wrapper pid=\(process.processIdentifier)")
    } catch {
        log("failed to launch wrapper: \(error)")
    }
}

let promptKey = kAXTrustedCheckOptionPrompt.takeUnretainedValue() as String
let trusted = AXIsProcessTrustedWithOptions([promptKey: true] as CFDictionary)
log("event tap starting axTrusted=\(trusted)")
if !trusted {
    hotkeyStatus.setStatus("error", "Accessibility required")
}

var activeEventTap: CFMachPort?
var shortcutIsActive = false

func reenableEventTap() {
    guard let tap = activeEventTap else {
        log("event tap re-enable skipped; no active tap")
        return
    }

    shortcutIsActive = false
    CGEvent.tapEnable(tap: tap, enable: true)
    log("event tap re-enabled")
}

let callback: CGEventTapCallBack = { _, type, event, _ in
    if type == .tapDisabledByTimeout || type == .tapDisabledByUserInput {
        log("event tap disabled type=\(type.rawValue)")
        reenableEventTap()
        return Unmanaged.passUnretained(event)
    }

    guard type == .keyDown || type == .keyUp else {
        return Unmanaged.passUnretained(event)
    }

    let keyCode = event.getIntegerValueField(.keyboardEventKeycode)

    if type == .keyUp && keyCode == spaceKeyCode && shortcutIsActive {
        shortcutIsActive = false
        return nil
    }

    guard type == .keyDown else {
        return Unmanaged.passUnretained(event)
    }

    let isAutoRepeat = event.getIntegerValueField(.keyboardEventAutorepeat) != 0

    if keyCode == spaceKeyCode && shortcutIsActive {
        return nil
    }

    let flags = event.flags
    let hasOption = flags.contains(.maskAlternate)
    let hasCommand = flags.contains(.maskCommand)
    let hasControl = flags.contains(.maskControl)

    if keyCode == spaceKeyCode && hasOption && !hasCommand && !hasControl {
        if isAutoRepeat {
            return nil
        }
        shortcutIsActive = true
        log("Option+Space detected")
        runWrapper()
        return nil
    }

    return Unmanaged.passUnretained(event)
}

let mask = CGEventMask(1 << CGEventType.keyDown.rawValue) |
    CGEventMask(1 << CGEventType.keyUp.rawValue)
guard let eventTap = CGEvent.tapCreate(
    tap: .cgSessionEventTap,
    place: .headInsertEventTap,
    options: .defaultTap,
    eventsOfInterest: mask,
    callback: callback,
    userInfo: nil
) else {
    log("failed to create event tap; Accessibility/Input Monitoring is required")
    hotkeyStatus.setStatus("error", "Accessibility required")
    app.run()
    exit(0)
}

activeEventTap = eventTap
let runLoopSource = CFMachPortCreateRunLoopSource(kCFAllocatorDefault, eventTap, 0)
CFRunLoopAddSource(CFRunLoopGetCurrent(), runLoopSource, .commonModes)
CGEvent.tapEnable(tap: eventTap, enable: true)
log("event tap listening for Option+Space")
if trusted {
    hotkeyStatus.setStatus("ready", "Ready")
}
app.run()
