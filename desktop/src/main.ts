import { invoke } from "@tauri-apps/api/core";
import { listen } from "@tauri-apps/api/event";
import {
  CheckMenuItem,
  Menu,
  MenuItem,
  PredefinedMenuItem,
  Submenu,
} from "@tauri-apps/api/menu";
import { TrayIcon } from "@tauri-apps/api/tray";
import { getCurrentWindow } from "@tauri-apps/api/window";
import { readText, writeText } from "@tauri-apps/plugin-clipboard-manager";
import { WavRecorder } from "./recorder";

const serviceUrl = "http://127.0.0.1:8765";
const recorderSmokeUrl = import.meta.env.VITE_GRANITE_TAURI_RECORDER_SMOKE_URL as
  | string
  | undefined;
const pasteSmokeUrl = import.meta.env.VITE_GRANITE_TAURI_PASTE_SMOKE_URL as string | undefined;
const serviceRecordSmokeUrl = import.meta.env
  .VITE_GRANITE_TAURI_SERVICE_RECORD_SMOKE_URL as string | undefined;
const pasteSmokeText =
  (import.meta.env.VITE_GRANITE_TAURI_PASTE_SMOKE_TEXT as string | undefined) ??
  "Granite Speach paste smoke";
const gnomeShortcutLabel = "Copilot key via GNOME custom shortcut";

type Health = {
  status: string;
  hotkey: string;
  cleanup_enabled: boolean;
  max_recording_seconds: number;
  min_recording_ms: number;
  clipboard_restore_delay_ms: number;
  restore_clipboard_after_paste: boolean;
};

let latestTranscript = "";
let cleanupEnabled = true;
let healthPayload: Health | null = null;
let recording = false;
let trayIcon: TrayIcon | null = null;
let currentStatus = "Loading";
let configuredShortcut = "";
let hotkeyBackend = "";
let recordingStartedAt = 0;
let maxRecordingTimer: number | undefined;
const recentTranscripts: string[] = [];

type HotkeyEventPayload = {
  state: "Pressed" | "Released";
  shortcut: string;
  backend: string;
};

function mustGet<T extends HTMLElement>(selector: string): T {
  const element = document.querySelector<T>(selector);
  if (!element) {
    throw new Error(`Missing element: ${selector}`);
  }
  return element;
}

function setStatus(value: string) {
  currentStatus = value;
  mustGet<HTMLDivElement>("#status").textContent = value;
  updateRecordingControls();
  void refreshTrayMenu();
}

function clearError() {
  mustGet<HTMLParagraphElement>("#error").textContent = "";
}

function showError(context: string, error: unknown) {
  const message = formatError(error);
  mustGet<HTMLParagraphElement>("#error").textContent = `${context}: ${message}`;
  mustGet<HTMLPreElement>("#health").textContent = `${context}: ${message}`;
  console.error(context, error);
  void refreshTrayMenu();
}

function formatError(error: unknown): string {
  if (error instanceof Error) {
    return `${error.name}: ${error.message}`;
  }
  return String(error);
}

function setLatest(value: string) {
  latestTranscript = value;
  mustGet<HTMLTextAreaElement>("#latest").value = value;
  rememberTranscript(value);
  updateRecordingControls();
  void refreshTrayMenu();
}

function rememberTranscript(value: string) {
  const trimmed = value.trim();
  if (!trimmed || recentTranscripts[0] === trimmed) {
    return;
  }
  recentTranscripts.unshift(trimmed);
  recentTranscripts.splice(5);
}

async function refreshHealth(options: { showLoading?: boolean } = {}) {
  const health = mustGet<HTMLPreElement>("#health");
  clearError();
  if (options.showLoading ?? true) {
    setStatus("Loading");
  }
  try {
    const response = await fetch(`${serviceUrl}/health`);
    const payload = await response.json();
    healthPayload = payload;
    health.textContent = JSON.stringify(payload, null, 2);
    cleanupEnabled = Boolean(payload.cleanup_enabled);
    mustGet<HTMLInputElement>("#cleanup").checked = cleanupEnabled;
    const wasRecording = recording;
    recording = payload.status === "recording";
    if (!recording) {
      window.clearTimeout(maxRecordingTimer);
      maxRecordingTimer = undefined;
    } else if (!wasRecording) {
      recordingStartedAt = Date.now();
    }
    configuredShortcut = gnomeShortcutLabel;
    hotkeyBackend = "";
    setStatus(recording ? "Recording" : "Ready");
  } catch (error) {
    showError("Health check failed", error);
    setStatus("Error");
  }
}

async function startService() {
  clearError();
  setStatus("Loading");
  try {
    await invoke("start_service");
    await delay(1000);
    await refreshHealth();
  } catch (error) {
    showError("Start service failed", error);
    setStatus("Error");
  }
}

async function handleHotkeyState(state: "Pressed" | "Released") {
  if (state === "Pressed") {
    await startRecording();
  } else {
    await stopRecording(true);
  }
}

async function startRecording() {
  if (recording) return;
  clearError();
  setStatus("Recording");
  try {
    await fetchJson("/record/start", {});
    recording = true;
    recordingStartedAt = Date.now();
    updateRecordingControls();
    await refreshHealth({ showLoading: false });
    void refreshTrayMenu();
    window.clearTimeout(maxRecordingTimer);
    maxRecordingTimer = window.setTimeout(
      () => stopRecording(true),
      (healthPayload?.max_recording_seconds ?? 60) * 1000,
    );
  } catch (error) {
    recording = false;
    setStatus("Error");
    showError("Recording failed", error);
  }
}

async function stopRecording(pasteAfter: boolean, keepClipboardAfterPaste = false) {
  if (!recording) return;
  window.clearTimeout(maxRecordingTimer);
  maxRecordingTimer = undefined;
  const elapsedMs = Date.now() - recordingStartedAt;
  if (elapsedMs < (healthPayload?.min_recording_ms ?? 250)) {
    try {
      await fetchJson("/record/stop", { discard: true });
      recording = false;
      setStatus("Ready");
      await refreshHealth({ showLoading: false });
    } catch (error) {
      recording = false;
      setStatus("Error");
      showError("Recording stop failed", error);
    }
    return;
  }
  recording = false;
  setStatus("Transcribing");
  try {
    if (cleanupEnabled) {
      setStatus("Cleaning");
    }
    const payload = await fetchJson("/record/stop", { cleanup: cleanupEnabled });
    setLatest(payload.text ?? "");
    if (pasteAfter) {
      await pasteTranscript(latestTranscript, keepClipboardAfterPaste);
    } else {
      await copyLatest();
      setStatus("Ready");
    }
    await refreshHealth({ showLoading: false });
  } catch (error) {
    recording = false;
    setStatus("Error");
    showError("Transcribe failed", error);
  }
}

function updateRecordingControls() {
  const isRecording = recording;
  const hasLatest = Boolean(latestTranscript);
  mustGet<HTMLButtonElement>("#record-start").disabled = isRecording;
  mustGet<HTMLButtonElement>("#record-stop-paste").disabled = !isRecording;
  mustGet<HTMLButtonElement>("#record-stop").disabled = !isRecording;
  mustGet<HTMLButtonElement>("#copy").disabled = !hasLatest;
  mustGet<HTMLButtonElement>("#paste").disabled = !hasLatest;
}

async function copyLatest() {
  await writeText(latestTranscript);
}

async function pasteLatest() {
  await pasteTranscript(latestTranscript, true);
}

async function pasteTranscript(text: string, keepClipboardAfterPaste = false) {
  if (!text) return;
  setStatus("Pasting");
  const previous = await readText();
  await writeText(text);
  try {
    await invoke("simulate_paste");
    if (healthPayload?.restore_clipboard_after_paste && !keepClipboardAfterPaste) {
      await delay(healthPayload.clipboard_restore_delay_ms ?? 500);
      if ((await readText()) === text) {
        await writeText(previous);
      }
    } else if (keepClipboardAfterPaste) {
      await writeText(text);
    }
    setStatus("Ready");
  } catch (error) {
    await writeText(text);
    setStatus("Error");
    showError("Paste failed", error);
    void invoke("show_notification", {
      title: "Granite Speach",
      message: "Paste failed. The transcript is still on the clipboard.",
    }).catch((notificationError) => console.warn("Notification failed", notificationError));
  }
}

async function showStatusWindow() {
  await refreshHealth({ showLoading: false });
  const appWindow = getCurrentWindow();
  await appWindow.show();
  await appWindow.setFocus();
}

async function setupTray() {
  trayIcon = await TrayIcon.new({
    tooltip: "Granite Speach",
    menuOnLeftClick: true,
    action: () => {
      void refreshHealth({ showLoading: false });
    },
  });
  await refreshTrayMenu();
}

async function refreshTrayMenu() {
  if (!trayIcon) return;
  const shortcut = configuredShortcut || healthPayload?.hotkey || "";
  const backend = hotkeyBackend ? ` via ${hotkeyBackend}` : "";
  await trayIcon.setTooltip(
    shortcut
      ? `Granite Speach - ${currentStatus} - ${shortcut}${backend}`
      : `Granite Speach - ${currentStatus}`,
  );
  await trayIcon.setMenu(await buildTrayMenu());
}

async function buildTrayMenu() {
  const recentItems = recentTranscripts.length
    ? recentTranscripts.map((text, index) => ({
        id: `recent_${index}`,
        text: `${index + 1}. ${menuPreview(text)}`,
        action: () => copyTranscript(text),
      }))
    : [{ id: "recent_empty", text: "No recent transcripts", enabled: false }];

  return Menu.new({
    items: [
      await MenuItem.new({
        id: "status",
        text: `Status: ${currentStatus}`,
        enabled: false,
      }),
      await MenuItem.new({
        id: "hotkey",
        text: `Shortcut: ${configuredShortcut || gnomeShortcutLabel}`,
        enabled: false,
      }),
      await PredefinedMenuItem.new({ item: "Separator" }),
      await CheckMenuItem.new({
        id: "cleanup",
        text: "Cleanup",
        checked: cleanupEnabled,
        action: toggleCleanup,
      }),
      await MenuItem.new({
        id: "record",
        text: "Record",
        enabled: !recording,
        action: startRecording,
      }),
      await MenuItem.new({
        id: "stop_paste",
        text: "Stop + Paste",
        enabled: recording,
        action: () => stopRecording(true, true),
      }),
      await MenuItem.new({
        id: "stop",
        text: "Stop",
        enabled: recording,
        action: () => stopRecording(false),
      }),
      await MenuItem.new({
        id: "paste_latest",
        text: "Paste latest transcript",
        enabled: Boolean(latestTranscript),
        action: pasteLatest,
      }),
      await MenuItem.new({
        id: "copy_latest",
        text: "Copy latest transcript",
        enabled: Boolean(latestTranscript),
        action: copyLatest,
      }),
      await Submenu.new({
        id: "recent",
        text: "Recent transcripts",
        items: recentItems,
      }),
      await PredefinedMenuItem.new({ item: "Separator" }),
      await MenuItem.new({
        id: "start_service",
        text: "Start service",
        action: startService,
      }),
      await MenuItem.new({
        id: "show_status",
        text: "Show status window",
        action: showStatusWindow,
      }),
      await MenuItem.new({
        id: "open_keywords",
        text: "Open keyword glossary",
        action: () => invoke("open_config_file", { kind: "keywords" }),
      }),
      await MenuItem.new({
        id: "open_settings",
        text: "Open settings",
        action: () => invoke("open_config_file", { kind: "settings" }),
      }),
      await PredefinedMenuItem.new({ item: "Separator" }),
      await MenuItem.new({ id: "quit", text: "Quit", action: () => invoke("quit") }),
    ],
  });
}

function toggleCleanup() {
  cleanupEnabled = !cleanupEnabled;
  mustGet<HTMLInputElement>("#cleanup").checked = cleanupEnabled;
  void refreshTrayMenu();
}

async function copyTranscript(text: string) {
  await writeText(text);
  setStatus("Ready");
}

function menuPreview(text: string): string {
  const compact = text.replace(/\s+/g, " ").trim();
  return compact.length > 42 ? `${compact.slice(0, 39)}...` : compact;
}

function delay(ms: number) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

async function fetchJson(path: string, payload: Record<string, unknown>) {
  const response = await fetch(`${serviceUrl}${path}`, {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify(payload),
  });
  const body = await response.json();
  if (!response.ok) {
    throw new Error(body.error ?? `${path} failed with HTTP ${response.status}`);
  }
  return body;
}

async function maybeRunTauriRecorderSmoke() {
  if (!recorderSmokeUrl) {
    return;
  }
  const appWindow = getCurrentWindow();
  const result: Record<string, unknown> = {
    ok: false,
    userAgent: navigator.userAgent,
  };
  try {
    await appWindow.show();
    await appWindow.setFocus();
    const smokeRecorder = new WavRecorder();
    await withTimeout(smokeRecorder.start(), 10_000, "timed out starting WebAudio recorder");
    await delay(900);
    const wavBase64 = await withTimeout(
      smokeRecorder.stop(),
      10_000,
      "timed out stopping WebAudio recorder",
    );
    result.ok = true;
    result.wavBase64 = wavBase64;
  } catch (error) {
    result.error = formatError(error);
  }
  try {
    await fetch(recorderSmokeUrl, {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify(result),
    });
  } catch (error) {
    showError("Tauri recorder smoke callback failed", error);
  }
}

async function maybeRunTauriPasteSmoke() {
  if (!pasteSmokeUrl) {
    return;
  }
  const appWindow = getCurrentWindow();
  const result: Record<string, unknown> = {
    ok: false,
    userAgent: navigator.userAgent,
  };
  const previousClipboard = `granite-previous-${Date.now()}`;
  try {
    await appWindow.show();
    await appWindow.setFocus();
    await refreshHealth();
    await writeText(previousClipboard);
    const target = mustGet<HTMLTextAreaElement>("#latest");
    target.value = "";
    target.focus();
    target.setSelectionRange(0, 0);
    await pasteTranscript(pasteSmokeText);
    await delay(200);
    const insertedText = target.value;
    const clipboardAfter = await readText();
    const errorText = mustGet<HTMLParagraphElement>("#error").textContent ?? "";
    result.insertedText = insertedText;
    result.clipboardAfter = clipboardAfter;
    result.previousClipboard = previousClipboard;
    result.errorText = errorText;
    result.ok =
      insertedText === pasteSmokeText &&
      !errorText &&
      (!healthPayload?.restore_clipboard_after_paste || clipboardAfter === previousClipboard);
  } catch (error) {
    result.error = formatError(error);
  }
  try {
    await fetch(pasteSmokeUrl, {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify(result),
    });
  } catch (error) {
    showError("Tauri paste smoke callback failed", error);
  }
}

async function maybeRunTauriServiceRecordSmoke() {
  if (!serviceRecordSmokeUrl) {
    return;
  }
  const appWindow = getCurrentWindow();
  const result: Record<string, unknown> = {
    ok: false,
    userAgent: navigator.userAgent,
  };
  try {
    await appWindow.show();
    await appWindow.setFocus();
    await refreshHealth();
    await startRecording();
    await delay(100);
    const statusWhileRecording = mustGet<HTMLDivElement>("#status").textContent ?? "";
    const stopDisabledWhileRecording = mustGet<HTMLButtonElement>("#record-stop").disabled;
    await stopRecording(false);
    await delay(100);
    const finalStatus = mustGet<HTMLDivElement>("#status").textContent ?? "";
    const errorText = mustGet<HTMLParagraphElement>("#error").textContent ?? "";
    result.statusWhileRecording = statusWhileRecording;
    result.stopEnabledWhileRecording = !stopDisabledWhileRecording;
    result.finalStatus = finalStatus;
    result.errorText = errorText;
    result.ok = statusWhileRecording === "Recording" && finalStatus === "Ready" && !errorText;
  } catch (error) {
    result.error = formatError(error);
  }
  try {
    await fetch(serviceRecordSmokeUrl, {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify(result),
    });
  } catch (error) {
    showError("Tauri service record smoke callback failed", error);
  }
}

function withTimeout<T>(promise: Promise<T>, ms: number, message: string): Promise<T> {
  return new Promise((resolve, reject) => {
    const timer = window.setTimeout(() => reject(new Error(message)), ms);
    promise.then(
      (value) => {
        window.clearTimeout(timer);
        resolve(value);
      },
      (error) => {
        window.clearTimeout(timer);
        reject(error);
      },
    );
  });
}

window.addEventListener("DOMContentLoaded", () => {
  const appWindow = getCurrentWindow();
  void appWindow.setSkipTaskbar(true);
  void appWindow.onCloseRequested((event) => {
    event.preventDefault();
    void appWindow.hide();
  });
  mustGet<HTMLParagraphElement>("#service-url").textContent = serviceUrl;
  mustGet<HTMLButtonElement>("#refresh").addEventListener("click", () => refreshHealth());
  mustGet<HTMLButtonElement>("#start-service").addEventListener("click", startService);
  mustGet<HTMLButtonElement>("#record-start").addEventListener("click", startRecording);
  mustGet<HTMLButtonElement>("#record-stop-paste").addEventListener("click", () =>
    stopRecording(true, true),
  );
  mustGet<HTMLButtonElement>("#record-stop").addEventListener("click", () => stopRecording(false));
  mustGet<HTMLButtonElement>("#copy").addEventListener("click", copyLatest);
  mustGet<HTMLButtonElement>("#paste").addEventListener("click", pasteLatest);
  mustGet<HTMLInputElement>("#cleanup").addEventListener("change", (event) => {
    cleanupEnabled = (event.target as HTMLInputElement).checked;
    void refreshTrayMenu();
  });
  updateRecordingControls();
  setupTray();
  void listen<HotkeyEventPayload>("granite-hotkey", (event) =>
    handleHotkeyEvent(event.payload),
  ).catch((error) => showError("Hotkey listener failed", error));
  refreshHealth();
  void maybeRunTauriRecorderSmoke();
  void maybeRunTauriPasteSmoke();
  void maybeRunTauriServiceRecordSmoke();
});

function handleHotkeyEvent(payload: HotkeyEventPayload) {
  configuredShortcut = payload.shortcut;
  hotkeyBackend = payload.backend;
  void handleHotkeyState(payload.state);
}

window.addEventListener("error", (event) => {
  showError("Unhandled frontend error", event.error ?? event.message);
  setStatus("Error");
});

window.addEventListener("unhandledrejection", (event) => {
  showError("Unhandled frontend rejection", event.reason);
  setStatus("Error");
});
