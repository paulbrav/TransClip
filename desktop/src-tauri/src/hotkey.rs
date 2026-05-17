use std::sync::{Mutex, OnceLock};

use tauri::Emitter;

const BACKEND_EVDEV: &str = "linux-evdev";
const BACKEND_TAURI: &str = "tauri-global-shortcut";
const SUPPORTED_MODIFIERS: &str = "Ctrl/Control, Alt/Option, Shift";
const SUPPORTED_KEYS: &str = "Space, Home, Insert, Pause, ScrollLock, F1-F12";

static ACTIVE_HOTKEY: OnceLock<Mutex<Option<ActiveHotkey>>> = OnceLock::new();

#[derive(Clone, serde::Serialize)]
pub struct HotkeyStatus {
    shortcut: String,
    backend: String,
    registered: bool,
    message: String,
}

#[derive(Clone, serde::Serialize)]
struct HotkeyEventPayload {
    state: &'static str,
    shortcut: String,
    backend: String,
}

#[derive(Debug)]
struct ParsedHotkey {
    shortcut: String,
    ctrl: bool,
    alt: bool,
    shift: bool,
    key: HotkeyKey,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum HotkeyKey {
    Space,
    Home,
    Insert,
    Pause,
    ScrollLock,
    F(u8),
}

enum HotkeyBackend {
    Evdev,
    TauriGlobalShortcut,
    Unsupported,
}

enum ActiveHotkey {
    #[cfg(target_os = "linux")]
    Evdev(EvdevHotkeyListener),
    Tauri(String),
}

#[tauri::command]
pub fn configure_hotkey(app: tauri::AppHandle, hotkey: String) -> Result<HotkeyStatus, String> {
    let parsed = parse_hotkey(&hotkey)?;
    let backend = selected_backend();
    stop_hotkey(&app);

    match backend {
        HotkeyBackend::Evdev => configure_evdev_hotkey(app, parsed),
        HotkeyBackend::TauriGlobalShortcut => configure_tauri_hotkey(app, parsed),
        HotkeyBackend::Unsupported => Ok(HotkeyStatus {
            shortcut: parsed.shortcut,
            backend: "unsupported".into(),
            registered: false,
            message: "Global hotkeys are only implemented for Linux and macOS".into(),
        }),
    }
}

pub fn stop_hotkey(app: &tauri::AppHandle) {
    let Some(mutex) = ACTIVE_HOTKEY.get() else {
        return;
    };
    let Ok(mut slot) = mutex.lock() else {
        return;
    };
    if let Some(active) = slot.take() {
        match active {
            #[cfg(target_os = "linux")]
            ActiveHotkey::Evdev(listener) => listener.stop(),
            ActiveHotkey::Tauri(shortcut) => {
                use tauri_plugin_global_shortcut::GlobalShortcutExt;
                let _ = app.global_shortcut().unregister(shortcut.as_str());
            }
        }
    }
}

fn configure_tauri_hotkey(
    app: tauri::AppHandle,
    parsed: ParsedHotkey,
) -> Result<HotkeyStatus, String> {
    use tauri_plugin_global_shortcut::{GlobalShortcutExt, ShortcutState};

    let shortcut = parsed.shortcut;
    let emit_shortcut = shortcut.clone();
    app.global_shortcut()
        .on_shortcut(shortcut.as_str(), move |app, _shortcut, event| {
            let state = match event.state {
                ShortcutState::Pressed => "Pressed",
                ShortcutState::Released => "Released",
            };
            let _ = app.emit(
                "granite-hotkey",
                HotkeyEventPayload {
                    state,
                    shortcut: emit_shortcut.clone(),
                    backend: BACKEND_TAURI.into(),
                },
            );
        })
        .map_err(|error| {
            format!("Could not register {shortcut} with Tauri global-shortcut: {error}")
        })?;

    store_active(ActiveHotkey::Tauri(shortcut.clone()))?;
    Ok(HotkeyStatus {
        shortcut,
        backend: BACKEND_TAURI.into(),
        registered: true,
        message: "registered".into(),
    })
}

#[cfg(target_os = "linux")]
fn configure_evdev_hotkey(
    app: tauri::AppHandle,
    parsed: ParsedHotkey,
) -> Result<HotkeyStatus, String> {
    let shortcut = parsed.shortcut.clone();
    let hotkey = EvdevHotkey::try_from(parsed)?;
    let devices = match open_keyboard_devices(hotkey.key) {
        Ok(devices) => devices,
        Err(message) => {
            return Ok(HotkeyStatus {
                shortcut,
                backend: BACKEND_EVDEV.into(),
                registered: false,
                message,
            });
        }
    };
    let listener = match EvdevHotkeyListener::start(app, shortcut.clone(), hotkey, devices) {
        Ok(listener) => listener,
        Err(message) => {
            return Ok(HotkeyStatus {
                shortcut,
                backend: BACKEND_EVDEV.into(),
                registered: false,
                message,
            });
        }
    };
    store_active(ActiveHotkey::Evdev(listener))?;
    Ok(HotkeyStatus {
        shortcut,
        backend: BACKEND_EVDEV.into(),
        registered: true,
        message: "registered".into(),
    })
}

#[cfg(not(target_os = "linux"))]
fn configure_evdev_hotkey(
    _app: tauri::AppHandle,
    parsed: ParsedHotkey,
) -> Result<HotkeyStatus, String> {
    Ok(HotkeyStatus {
        shortcut: parsed.shortcut,
        backend: BACKEND_EVDEV.into(),
        registered: false,
        message: "Linux evdev hotkeys are only available on Linux".into(),
    })
}

fn store_active(active: ActiveHotkey) -> Result<(), String> {
    let mutex = ACTIVE_HOTKEY.get_or_init(|| Mutex::new(None));
    let mut slot = mutex
        .lock()
        .map_err(|_| "Hotkey registration lock is poisoned".to_string())?;
    *slot = Some(active);
    Ok(())
}

fn selected_backend() -> HotkeyBackend {
    #[cfg(target_os = "linux")]
    {
        return select_backend_for(
            std::env::var("XDG_SESSION_TYPE").ok().as_deref(),
            std::env::var("WAYLAND_DISPLAY").ok().as_deref(),
            "linux",
        );
    }

    #[cfg(target_os = "macos")]
    {
        return HotkeyBackend::TauriGlobalShortcut;
    }

    #[cfg(not(any(target_os = "linux", target_os = "macos")))]
    {
        HotkeyBackend::Unsupported
    }
}

fn select_backend_for(
    session_type: Option<&str>,
    wayland_display: Option<&str>,
    target_os: &str,
) -> HotkeyBackend {
    if target_os == "linux" {
        let is_wayland = session_type
            .map(|value| value.eq_ignore_ascii_case("wayland"))
            .unwrap_or(false)
            || (session_type.is_none() && wayland_display.is_some());
        if is_wayland {
            HotkeyBackend::Evdev
        } else {
            HotkeyBackend::TauriGlobalShortcut
        }
    } else if target_os == "macos" {
        HotkeyBackend::TauriGlobalShortcut
    } else {
        HotkeyBackend::Unsupported
    }
}

fn parse_hotkey(value: &str) -> Result<ParsedHotkey, String> {
    let parts = value
        .split('+')
        .map(str::trim)
        .filter(|part| !part.is_empty())
        .collect::<Vec<_>>();
    let Some(key_part) = parts.last() else {
        return Err("Hotkey is empty".into());
    };

    let key = parse_key(key_part)?;
    let mut ctrl = false;
    let mut alt = false;
    let mut shift = false;
    for modifier in &parts[..parts.len() - 1] {
        match modifier.to_ascii_lowercase().as_str() {
            "ctrl" | "control" | "cmdorcontrol" | "commandorcontrol" => ctrl = true,
            "alt" | "option" => alt = true,
            "shift" => shift = true,
            other => {
                return Err(format!(
                    "Unsupported hotkey modifier '{other}'. Supported modifiers: {SUPPORTED_MODIFIERS}; supported keys: {SUPPORTED_KEYS}"
                ));
            }
        }
    }

    let mut shortcut = Vec::new();
    if ctrl {
        shortcut.push("Ctrl".to_string());
    }
    if alt {
        shortcut.push("Alt".to_string());
    }
    if shift {
        shortcut.push("Shift".to_string());
    }
    shortcut.push(key.shortcut_name());

    Ok(ParsedHotkey {
        shortcut: shortcut.join("+"),
        ctrl,
        alt,
        shift,
        key,
    })
}

fn parse_key(value: &str) -> Result<HotkeyKey, String> {
    let lower = value.to_ascii_lowercase();
    let key = match lower.as_str() {
        "space" => HotkeyKey::Space,
        "home" => HotkeyKey::Home,
        "insert" => HotkeyKey::Insert,
        "pause" => HotkeyKey::Pause,
        "scrolllock" | "scroll_lock" | "scroll-lock" => HotkeyKey::ScrollLock,
        _ if lower.starts_with('f') => {
            let number = lower[1..].parse::<u8>().ok();
            match number {
                Some(value @ 1..=12) => HotkeyKey::F(value),
                _ => return unsupported_key(&lower),
            }
        }
        _ => return unsupported_key(&lower),
    };
    Ok(key)
}

fn unsupported_key(key: &str) -> Result<HotkeyKey, String> {
    Err(format!(
        "Unsupported hotkey key '{key}'. Supported keys: {SUPPORTED_KEYS}; supported modifiers: {SUPPORTED_MODIFIERS}"
    ))
}

impl HotkeyKey {
    fn shortcut_name(self) -> String {
        match self {
            HotkeyKey::Space => "Space".into(),
            HotkeyKey::Home => "Home".into(),
            HotkeyKey::Insert => "Insert".into(),
            HotkeyKey::Pause => "Pause".into(),
            HotkeyKey::ScrollLock => "ScrollLock".into(),
            HotkeyKey::F(number) => format!("F{number}"),
        }
    }
}

#[cfg(target_os = "linux")]
#[derive(Clone, Copy, Debug)]
struct EvdevHotkey {
    key: evdev::Key,
    ctrl: bool,
    alt: bool,
    shift: bool,
}

#[cfg(target_os = "linux")]
impl TryFrom<ParsedHotkey> for EvdevHotkey {
    type Error = String;

    fn try_from(value: ParsedHotkey) -> Result<Self, Self::Error> {
        Ok(Self {
            key: evdev_key(value.key),
            ctrl: value.ctrl,
            alt: value.alt,
            shift: value.shift,
        })
    }
}

#[cfg(target_os = "linux")]
fn evdev_key(key: HotkeyKey) -> evdev::Key {
    match key {
        HotkeyKey::Space => evdev::Key::KEY_SPACE,
        HotkeyKey::Home => evdev::Key::KEY_HOME,
        HotkeyKey::Insert => evdev::Key::KEY_INSERT,
        HotkeyKey::Pause => evdev::Key::KEY_PAUSE,
        HotkeyKey::ScrollLock => evdev::Key::KEY_SCROLLLOCK,
        HotkeyKey::F(1) => evdev::Key::KEY_F1,
        HotkeyKey::F(2) => evdev::Key::KEY_F2,
        HotkeyKey::F(3) => evdev::Key::KEY_F3,
        HotkeyKey::F(4) => evdev::Key::KEY_F4,
        HotkeyKey::F(5) => evdev::Key::KEY_F5,
        HotkeyKey::F(6) => evdev::Key::KEY_F6,
        HotkeyKey::F(7) => evdev::Key::KEY_F7,
        HotkeyKey::F(8) => evdev::Key::KEY_F8,
        HotkeyKey::F(9) => evdev::Key::KEY_F9,
        HotkeyKey::F(10) => evdev::Key::KEY_F10,
        HotkeyKey::F(11) => evdev::Key::KEY_F11,
        HotkeyKey::F(12) => evdev::Key::KEY_F12,
        HotkeyKey::F(_) => unreachable!("F key parser only accepts F1-F12"),
    }
}

#[cfg(target_os = "linux")]
struct EvdevHotkeyListener {
    stop: std::sync::Arc<std::sync::atomic::AtomicBool>,
    handle: Option<std::thread::JoinHandle<()>>,
}

#[cfg(target_os = "linux")]
impl EvdevHotkeyListener {
    fn start(
        app: tauri::AppHandle,
        shortcut: String,
        hotkey: EvdevHotkey,
        mut devices: Vec<evdev::Device>,
    ) -> Result<Self, String> {
        for device in &devices {
            set_nonblocking(device)?;
        }
        let stop = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
        let thread_stop = stop.clone();
        let handle = std::thread::spawn(move || {
            let mut ctrl = false;
            let mut alt = false;
            let mut shift = false;
            let mut hotkey_down = false;
            while !thread_stop.load(std::sync::atomic::Ordering::Relaxed) {
                for device in &mut devices {
                    let Ok(events) = device.fetch_events() else {
                        continue;
                    };
                    for event in events {
                        let evdev::InputEventKind::Key(key) = event.kind() else {
                            continue;
                        };
                        let pressed = event.value() == 1;
                        let released = event.value() == 0;
                        if !pressed && !released {
                            continue;
                        }
                        match key {
                            evdev::Key::KEY_LEFTCTRL | evdev::Key::KEY_RIGHTCTRL => {
                                ctrl = pressed || (!released && ctrl);
                            }
                            evdev::Key::KEY_LEFTALT | evdev::Key::KEY_RIGHTALT => {
                                alt = pressed || (!released && alt);
                            }
                            evdev::Key::KEY_LEFTSHIFT | evdev::Key::KEY_RIGHTSHIFT => {
                                shift = pressed || (!released && shift);
                            }
                            _ => {}
                        }
                        if key == hotkey.key {
                            let modifiers_match =
                                ctrl == hotkey.ctrl && alt == hotkey.alt && shift == hotkey.shift;
                            if pressed && modifiers_match && !hotkey_down {
                                hotkey_down = true;
                                let _ = app.emit(
                                    "granite-hotkey",
                                    HotkeyEventPayload {
                                        state: "Pressed",
                                        shortcut: shortcut.clone(),
                                        backend: BACKEND_EVDEV.into(),
                                    },
                                );
                            } else if released && hotkey_down {
                                hotkey_down = false;
                                let _ = app.emit(
                                    "granite-hotkey",
                                    HotkeyEventPayload {
                                        state: "Released",
                                        shortcut: shortcut.clone(),
                                        backend: BACKEND_EVDEV.into(),
                                    },
                                );
                            }
                        }
                    }
                }
                std::thread::sleep(std::time::Duration::from_millis(10));
            }
        });
        Ok(Self {
            stop,
            handle: Some(handle),
        })
    }

    fn stop(mut self) {
        self.stop.store(true, std::sync::atomic::Ordering::Relaxed);
        if let Some(handle) = self.handle.take() {
            let _ = handle.join();
        }
    }
}

#[cfg(target_os = "linux")]
fn open_keyboard_devices(target_key: evdev::Key) -> Result<Vec<evdev::Device>, String> {
    let entries = std::fs::read_dir("/dev/input")
        .map_err(|error| format!("Could not inspect /dev/input for Wayland hotkeys: {error}"))?;
    let mut devices = Vec::new();
    let mut permission_errors = 0;
    for entry in entries.flatten() {
        let path = entry.path();
        let is_event = path
            .file_name()
            .and_then(|name| name.to_str())
            .map(|name| name.starts_with("event"))
            .unwrap_or(false);
        if !is_event {
            continue;
        }
        match evdev::Device::open(&path) {
            Ok(device) => {
                if device
                    .supported_keys()
                    .map(|keys| keys.contains(evdev::Key::KEY_A) && keys.contains(target_key))
                    .unwrap_or(false)
                {
                    devices.push(device);
                }
            }
            Err(error) if error.kind() == std::io::ErrorKind::PermissionDenied => {
                permission_errors += 1;
            }
            Err(_) => {}
        }
    }
    if !devices.is_empty() {
        return Ok(devices);
    }
    if permission_errors > 0 {
        Err("Wayland hotkeys cannot read /dev/input/event* devices. Add this user to the input group, log out and back in, then restart Granite Speach: sudo usermod -aG input $USER".into())
    } else {
        Err("Wayland hotkeys found no readable keyboard event devices under /dev/input".into())
    }
}

#[cfg(target_os = "linux")]
fn set_nonblocking(device: &evdev::Device) -> Result<(), String> {
    use std::os::fd::AsRawFd;
    let fd = device.as_raw_fd();
    let flags = unsafe { libc::fcntl(fd, libc::F_GETFL) };
    if flags < 0 {
        return Err("Could not read keyboard device flags".into());
    }
    let result = unsafe { libc::fcntl(fd, libc::F_SETFL, flags | libc::O_NONBLOCK) };
    if result < 0 {
        Err("Could not set keyboard device non-blocking mode".into())
    } else {
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_ctrl_alt_space() {
        let hotkey = parse_hotkey("Ctrl+Alt+Space").unwrap();

        assert_eq!(hotkey.shortcut, "Ctrl+Alt+Space");
        assert!(hotkey.ctrl);
        assert!(hotkey.alt);
        assert!(!hotkey.shift);
        assert_eq!(hotkey.key, HotkeyKey::Space);
    }

    #[test]
    fn parses_control_option_aliases() {
        let hotkey = parse_hotkey("Control+Option+Space").unwrap();

        assert_eq!(hotkey.shortcut, "Ctrl+Alt+Space");
        assert!(hotkey.ctrl);
        assert!(hotkey.alt);
        assert_eq!(hotkey.key, HotkeyKey::Space);
    }

    #[test]
    fn rejects_unsupported_key_with_supported_set() {
        let error = parse_hotkey("Ctrl+Alt+A").unwrap_err();

        assert!(error.contains("Unsupported hotkey key"));
        assert!(error.contains("Space"));
        assert!(error.contains("F1-F12"));
    }

    #[test]
    fn selects_evdev_on_wayland() {
        assert!(matches!(
            select_backend_for(Some("wayland"), None, "linux"),
            HotkeyBackend::Evdev
        ));
    }

    #[test]
    fn selects_tauri_global_shortcut_on_non_wayland() {
        assert!(matches!(
            select_backend_for(Some("x11"), None, "linux"),
            HotkeyBackend::TauriGlobalShortcut
        ));
    }
}
