use std::fs;
use std::net::TcpStream;
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::sync::{Mutex, OnceLock};
use std::time::Duration;
use tauri::Manager;

mod hotkey;

static MANAGED_SERVICE: OnceLock<Mutex<Option<Child>>> = OnceLock::new();

#[tauri::command]
fn simulate_paste() -> Result<(), String> {
    #[cfg(target_os = "macos")]
    {
        let status = Command::new("osascript")
            .arg("-e")
            .arg("tell application \"System Events\" to keystroke \"v\" using command down")
            .status()
            .map_err(|error| error.to_string())?;
        return if status.success() {
            Ok(())
        } else {
            Err("osascript paste command failed".into())
        };
    }

    #[cfg(target_os = "linux")]
    {
        let mut errors = Vec::new();
        if command_exists("wtype") {
            match run_command("wtype", &["-M", "ctrl", "v", "-m", "ctrl"]) {
                Ok(()) => return Ok(()),
                Err(error) => errors.push(error),
            }
        }
        if command_exists("xdotool") {
            match run_command("xdotool", &["key", "ctrl+v"]) {
                Ok(()) => return Ok(()),
                Err(error) => errors.push(error),
            }
        }
        if command_exists("ydotool") {
            match run_command("ydotool", &["key", "ctrl+v"]) {
                Ok(()) => return Ok(()),
                Err(error) => errors.push(error),
            }
        }
        if errors.is_empty() {
            Err("No supported paste injector found: install wtype, xdotool, or ydotool".into())
        } else {
            Err(errors.join("; "))
        }
    }

    #[cfg(not(any(target_os = "macos", target_os = "linux")))]
    {
        Err("Paste injection is only implemented for Linux and macOS".into())
    }
}

#[tauri::command]
fn open_config_file(kind: &str) -> Result<(), String> {
    let path = config_path(kind)?;
    ensure_config_file(kind, &path)?;
    open_path(&path)
}

#[tauri::command]
fn start_service() -> Result<String, String> {
    if service_is_running() {
        return Ok("service already running".into());
    }

    let repo_root = find_repo_root()?;
    let service = service_command(&repo_root);
    let mut command = Command::new(&service.program);
    command.args(&service.args).current_dir(&repo_root);
    if let Some(venv) = &service.virtual_env {
        command.env("VIRTUAL_ENV", venv);
    }
    let child = command
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .spawn()
        .map_err(|error| format!("Could not start service with {}: {error}", service.program))?;

    let mutex = MANAGED_SERVICE.get_or_init(|| Mutex::new(None));
    let mut slot = mutex
        .lock()
        .map_err(|_| "Service lock is poisoned".to_string())?;
    *slot = Some(child);
    Ok("service starting".into())
}

#[tauri::command]
fn quit(app: tauri::AppHandle) {
    hotkey::stop_hotkey(&app);
    stop_managed_service();
    app.exit(0);
}

#[tauri::command]
fn show_notification(title: &str, message: &str) -> Result<bool, String> {
    #[cfg(target_os = "macos")]
    {
        let script = format!(
            "display notification \"{}\" with title \"{}\"",
            escape_osascript(message),
            escape_osascript(title)
        );
        return Command::new("osascript")
            .arg("-e")
            .arg(script)
            .status()
            .map(|status| status.success())
            .map_err(|error| error.to_string());
    }

    #[cfg(target_os = "linux")]
    {
        if command_exists("notify-send") {
            return Command::new("notify-send")
                .args([title, message])
                .status()
                .map(|status| status.success())
                .map_err(|error| error.to_string());
        }
        Ok(false)
    }

    #[cfg(not(any(target_os = "macos", target_os = "linux")))]
    {
        let _ = title;
        let _ = message;
        Ok(false)
    }
}

fn config_path(kind: &str) -> Result<String, String> {
    let home = std::env::var("HOME").map_err(|_| "HOME is not set".to_string())?;
    let filename = match kind {
        "settings" => "settings.toml",
        "keywords" => "keywords.txt",
        _ => return Err("Unknown config file kind".into()),
    };
    #[cfg(target_os = "macos")]
    {
        Ok(format!(
            "{home}/Library/Application Support/granite-speach/{filename}"
        ))
    }
    #[cfg(not(target_os = "macos"))]
    {
        Ok(format!("{home}/.config/granite-speach/{filename}"))
    }
}

fn ensure_config_file(kind: &str, path: &str) -> Result<(), String> {
    let path = Path::new(path);
    if path.exists() {
        return Ok(());
    }
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).map_err(|error| error.to_string())?;
    }
    let content = match kind {
        "settings" => DEFAULT_SETTINGS,
        "keywords" => DEFAULT_KEYWORDS,
        _ => return Err("Unknown config file kind".into()),
    };
    fs::write(path, content).map_err(|error| error.to_string())
}

fn open_path(path: &str) -> Result<(), String> {
    #[cfg(target_os = "macos")]
    let status = Command::new("open")
        .arg(path)
        .status()
        .map_err(|error| error.to_string())?;

    #[cfg(target_os = "linux")]
    let status = Command::new("xdg-open")
        .arg(path)
        .status()
        .map_err(|error| error.to_string())?;

    #[cfg(not(any(target_os = "macos", target_os = "linux")))]
    let status = Command::new("cmd")
        .args(["/C", "start", path])
        .status()
        .map_err(|error| error.to_string())?;

    if status.success() {
        Ok(())
    } else {
        Err(format!("Could not open {path}"))
    }
}

fn service_is_running() -> bool {
    TcpStream::connect_timeout(
        &"127.0.0.1:8765"
            .parse()
            .expect("valid localhost socket address"),
        Duration::from_millis(200),
    )
    .is_ok()
}

fn find_repo_root() -> Result<PathBuf, String> {
    let mut candidates = Vec::new();
    if let Ok(current_dir) = std::env::current_dir() {
        candidates.push(current_dir);
    }
    if let Ok(exe) = std::env::current_exe() {
        if let Some(parent) = exe.parent() {
            candidates.push(parent.to_path_buf());
        }
    }
    for candidate in candidates {
        for ancestor in candidate.ancestors() {
            if ancestor.join("pyproject.toml").exists() && ancestor.join("granite_speach").is_dir()
            {
                return Ok(ancestor.to_path_buf());
            }
        }
    }
    Err("Could not find granite-speach repository root".into())
}

struct ServiceCommand {
    program: String,
    args: Vec<String>,
    virtual_env: Option<PathBuf>,
}

fn service_command(repo_root: &Path) -> ServiceCommand {
    let uv = std::env::var("GRANITE_SPEACH_UV").unwrap_or_else(|_| "uv".into());
    let mut args = vec!["run".into(), "--active".into()];
    let mut virtual_env = None;
    for relative in [".venv-gfx1151/bin/python", ".venv/bin/python"] {
        let path = repo_root.join(relative);
        if path.exists() {
            if let Some(parent) = path.parent().and_then(Path::parent) {
                virtual_env = Some(parent.to_path_buf());
            }
            break;
        }
    }
    args.extend(["-m".into(), "granite_speach.cli".into(), "serve".into()]);
    ServiceCommand {
        program: uv,
        args,
        virtual_env,
    }
}

fn stop_managed_service() {
    let Some(mutex) = MANAGED_SERVICE.get() else {
        return;
    };
    let Ok(mut slot) = mutex.lock() else {
        return;
    };
    if let Some(mut child) = slot.take() {
        let _ = child.kill();
        let _ = child.wait();
    }
}

#[cfg(target_os = "macos")]
fn escape_osascript(value: &str) -> String {
    value.replace('\\', "\\\\").replace('"', "\\\"")
}

const DEFAULT_SETTINGS: &str = r#"hotkey_linux = "<Super><Shift>XF86TouchpadOff"
hotkey_macos = "Option+Space"
language = "en"

asr_model = "ibm-granite/granite-speech-4.1-2b-nar"
cleanup_model = "google/gemma-4-E2B-it"
cleanup_enabled = true
cleanup_runtime = "rule"
cleanup_model_path = ""
models_local_files_only = true
model_cache_dir = ""

restore_clipboard_after_paste = true
clipboard_restore_delay_ms = 500

max_recording_seconds = 60
min_recording_ms = 250

debug_capture = false
debug_capture_dir = "debug-captures"

asr_backend = "granite_nar"
asr_device = "auto"
sample_rate = 16000
host = "127.0.0.1"
port = 8765
"#;

const DEFAULT_KEYWORDS: &str = r#"PyTorch
ROCm
gfx1151
Tauri
llama.cpp
Gemma
Granite
Qwen
Transformers
Hugging Face
MLX
Wayland
"#;

#[cfg(target_os = "linux")]
fn command_exists(command: &str) -> bool {
    Command::new("sh")
        .arg("-c")
        .arg(format!("command -v {command} >/dev/null 2>&1"))
        .status()
        .map(|status| status.success())
        .unwrap_or(false)
}

#[cfg(target_os = "linux")]
fn run_command(program: &str, args: &[&str]) -> Result<(), String> {
    let output = Command::new(program)
        .args(args)
        .output()
        .map_err(|error| format!("{program} paste command could not start: {error}"))?;
    if output.status.success() {
        return Ok(());
    }
    let stderr = String::from_utf8_lossy(&output.stderr).trim().to_string();
    let stdout = String::from_utf8_lossy(&output.stdout).trim().to_string();
    let detail = if !stderr.is_empty() {
        stderr
    } else if !stdout.is_empty() {
        stdout
    } else {
        format!("exit status {}", output.status)
    };
    Err(format!("{program} paste command failed: {detail}"))
}

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    tauri::Builder::default()
        .plugin(tauri_plugin_opener::init())
        .plugin(tauri_plugin_clipboard_manager::init())
        .setup(|app| {
            allow_linux_microphone_requests(app);
            #[cfg(desktop)]
            app.handle()
                .plugin(tauri_plugin_global_shortcut::Builder::new().build())?;
            Ok(())
        })
        .invoke_handler(tauri::generate_handler![
            simulate_paste,
            open_config_file,
            start_service,
            hotkey::configure_hotkey,
            show_notification,
            quit
        ])
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}

#[cfg(target_os = "linux")]
fn allow_linux_microphone_requests(app: &tauri::App) {
    use webkit2gtk::{
        glib::prelude::*, PermissionRequestExt, UserMediaPermissionRequest, WebViewExt,
    };

    if let Some(webview) = app.get_webview_window("main") {
        let _ = webview.with_webview(|webview| {
            webview.inner().connect_permission_request(|_, request| {
                if request.is::<UserMediaPermissionRequest>() {
                    request.allow();
                    true
                } else {
                    false
                }
            });
        });
    }
}

#[cfg(not(target_os = "linux"))]
fn allow_linux_microphone_requests(_: &tauri::App) {}
