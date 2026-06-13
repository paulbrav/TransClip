import io
import os
import subprocess
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path, PureWindowsPath
from unittest.mock import patch

from transclip.cli import main
from transclip.desktop.hotkey.macos import (
    HOTKEY_BUNDLE_ID,
    HOTKEY_LAUNCHD_LABEL,
    build_macos_hotkey_source,
    build_macos_toggle_wrapper,
    install_macos_hotkey,
    macos_hotkey_app_path,
    macos_hotkey_launch_agent_path,
    macos_hotkey_source_path,
    macos_hotkey_state_path,
    macos_toggle_wrapper_path,
    uninstall_macos_hotkey,
)
from transclip.settings import Settings

from tests.service_helpers import FakeRuntime, normalize_path_text


class MacOSHotkeyTests(unittest.TestCase):
    @unittest.skipIf(os.name == "nt", "generated macOS shell wrapper requires a POSIX shell")
    def test_toggle_wrapper_start_reaches_recording_without_health_probe(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            runtime = FakeRuntime(system="Darwin", home=root / "home")
            fake_bin = root / "bin"
            fake_bin.mkdir()
            calls_path = root / "curl-calls.txt"
            fake_curl = fake_bin / "curl"
            fake_curl.write_text(
                f"""#!/bin/sh
printf '%s\\n' "$*" >> {calls_path}
case "$*" in
  */record/start*)
    printf '%s\\n' '{{"status":"recording","already_recording":false}}'
    ;;
  */health*)
    printf '%s\\n' '{{"status":"ready"}}'
    ;;
  *)
    printf '%s\\n' '{{"error":"unexpected endpoint"}}'
    exit 22
    ;;
esac
""",
                encoding="utf-8",
            )
            fake_curl.chmod(0o755)
            wrapper = root / "transclip-toggle"
            script = build_macos_toggle_wrapper(Settings(host="127.0.0.1", port=8765), runtime=runtime)
            script = script.replace("LOCK=/tmp/transclip-toggle.lock", f"LOCK={root / 'transclip-toggle.lock'}")
            wrapper.write_text(script, encoding="utf-8")
            wrapper.chmod(0o755)
            env = {**os.environ, "PATH": f"{fake_bin}{os.pathsep}{os.environ.get('PATH', '')}"}

            result = subprocess.run([str(wrapper)], env=env, capture_output=True, text=True, timeout=5, check=False)

            self.assertEqual(result.returncode, 0, result.stderr)
            calls = calls_path.read_text(encoding="utf-8").splitlines()
            self.assertEqual(len(calls), 1)
            self.assertIn("/record/start", calls[0])
            self.assertNotIn("/health", calls[0])
            self.assertIn("listening\tRecording", macos_hotkey_state_path(runtime).read_text(encoding="utf-8"))

    @unittest.skipIf(os.name == "nt", "generated macOS shell wrapper requires a POSIX shell")
    def test_toggle_wrapper_stop_falls_back_when_start_reports_already_recording(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            runtime = FakeRuntime(system="Darwin", home=root / "home")
            fake_bin = root / "bin"
            fake_bin.mkdir()
            calls_path = root / "curl-calls.txt"
            paste_path = root / "paste.txt"
            fake_curl = fake_bin / "curl"
            fake_curl.write_text(
                f"""#!/bin/sh
printf '%s\\n' "$*" >> {calls_path}
case "$*" in
  */record/start*)
    printf '%s\\n' '{{"status":"recording","already_recording":true}}'
    ;;
  */record/stop*)
    printf '%s\\n' '{{"status":"ready","text":"hello pasted"}}'
    ;;
  */health*)
    printf '%s\\n' '{{"status":"recording"}}'
    ;;
  *)
    printf '%s\\n' '{{"error":"unexpected endpoint"}}'
    exit 22
    ;;
esac
""",
                encoding="utf-8",
            )
            fake_curl.chmod(0o755)
            fake_pbcopy = fake_bin / "pbcopy"
            fake_pbcopy.write_text(f"#!/bin/sh\ncat > {paste_path}\n", encoding="utf-8")
            fake_pbcopy.chmod(0o755)
            wrapper = root / "transclip-toggle"
            script = build_macos_toggle_wrapper(Settings(host="127.0.0.1", port=8765), runtime=runtime)
            script = script.replace("LOCK=/tmp/transclip-toggle.lock", f"LOCK={root / 'transclip-toggle.lock'}")
            wrapper.write_text(script, encoding="utf-8")
            wrapper.chmod(0o755)
            env = {**os.environ, "PATH": f"{fake_bin}{os.pathsep}{os.environ.get('PATH', '')}"}

            result = subprocess.run([str(wrapper)], env=env, capture_output=True, text=True, timeout=5, check=False)

            self.assertEqual(result.returncode, 0, result.stderr)
            calls = calls_path.read_text(encoding="utf-8").splitlines()
            self.assertGreaterEqual(len(calls), 2)
            joined_calls = "\n".join(calls)
            start_index = next(index for index, call in enumerate(calls) if "/record/start" in call)
            stop_index = next(index for index, call in enumerate(calls) if "/record/stop" in call)
            self.assertLess(start_index, stop_index)
            self.assertIn("/health", joined_calls)
            self.assertEqual(paste_path.read_text(encoding="utf-8"), "hello pasted")
            self.assertIn(
                "paste_requested\tPaste transcript",
                macos_hotkey_state_path(runtime).read_text(encoding="utf-8"),
            )

    @unittest.skipIf(os.name == "nt", "generated macOS shell wrapper requires a POSIX shell")
    def test_toggle_wrapper_keeps_recording_state_until_stop_returns(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            runtime = FakeRuntime(system="Darwin", home=root / "home")
            fake_bin = root / "bin"
            fake_bin.mkdir()
            state_path = macos_hotkey_state_path(runtime)
            calls_path = root / "curl-calls.txt"
            paste_path = root / "paste.txt"
            fake_curl = fake_bin / "curl"
            fake_curl.write_text(
                f"""#!/bin/sh
printf '%s\\n' "$*" >> {calls_path}
case "$*" in
  */record/start*)
    printf '%s\\n' '{{"status":"recording","already_recording":true}}'
    ;;
  */record/stop*)
    if ! grep -q '	stopping	Stopping recording' {state_path}; then
      printf 'unexpected state before stop returned: ' >&2
      cat {state_path} >&2
      exit 42
    fi
    printf '%s\\n' '{{"status":"ready","text":"hello pasted"}}'
    ;;
  *)
    printf '%s\\n' '{{"error":"unexpected endpoint"}}'
    exit 22
    ;;
esac
""",
                encoding="utf-8",
            )
            fake_curl.chmod(0o755)
            fake_pbcopy = fake_bin / "pbcopy"
            fake_pbcopy.write_text(f"#!/bin/sh\ncat > {paste_path}\n", encoding="utf-8")
            fake_pbcopy.chmod(0o755)
            wrapper = root / "transclip-toggle"
            script = build_macos_toggle_wrapper(Settings(host="127.0.0.1", port=8765), runtime=runtime)
            script = script.replace("LOCK=/tmp/transclip-toggle.lock", f"LOCK={root / 'transclip-toggle.lock'}")
            wrapper.write_text(script, encoding="utf-8")
            wrapper.chmod(0o755)
            env = {**os.environ, "PATH": f"{fake_bin}{os.pathsep}{os.environ.get('PATH', '')}"}

            result = subprocess.run([str(wrapper)], env=env, capture_output=True, text=True, timeout=5, check=False)

            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(paste_path.read_text(encoding="utf-8"), "hello pasted")

    @unittest.skipIf(os.name == "nt", "generated macOS shell wrapper requires a POSIX shell")
    def test_toggle_wrapper_marks_transcribing_after_capture_stops(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            runtime = FakeRuntime(system="Darwin", home=root / "home")
            fake_bin = root / "bin"
            fake_bin.mkdir()
            state_path = macos_hotkey_state_path(runtime)
            calls_path = root / "curl-calls.txt"
            paste_path = root / "paste.txt"
            restart_path = root / "restart.txt"
            stop_started = root / "stop-started"
            fake_curl = fake_bin / "curl"
            fake_curl.write_text(
                f"""#!/bin/sh
printf '%s\\n' "$*" >> {calls_path}
case "$*" in
  */record/start*)
    printf '%s\\n' '{{"status":"recording","already_recording":true}}'
    ;;
  */record/stop*)
    touch {stop_started}
    sleep 1
    if ! grep -q '	transcribing	Transcribing' {state_path}; then
      printf 'expected transcribing before stop response: ' >&2
      cat {state_path} >&2
      exit 42
    fi
    printf '%s\\n' '{{"status":"ready","text":"hello pasted"}}'
    ;;
  */health*)
    if [ -f {restart_path} ]; then
      printf '%s\\n' '{{"status":"ready"}}'
    elif [ -f {stop_started} ]; then
      printf '%s\\n' '{{"status":"transcribing"}}'
    else
      printf '%s\\n' '{{"status":"recording"}}'
    fi
    ;;
  *)
    printf '%s\\n' '{{"error":"unexpected endpoint"}}'
    exit 22
    ;;
esac
""",
                encoding="utf-8",
            )
            fake_curl.chmod(0o755)
            fake_pbcopy = fake_bin / "pbcopy"
            fake_pbcopy.write_text(f"#!/bin/sh\ncat > {paste_path}\n", encoding="utf-8")
            fake_pbcopy.chmod(0o755)
            wrapper = root / "transclip-toggle"
            script = build_macos_toggle_wrapper(
                Settings(host="127.0.0.1", port=8765),
                runtime=runtime,
                restart_command=f"printf restarted > {restart_path}",
            )
            script = script.replace("LOCK=/tmp/transclip-toggle.lock", f"LOCK={root / 'transclip-toggle.lock'}")
            wrapper.write_text(script, encoding="utf-8")
            wrapper.chmod(0o755)
            env = {**os.environ, "PATH": f"{fake_bin}{os.pathsep}{os.environ.get('PATH', '')}"}

            result = subprocess.run([str(wrapper)], env=env, capture_output=True, text=True, timeout=5, check=False)

            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(paste_path.read_text(encoding="utf-8"), "hello pasted")
            calls = calls_path.read_text(encoding="utf-8")
            self.assertIn("/health", calls)

    @unittest.skipIf(os.name == "nt", "generated macOS shell wrapper requires a POSIX shell")
    def test_toggle_wrapper_timeout_recovers_to_ready_after_restart(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            runtime = FakeRuntime(system="Darwin", home=root / "home")
            fake_bin = root / "bin"
            fake_bin.mkdir()
            calls_path = root / "curl-calls.txt"
            restart_path = root / "restart.txt"
            fake_curl = fake_bin / "curl"
            fake_curl.write_text(
                f"""#!/bin/sh
printf '%s\\n' "$*" >> {calls_path}
case "$*" in
  */record/start*)
    printf '%s\\n' '{{"status":"recording","already_recording":true}}'
    ;;
  */record/stop*)
    exit 28
    ;;
  */health*)
    printf '%s\\n' '{{"status":"ready"}}'
    ;;
  *)
    printf '%s\\n' '{{"error":"unexpected endpoint"}}'
    exit 22
    ;;
esac
""",
                encoding="utf-8",
            )
            fake_curl.chmod(0o755)
            wrapper = root / "transclip-toggle"
            script = build_macos_toggle_wrapper(
                Settings(host="127.0.0.1", port=8765),
                runtime=runtime,
                restart_command=f"printf restarted > {restart_path}",
            )
            script = script.replace("LOCK=/tmp/transclip-toggle.lock", f"LOCK={root / 'transclip-toggle.lock'}")
            wrapper.write_text(script, encoding="utf-8")
            wrapper.chmod(0o755)
            env = {**os.environ, "PATH": f"{fake_bin}{os.pathsep}{os.environ.get('PATH', '')}"}

            result = subprocess.run([str(wrapper)], env=env, capture_output=True, text=True, timeout=5, check=False)

            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(restart_path.read_text(encoding="utf-8"), "restarted")
            self.assertIn("/health", calls_path.read_text(encoding="utf-8"))
            self.assertIn("ready\tReady", macos_hotkey_state_path(runtime).read_text(encoding="utf-8"))

    def test_toggle_wrapper_uses_explicit_start_stop_and_lock(self):
        runtime = FakeRuntime(system="Darwin", home=Path("/Users/test"))
        script = build_macos_toggle_wrapper(Settings(host="127.0.0.1", port=8765), runtime=runtime)

        self.assertIn("LOCK=/tmp/transclip-toggle.lock", script)
        self.assertIn("/record/start", script)
        self.assertIn("/record/stop", script)
        self.assertIn("MAX_SECONDS=75", script)
        self.assertIn("STALE_LOCK_SECONDS=90", script)
        self.assertIn('printf \'%s\\n\' "$$" > "$LOCK/pid"', script)
        self.assertIn('write_state recovering "Clearing stale action"', script)
        self.assertIn("kill_process_tree", script)
        self.assertIn("STATE=/Users/test/Library/Logs/transclip/hotkey-state.tsv", script)
        self.assertIn("STOP_POLL_SECONDS=0.05", script)
        self.assertIn('write_state shortcut "Starting recording"', script)
        self.assertIn('write_state listening "Recording"', script)
        self.assertIn('write_state stopping "Stopping recording"', script)
        self.assertIn('write_state paste_requested "Paste transcript"', script)
        self.assertIn('write_state finished "No transcript"', script)
        self.assertIn('write_state recovering "Restarting service"', script)
        self.assertIn('write_state ready "Ready"', script)
        self.assertIn('write_state error "Service restart failed"', script)
        self.assertIn("wait_for_ready", script)
        self.assertIn('if [ -f "$stop_status_file" ]; then', script)
        self.assertIn('sleep "$STOP_POLL_SECONDS"', script)
        self.assertIn("already_recording", script)
        self.assertIn("$BASE/health", script)
        self.assertNotIn("osascript -e", script)
        self.assertIn("stop failed; restarting service", script)

    def test_generated_macos_artifacts_use_forward_slash_paths(self):
        runtime = FakeRuntime(system="Darwin", home=PureWindowsPath("/Users/test"))
        script = build_macos_toggle_wrapper(Settings(host="127.0.0.1", port=8765), runtime=runtime)
        source = build_macos_hotkey_source(
            PureWindowsPath("/Users/test/bin/transclip-toggle"),
            PureWindowsPath("/Users/test/Library/Logs/transclip/hotkey.log"),
            PureWindowsPath("/Users/test/Library/Logs/transclip/hotkey-state.tsv"),
        )

        self.assertIn("STATE=/Users/test/Library/Logs/transclip/hotkey-state.tsv", script)
        self.assertIn('let statePath = "/Users/test/Library/Logs/transclip/hotkey-state.tsv"', source)

    def test_hotkey_source_builds_status_item_from_state_file(self):
        source = build_macos_hotkey_source(
            Path("/Users/test/bin/transclip-toggle"),
            Path("/Users/test/Library/Logs/transclip/hotkey.log"),
            Path("/Users/test/Library/Logs/transclip/hotkey-state.tsv"),
        )

        self.assertIn("import AppKit", source)
        self.assertIn('let statePath = "/Users/test/Library/Logs/transclip/hotkey-state.tsv"', source)
        self.assertIn("NSStatusBar.system.statusItem", source)
        self.assertIn("NSAttributedString", source)
        self.assertIn("systemOrange", source)
        self.assertIn("systemPurple", source)
        self.assertIn("systemTeal", source)
        self.assertIn("pollState", source)
        self.assertIn('title = "TXT..."', source)
        self.assertIn('title = "PST..."', source)
        self.assertIn('case "listening":', source)
        self.assertIn('case "stopping":', source)
        self.assertIn('case "recovering":', source)
        self.assertIn('case "transcribing":', source)
        self.assertIn('case "paste_requested":', source)
        self.assertIn('case "finished":', source)
        self.assertIn('case "ready":', source)
        self.assertIn("return .labelColor", source)
        self.assertIn("postCommandV", source)
        self.assertIn('writeStateFile("finished", "Pasted")', source)
        self.assertIn("var activeEventTap: CFMachPort?", source)
        self.assertIn("activeEventTap = eventTap", source)
        self.assertIn("CGEvent.tapEnable(tap: tap, enable: true)", source)
        self.assertIn("event tap re-enabled", source)
        self.assertIn("event tap listening for Option+Space", source)
        self.assertIn('writeStateFile("ready", "Ready")', source)

    def test_installer_writes_helper_app_launch_agent_and_wrapper(self):
        calls = []

        def runner(command, **_kwargs):
            calls.append(command)
            if command[0].endswith("swiftc") and "-o" in command:
                output = Path(command[command.index("-o") + 1])
                output.parent.mkdir(parents=True, exist_ok=True)
                output.write_text("# fake binary", encoding="utf-8")
            return type("Completed", (), {"returncode": 0, "stdout": ""})()

        with tempfile.TemporaryDirectory() as tmp:
            home = Path(tmp)
            runtime = FakeRuntime(
                system="Darwin",
                home=home,
                available={"swiftc": "/usr/bin/swiftc", "codesign": "/usr/bin/codesign"},
                check_output_text="501",
            )
            install, results = install_macos_hotkey(
                Settings(host="127.0.0.1", port=8765),
                runner=runner,
                runtime=runtime,
            )

            self.assertEqual(install.app_path, home / "Applications" / "TransClipHotkey.app")
            self.assertTrue(install.wrapper_path.exists())
            self.assertTrue(install.source_path.exists())
            self.assertTrue((install.app_path / "Contents" / "Info.plist").exists())
            self.assertTrue(install.launch_agent_path.exists())
            self.assertIn("hotkey-state.tsv", install.source_path.read_text(encoding="utf-8"))
            self.assertIn(
                [
                    "/usr/bin/swiftc",
                    str(install.source_path),
                    "-o",
                    str(install.app_path / "Contents/MacOS/TransClipHotkey"),
                ],
                calls,
            )
            self.assertIn(["/usr/bin/codesign", "--force", "--deep", "--sign", "-", str(install.app_path)], calls)
            self.assertIn(["launchctl", "bootstrap", "gui/501", str(install.launch_agent_path)], calls)
            self.assertTrue(any("Grant Accessibility" in result.detail for result in results))
            self.assertFalse(any("System Events" in result.detail for result in results))
            plist_text = normalize_path_text(install.launch_agent_path.read_text(encoding="utf-8"))
            self.assertIn(HOTKEY_LAUNCHD_LABEL, plist_text)
            self.assertIn("TransClipHotkey.app/Contents/MacOS/TransClipHotkey", plist_text)
            info_text = (install.app_path / "Contents" / "Info.plist").read_text(encoding="utf-8")
            self.assertIn(HOTKEY_BUNDLE_ID, info_text)

    def test_cli_install_macos_hotkey_prints_paths_and_results(self):
        install = type(
            "Install",
            (),
            {
                "app_path": Path("/Users/test/Applications/TransClipHotkey.app"),
                "launch_agent_path": Path("/Users/test/Library/LaunchAgents/com.paulbrav.transclip-hotkey.plist"),
                "wrapper_path": Path("/Users/test/bin/transclip-toggle"),
            },
        )()
        result = type("Result", (), {"ok": True, "detail": "wrote helper"})()

        stdout = io.StringIO()
        with (
            patch("transclip.cli.shortcut_cmd.install_macos_hotkey", return_value=(install, [result])) as installer,
            redirect_stdout(stdout),
        ):
            code = main(["install-macos-hotkey"])

        self.assertEqual(code, 0)
        self.assertTrue(installer.called)
        self.assertIn("TransClipHotkey.app", stdout.getvalue())
        self.assertIn("ok\twrote helper", stdout.getvalue())

    def test_uninstall_removes_launch_agent_app_wrapper_and_source(self):
        calls = []

        def runner(command, **_kwargs):
            calls.append(command)
            return type("Completed", (), {"returncode": 0, "stdout": ""})()

        with tempfile.TemporaryDirectory() as tmp:
            home = Path(tmp)
            runtime = FakeRuntime(system="Darwin", home=home, check_output_text="501")
            app_path = macos_hotkey_app_path(runtime)
            launch_agent_path = macos_hotkey_launch_agent_path(runtime)
            wrapper_path = macos_toggle_wrapper_path(runtime)
            source_path = macos_hotkey_source_path(runtime)
            (app_path / "Contents").mkdir(parents=True)
            launch_agent_path.parent.mkdir(parents=True)
            wrapper_path.parent.mkdir(parents=True)
            source_path.parent.mkdir(parents=True)
            launch_agent_path.write_text("plist", encoding="utf-8")
            wrapper_path.write_text("wrapper", encoding="utf-8")
            source_path.write_text("swift", encoding="utf-8")

            results = uninstall_macos_hotkey(runner=runner, runtime=runtime)

            self.assertIn(["launchctl", "bootout", "gui/501/com.paulbrav.transclip-hotkey"], calls)
            self.assertFalse(app_path.exists())
            self.assertFalse(launch_agent_path.exists())
            self.assertFalse(wrapper_path.exists())
            self.assertFalse(source_path.exists())
            self.assertTrue(any("removed" in result.detail for result in results))


if __name__ == "__main__":
    unittest.main()
