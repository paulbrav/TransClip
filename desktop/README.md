# Granite Speach Desktop

Tauri 2 shell for the local Granite Speach inference service.

The shell expects `uv run -m granite_speach.cli serve` to be running on
`http://127.0.0.1:8765`. It provides tray actions, global hotkey recording,
browser-side WAV capture, and latest transcript copy/paste controls.

```bash
npm install
npm run build
npm run tauri dev
```

On Linux, install the Tauri prerequisites first, including WebKitGTK,
JavaScriptCoreGTK, libsoup 3, and rsvg.
