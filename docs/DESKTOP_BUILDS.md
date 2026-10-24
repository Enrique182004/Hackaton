# Desktop builds (Windows installer, macOS app)

Teachers should not need Python, a terminal or `pip`. CI turns the repo into:

| Platform | File | What it is |
| --- | --- | --- |
| Windows 10/11 (x64) | `SAR-Whiteboard-Assistant-Setup-<version>.exe` | Per-user installer (no administrator rights), Start-menu shortcut, optional desktop shortcut, uninstaller |
| macOS 12+ (Apple Silicon) | `SAR-Whiteboard-Assistant-<version>-macOS-arm64.zip` (~480 MB) | `SAR Whiteboard Assistant.app` (~1.2 GB unpacked) |

Both bundle Python, every package in `requirements.txt` (CPU-only PyTorch) and the
`all-MiniLM-L6-v2` embedding model (~90 MB), so the knowledge base works offline from the first start.
They do **not** bundle:

- **Ollama and the vision model** (`qwen2.5vl:3b`, ~3 GB): installed once per computer, see below.
- **Whisper `base`** (~140 MB): downloaded the first time someone uses the microphone, into
  `~/.cache/whisper` (`%USERPROFILE%\.cache\whisper` on Windows). A manual CI run can bundle it instead
  (`bundle_whisper`), for rooms with no internet.
- **Tesseract** (optional): only needed to read text from uploaded photos and scanned PDFs.

Phones and tablets can't run the assistant (it needs PyTorch, Whisper and Ollama), but they can drive
it: see [Phones and tablets](#phones-and-tablets).

## Getting a build

`.github/workflows/build.yml` runs on Windows (`windows-latest`) and macOS (`macos-14`, arm64):

- **Pull requests** that touch `packaging/**`, `biobench_sar/paths.py`, `scripts/build_*` or the workflow:
  build + smoke test; the installer and zip are kept as run artifacts for 14 days
  (Actions → the run → *Artifacts*).
- **Manual run**: Actions → *desktop builds* → *Run workflow* (tick *bundle_whisper* to ship Whisper too).
- **Release**: bump `__version__` in `biobench_sar/__init__.py` (the only place the version lives), merge,
  then tag and push: `git tag v0.1.0 && git push origin v0.1.0`. The workflow refuses a tag that doesn't
  match `__version__`, builds both platforms and attaches them to a GitHub release (a tag with a `-`,
  such as `v0.2.0-rc1`, makes a pre-release).

Every build is smoke-tested before it is uploaded: CI runs the built app with `--self-test` (see
[Command line](#command-line)), which starts the real server with a mock model and checks the pages,
Socket.IO, every lazily imported library, the bundled embedding model and Whisper's assets.

## First run (for teachers)

### Windows

1. Download `SAR-Whiteboard-Assistant-Setup-<version>.exe` and open it. The builds are not code-signed
   yet, so SmartScreen says *Windows protected your PC*: click **More info → Run anyway**.
2. Install. It goes to `%LOCALAPPDATA%\Programs\SAR Whiteboard Assistant`; no administrator needed.
3. Install Ollama from <https://ollama.com/download>, then open *Command Prompt* and run once:
   `ollama pull qwen2.5vl:3b` (~3 GB).
4. Start **SAR Whiteboard Assistant** from the Start menu. A console window shows the addresses and any
   error (keep it open; closing it stops the assistant), and the dashboard opens in the browser.
5. Open `http://127.0.0.1:5050/` full screen on the projector (drag the browser window to the projector
   display, press F11).
6. If the camera or microphone stays dark: *Settings → Privacy & security → Camera* (and *Microphone*) →
   turn on **Let desktop apps access your camera**.

### macOS

1. Download the zip, double-click it, and drag **SAR Whiteboard Assistant** into *Applications*
   (run it from there: a quarantined app in *Downloads* is run from a hidden read-only copy).
2. The app is not notarized yet, so the first open is refused. macOS 15+: try to open it once, then
   *System Settings → Privacy & Security* → **Open Anyway**. macOS 14 and older: right-click the app →
   **Open** → **Open**. (Same effect from Terminal:
   `xattr -dr com.apple.quarantine "/Applications/SAR Whiteboard Assistant.app"`.)
3. Install Ollama from <https://ollama.com/download> and run `ollama pull qwen2.5vl:3b` in Terminal (~3 GB).
4. Open the app. Allow the **camera** and **microphone** when macOS asks. There is no window: the
   dashboard opens in the browser and the app shows in the Dock. If something is missing (Ollama, a busy
   port) a dialog says so; the full log is in
   `~/Library/Application Support/SAR Whiteboard Assistant/logs/sar.log`.
5. To stop it: right-click its Dock icon → **Quit** (hold Option for **Force Quit** if it doesn't react).
   To watch the log live instead, start it from Terminal:
   `"/Applications/SAR Whiteboard Assistant.app/Contents/MacOS/SAR Whiteboard Assistant"` (Ctrl+C stops it).

### If Ollama isn't ready

The app still starts and the dashboard works (uploads, calibration, camera). The console (or, on macOS,
a dialog) says what to do, and answers say the model is unreachable until Ollama runs and the model is
pulled. A missing model only needs `ollama pull qwen2.5vl:3b`; no restart.

## Settings: `sar.env`

Room settings (camera index, projector size, subject, model, port, …) are the same `SAR_*` variables as
in the [README](../README.md); `sar.env.example` documents the common ones. Put them in a file called
`sar.env` (copy the example, then edit). The app reads, in this order, and the first value it finds for
a setting wins:

1. real environment variables;
2. the file named by `SAR_ENV_FILE`, if set;
3. `sar.env` **next to the app**: next to `SAR Whiteboard Assistant.exe` (the installer puts
   `sar.env.example` there), or in the folder that holds `SAR Whiteboard Assistant.app`;
4. `sar.env` in the **data folder** (below), where the app leaves a `sar.env.example` on first start.

Use the data-folder copy when several people share one computer image and the app folder is read-only;
use the one next to the app to carry a room's settings on a USB stick with a portable copy.
`--port N` on the command line beats all of them.

## Where data lives

Everything the app writes (calibration, the knowledge base, uploads, speech audio, quiz bank, file
selection, logs) goes to a per-user data folder, never next to the program:

| OS | Data folder |
| --- | --- |
| Windows | `%LOCALAPPDATA%\SAR Whiteboard Assistant` (Start menu → *SAR Whiteboard Assistant data folder*) |
| macOS | `~/Library/Application Support/SAR Whiteboard Assistant` |
| Linux | `$XDG_DATA_HOME/sar-whiteboard-assistant` (default `~/.local/share/sar-whiteboard-assistant`) |

Inside: `data/` (`calibration.json`, `chroma_db/`, `uploads/`, `tts/`, `selection.json`, `quiz_bank.json`),
`logs/` (macOS Finder launches), `sar.env.example`, and your `sar.env` if you put one there.
`SAR_DATA_DIR` moves `data/` elsewhere (e.g. a shared drive). Deleting `data/chroma_db/` resets the
knowledge base; the bundled seed documents come back on the next start.

## Uninstall

- **Windows:** *Settings → Apps → Installed apps → SAR Whiteboard Assistant → Uninstall*. Your data
  folder and any `sar.env` you created next to the exe are kept; delete
  `%LOCALAPPDATA%\SAR Whiteboard Assistant` (and `%USERPROFILE%\.cache\whisper`) to remove everything.
- **macOS:** drag the app to the Bin, then delete `~/Library/Application Support/SAR Whiteboard Assistant`
  and `~/.cache/whisper`. To forget the camera/microphone permission too:
  `tccutil reset All org.codecore.sarwhiteboard`.
- Ollama is a separate app: uninstall it the usual way (`ollama rm qwen2.5vl:3b` frees the 3 GB model).

## Command line

The built app (`SAR Whiteboard Assistant.exe`, or `….app/Contents/MacOS/SAR Whiteboard Assistant`) and
`python packaging/launcher.py` from a checkout take the same flags:

| Flag | Effect |
| --- | --- |
| `--no-browser` | Don't open the dashboard in the default browser |
| `--port N` | Serve on port N (default `SAR_PORT`, else 5050) |
| `--self-test` | Start on a free port with `SAR_VLM_BACKEND=mock`, `SAR_TTS_BACKEND=none`, `SAR_LISTEN_MODE=push` and a throw-away data folder (no camera, microphone or Ollama needed); check `/api/status`, `/dashboard`, `/`, static files, the Socket.IO handshake, imports, embeddings and Whisper assets; print `SELF-TEST OK` and exit 0, or the failures and exit 1, within 120 s |
| `--version` | Print the version |

`--self-test` is also the quickest way to check a teacher's install: run it from a terminal and send us
the output.

## Phones and tablets

Our focus is computers: the assistant needs a laptop or desktop next to the projector. A phone or
tablet can still be the **remote dashboard** (upload files, ask, run the tutor, watch status) while the
computer does the work, because the dashboard is a web page:

1. In `sar.env` set `SAR_HOST=0.0.0.0` and restart the app. The console now also prints
   `On a phone or tablet (same Wi-Fi): http://<computer-ip>:5050/dashboard`.
2. Allow the firewall prompt (Windows: *Private networks*). Put the phone on the same Wi-Fi and open
   that address; *Add to Home Screen* gives it an app-like icon.
3. Camera, microphone and speakers stay the computer's.

Only do this on a trusted classroom network: anyone who can reach that address can use the dashboard
(the server rejects requests from other web pages, but has no login).

## Building locally

The scripts do what CI does, on your machine.

**Windows** (PowerShell, after `scripts\setup_windows.ps1`):

```powershell
winget install JRSoftware.InnoSetup        # once, for the installer
.\scripts\build_windows.ps1                # -Whisper, -SkipSelfTest, -SkipInstaller
```

**macOS** (after `scripts/setup_mac.sh`):

```bash
bash scripts/build_macos.sh                # --whisper, --skip-self-test, --out /tmp/sar-build
```

Both install PyInstaller into `.venv`, run `packaging/fetch_models.py`, build `packaging/sar.spec`, run
the self-test, then make the installer / zip in `dist/`. Sizes from a macOS arm64 build: the app is
~1.2 GB unpacked (PyTorch alone is ~400 MB), the zip ~480 MB, the PyInstaller work folder ~300 MB; it
took about 2 minutes on an M4 with packages already installed. Plan for ~3 GB of free disk and delete
`build/` and `dist/` afterwards.
`packaging/models/`, `packaging/build/` and `packaging/dist/` are git-ignored.

Without a full build, `python packaging/launcher.py --self-test` checks the launcher and the app from
the checkout.

## How it fits together (for developers)

| File | Role |
| --- | --- |
| `biobench_sar/__init__.py` | `__version__`, the single source of the version (spec, installer, release tag) |
| `biobench_sar/paths.py` | `is_frozen()`, `resource_dir()` (`sys._MEIPASS` in a build, the checkout otherwise), `executable_dir()`, `user_data_dir()` |
| `packaging/launcher.py` | Entry point of the builds: loads `sar.env` files, sets `SAR_DATA_DIR`, points `SENTENCE_TRANSFORMERS_HOME` / `HF_HOME` at the bundled models with `HF_HUB_OFFLINE=1`, checks Ollama, opens the browser, then runs `biobench_sar.app.main()`; also `--self-test` |
| `packaging/fetch_models.py` | Downloads the embedding model (and with `--whisper` the Whisper checkpoint) into `packaging/models/` in the Hugging Face cache layout, with symlinks replaced by files; idempotent |
| `packaging/sar.spec` | PyInstaller one-folder spec: package data, hidden imports, Info.plist for macOS |
| `packaging/windows/installer.iss` | Inno Setup script for the Windows installer |
| `scripts/build_windows.ps1`, `scripts/build_macos.sh` | Local builds |

When you add a dependency that the app imports inside a function (as it does for parsers, TTS and
audio), add it to `RUNTIME_MODULES` in `packaging/launcher.py` so the self-test catches a build that
lost it; if PyInstaller can't see the import (a module named in a string), add it to `hiddenimports`
in `packaging/sar.spec` too. A frozen app that fails with `ModuleNotFoundError` almost always needs
exactly that.

Not done yet: code signing (removes the SmartScreen / Gatekeeper steps), an Intel macOS build, an app
icon, and an in-app *Quit* button for macOS.
