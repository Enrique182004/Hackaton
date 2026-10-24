# PyInstaller spec for the SAR Whiteboard Assistant desktop build (one folder, not one file: a 1 GB
# one-file exe would unpack torch to a temp folder on every start).
#
#   python packaging/fetch_models.py            # bundle the embedding model (offline first run)
#   pyinstaller packaging/sar.spec --noconfirm
#
# Windows -> dist/SAR Whiteboard Assistant/SAR Whiteboard Assistant.exe (console window: errors stay visible)
# macOS   -> dist/SAR Whiteboard Assistant.app (camera + microphone usage strings in Info.plist)
# Smoke test either with `--self-test`; see docs/DESKTOP_BUILDS.md.

import importlib.util
import re
import sys
from pathlib import Path

from PyInstaller.utils.hooks import collect_data_files, collect_submodules, copy_metadata

SPEC_DIR = Path(SPECPATH).resolve()  # noqa: F821  (SPECPATH is defined by PyInstaller)
ROOT = SPEC_DIR.parent
PACKAGE = ROOT / "biobench_sar"
APP_NAME = "SAR Whiteboard Assistant"
BUNDLE_ID = "org.codecore.sarwhiteboard"
VERSION = re.search(r'__version__\s*=\s*"([^"]+)"', (PACKAGE / "__init__.py").read_text(encoding="utf-8")).group(1)
IS_MAC = sys.platform == "darwin"
IS_WIN = sys.platform == "win32"
sys.path.insert(0, str(ROOT))  # collect_submodules("biobench_sar") imports the package from the checkout


def _existing(*names):
    """The modules among `names` that this environment has."""
    found = []
    for name in names:
        try:
            if importlib.util.find_spec(name) is not None:
                found.append(name)
        except ImportError:
            pass
    return found


# ---- data files ----------------------------------------------------------------------------------------
# Flask finds templates/static next to biobench_sar/app.py, and config.PACKAGE_DIR / "knowledge" holds the
# seed documents, so all three keep their place inside the package folder.
datas = [
    (str(PACKAGE / "templates"), "biobench_sar/templates"),
    (str(PACKAGE / "static"), "biobench_sar/static"),
    (str(PACKAGE / "knowledge"), "biobench_sar/knowledge"),
    (str(ROOT / "sar.env.example"), "."),
]
MODELS = SPEC_DIR / "models"
if MODELS.is_dir() and any(MODELS.iterdir()):
    datas.append((str(MODELS), "models"))  # launcher: resource_dir() / "models"
else:
    print("sar.spec: packaging/models is empty; the first run will download the embedding model")

datas += collect_data_files("whisper")  # assets/mel_filters.npz, gpt2 + multilingual .tiktoken, normalizers
datas += collect_data_files("chromadb")  # migrations/*.sql, log_config.yml

# Packages that check their own (or a dependency's) version through importlib.metadata at run time.
for dist in (
    "chromadb",
    "filelock",
    "flask",
    "flask-socketio",
    "huggingface-hub",
    "numpy",
    "openai-whisper",
    "packaging",
    "posthog",
    "python-engineio",
    "python-socketio",
    "pyyaml",
    "regex",
    "requests",
    "safetensors",
    "scikit-learn",
    "scipy",
    "sentence-transformers",
    "tiktoken",
    "tokenizers",
    "torch",
    "tqdm",
    "transformers",
):
    try:
        datas += copy_metadata(dist)
    except Exception:  # not installed in this environment
        pass

# ---- modules imported by name at run time ----------------------------------------------------------------
hiddenimports = [
    # Flask-SocketIO(async_mode="threading") loads its driver by name.
    "engineio.async_drivers.threading",
    "simple_websocket",
    "sentence_transformers",
    "whisper",
    "tiktoken_ext",
    "tiktoken_ext.openai_public",
    "cv2",
    "pyaudio",
    "edge_tts",
    "pymupdf",
    "docx",
    "pptx",
    "pytesseract",
    "pyttsx3.drivers",
    "pyttsx3.drivers.dummy",
]
if IS_WIN:
    hiddenimports += ["pyttsx3.drivers.sapi5", "comtypes.client"]
elif IS_MAC:
    hiddenimports += ["pyttsx3.drivers.nsss", "pyttsx3.drivers.avspeech"]
else:
    hiddenimports += ["pyttsx3.drivers.espeak", "pyttsx3.drivers._espeak"]


# sentence-transformers -> transformers picks the model class from config.json (BERT for MiniLM) by name.
# transformers 5 folded the *_fast tokenizer modules into the plain ones, so only list what exists.
hiddenimports += _existing(
    "transformers.models.bert.configuration_bert",
    "transformers.models.bert.modeling_bert",
    "transformers.models.bert.tokenization_bert",
    "transformers.models.bert.tokenization_bert_fast",
)
hiddenimports += collect_submodules("biobench_sar", filter=lambda name: name != "biobench_sar.__main__")
# chromadb picks its API, segment manager and telemetry classes from dotted-path strings in its Settings.
hiddenimports += collect_submodules("chromadb", filter=lambda name: not name.startswith("chromadb.test"))

# Test-only and notebook packages that dependencies import optionally.
excludes = [
    "pytest",
    "_pytest",
    "pluggy",
    "iniconfig",
    "hypothesis",
    "ruff",
    "IPython",
    "ipykernel",
    "jupyter",
    "jupyter_client",
    "notebook",
    "matplotlib",
    "tkinter",
    "tensorboard",
    "torch.utils.tensorboard",
    "chromadb.test",
    # Optional PyMuPDF add-ons (not in requirements.txt) that pymupdf imports when present; the app doesn't
    # use them, and pymupdf.layout would need ~50 MB of ONNX models collected as well.
    "pymupdf.layout",
    "pymupdf4llm",
]

a = Analysis(
    [str(SPEC_DIR / "launcher.py")],
    pathex=[str(ROOT)],
    binaries=[],
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=excludes,
    noarchive=False,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    # Unbuffered output, so logs show up live. UTF-8 mode is requested too, but CI builds have reported it
    # off (the self-test header says which), so the launcher also switches piped output to UTF-8 itself.
    [("u", None, "OPTION"), ("X utf8_mode=1", None, "OPTION")],
    exclude_binaries=True,
    name=APP_NAME,
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,  # UPX-packed DLLs trip antivirus scanners and save little next to torch
    console=not IS_MAC,  # macOS: a real .app; the launcher logs to the data folder when run from Finder
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,  # ad-hoc signature on macOS; see docs/DESKTOP_BUILDS.md for Gatekeeper
    entitlements_file=None,
)

coll = COLLECT(exe, a.binaries, a.datas, strip=False, upx=False, upx_exclude=[], name=APP_NAME)

if IS_MAC:
    app = BUNDLE(
        coll,
        name=f"{APP_NAME}.app",
        icon=None,
        bundle_identifier=BUNDLE_ID,
        version=VERSION,
        info_plist={
            "CFBundleName": APP_NAME,
            "CFBundleDisplayName": APP_NAME,
            "CFBundleShortVersionString": VERSION,
            "CFBundleVersion": VERSION,
            "LSMinimumSystemVersion": "12.0",
            "NSHighResolutionCapable": True,
            # Without these keys macOS refuses camera / microphone access instead of asking.
            "NSCameraUsageDescription": "The assistant watches the whiteboard through the webcam to see what is drawn.",
            "NSMicrophoneUsageDescription": "The assistant listens for spoken questions from the class.",
        },
    )
