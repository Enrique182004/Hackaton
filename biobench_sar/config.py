from __future__ import annotations

import math
import os
from dataclasses import dataclass, field
from pathlib import Path

from biobench_sar.lan import is_lan_mode, parse_hostnames
from biobench_sar.lang import normalize_language

PACKAGE_DIR = Path(__file__).resolve().parent
DATA_DIR = Path(os.environ.get("SAR_DATA_DIR", PACKAGE_DIR.parent / "data"))
PROFILES = ("math", "biology")
# Math boards are many small symbols (digits, signs, exponents): more regions survive the cut than a flowchart's.
INK_MAX_REGIONS = {"math": 20, "biology": 10}
# The classroom kit's Epson WXGA projector (16:10); SAR_PROJECTOR_WIDTH / SAR_PROJECTOR_HEIGHT override it.
PROJECTOR_SIZE = (1280, 800)
TRUE_WORDS = frozenset({"1", "true", "yes", "on"})


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except ValueError:
        return default


def _env_pos_int(name: str, default: int, minimum: int = 1) -> int:
    value = _env_int(name, default)
    return value if value >= minimum else default


def _env_nonneg_float(name: str, default: float) -> float:
    try:
        value = float(os.environ.get(name, default))
    except ValueError:
        return default
    return value if math.isfinite(value) and value >= 0 else default


def _env_nonneg_int(name: str, default: int) -> int:
    value = _env_int(name, default)
    return value if value >= 0 else default


def _env_optional_pos_int(name: str) -> int:
    """The variable as a positive int, or 0 when it is unset or invalid (the caller picks the default)."""
    value = _env_int(name, 0)
    return value if value > 0 else 0


def _env_pos_float(name: str, default: float) -> float:
    value = _env_nonneg_float(name, default)
    return value if value > 0 else default


def _env_ink_c(name: str, default: int) -> int:
    value = _env_int(name, default)
    return value if value >= 0 else default


def _env_block_size(name: str, default: int) -> int:
    return normalize_block_size(_env_int(name, default), default)


def lan_requested() -> bool:
    """SAR_LAN=1 (or true/yes/on): phones and tablets on the classroom Wi-Fi may join with the PIN."""
    return os.environ.get("SAR_LAN", "").strip().lower() in TRUE_WORDS


def _env_host() -> str:
    # An explicit SAR_HOST wins, "" (every interface) included; LAN mode alone listens on every interface.
    host = os.environ.get("SAR_HOST")
    if host is not None:
        return host.strip()
    return "0.0.0.0" if lan_requested() else "127.0.0.1"


def _hostnames(value: object) -> tuple[str, ...]:
    if isinstance(value, str):
        return parse_hostnames(value)
    if isinstance(value, (list, tuple, set, frozenset)):
        # Each item must be one valid name: "bad name" is not two names.
        names = (parse_hostnames(str(v)) for v in value)
        return tuple(dict.fromkeys(parsed[0] for parsed in names if len(parsed) == 1))
    return ()


def _env_profile() -> str:
    value = os.environ.get("SAR_PROFILE", PROFILES[0]).strip().lower()
    return value if value in PROFILES else PROFILES[0]


def _env_tts_backend() -> str:
    value = os.environ.get("SAR_TTS_BACKEND", "edge").strip().lower()
    # say and pyttsx3 (the computer's adult voices) were removed: an old sar.env that names them gets the kid voices.
    return "edge" if value in ("say", "pyttsx3") else value


def normalize_block_size(value: int, default: int = 31) -> int:
    """Adaptive-threshold block size must be odd and >= 3."""
    if value < 1:
        return default
    value = max(value, 3)
    return value if value % 2 == 1 else value + 1


@dataclass(frozen=True)
class Settings:
    # Subject the assistant tutors: picks the prompts, the seed documents and whether tutor modes apply.
    profile: str = field(default_factory=_env_profile)
    camera_index: int = field(default_factory=lambda: _env_int("SAR_CAMERA_INDEX", 0))
    camera_width: int = field(default_factory=lambda: _env_int("SAR_CAMERA_WIDTH", 1280))
    camera_height: int = field(default_factory=lambda: _env_int("SAR_CAMERA_HEIGHT", 720))
    projector_width: int = field(default_factory=lambda: _env_pos_int("SAR_PROJECTOR_WIDTH", PROJECTOR_SIZE[0]))
    projector_height: int = field(default_factory=lambda: _env_pos_int("SAR_PROJECTOR_HEIGHT", PROJECTOR_SIZE[1]))
    checker_cols: int = field(default_factory=lambda: _env_int("SAR_CHECKER_COLS", 9))
    checker_rows: int = field(default_factory=lambda: _env_int("SAR_CHECKER_ROWS", 6))
    calibration_file: Path = field(
        default_factory=lambda: Path(os.environ.get("SAR_CALIBRATION_FILE", DATA_DIR / "calibration.json"))
    )
    chroma_dir: Path = field(default_factory=lambda: Path(os.environ.get("SAR_CHROMA_DIR", DATA_DIR / "chroma_db")))
    upload_dir: Path = field(default_factory=lambda: Path(os.environ.get("SAR_UPLOAD_DIR", DATA_DIR / "uploads")))
    tts_dir: Path = field(default_factory=lambda: Path(os.environ.get("SAR_TTS_DIR", DATA_DIR / "tts")))
    # Which uploaded files answers may use: {"all": true} or {"sources": [...]}.
    selection_file: Path = field(
        default_factory=lambda: Path(os.environ.get("SAR_SELECTION_FILE", DATA_DIR / "selection.json"))
    )
    quiz_bank_file: Path = field(
        default_factory=lambda: Path(os.environ.get("SAR_QUIZ_BANK_FILE", DATA_DIR / "quiz_bank.json"))
    )
    # The room's accessibility settings (captions, text size, voice speed, calm mode, high contrast).
    accessibility_file: Path = field(
        default_factory=lambda: Path(os.environ.get("SAR_ACCESSIBILITY_FILE", DATA_DIR / "accessibility.json"))
    )
    # Problems per quiz: more in the ticked files -> an even spread over their levels; none -> this many invented.
    quiz_length: int = field(default_factory=lambda: _env_pos_int("SAR_QUIZ_LENGTH", 10))
    embed_model: str = field(default_factory=lambda: os.environ.get("SAR_EMBED_MODEL", "all-MiniLM-L6-v2"))
    rag_top_k: int = field(default_factory=lambda: _env_int("SAR_RAG_TOP_K", 4))
    # ollama | openai | mock
    vlm_backend: str = field(default_factory=lambda: os.environ.get("SAR_VLM_BACKEND", "ollama"))
    vlm_model: str = field(default_factory=lambda: os.environ.get("SAR_VLM_MODEL", "qwen2.5vl:3b"))
    vlm_base_url: str = field(default_factory=lambda: os.environ.get("SAR_VLM_BASE_URL", "http://localhost:11434"))
    vlm_api_key: str | None = field(default_factory=lambda: os.environ.get("SAR_VLM_API_KEY"))
    vlm_timeout: float = field(default_factory=lambda: _env_pos_float("SAR_VLM_TIMEOUT", 90.0))
    whisper_model: str = field(default_factory=lambda: os.environ.get("SAR_WHISPER_MODEL", "base"))
    wake_word: str = field(default_factory=lambda: os.environ.get("SAR_WAKE_WORD", "assistant"))
    # push (hotkey/button only) | wake (always listening for wake word)
    listen_mode: str = field(default_factory=lambda: os.environ.get("SAR_LISTEN_MODE", "push"))
    record_seconds: float = field(default_factory=lambda: _env_pos_float("SAR_RECORD_SECONDS", 5.0))
    # Wake-word chunks quieter than this RMS skip Whisper; push-to-talk only skips digital silence.
    silence_rms: float = field(default_factory=lambda: _env_nonneg_float("SAR_SILENCE_RMS", 0.01))
    # edge (the kid voices) | none (answers are shown only)
    tts_backend: str = field(default_factory=_env_tts_backend)
    tts_voice: str = field(default_factory=lambda: os.environ.get("SAR_TTS_VOICE", "en-US-AnaNeural"))
    # auto (reply in the language the student uses) | en | es
    language: str = field(default_factory=lambda: normalize_language(os.environ.get("SAR_LANGUAGE")))
    ink_min_area: int = field(default_factory=lambda: _env_pos_int("SAR_INK_MIN_AREA", 150))
    ink_block_size: int = field(default_factory=lambda: _env_block_size("SAR_INK_BLOCK_SIZE", 31))
    ink_c: int = field(default_factory=lambda: _env_ink_c("SAR_INK_C", 15))
    # 0 = the profile's default (INK_MAX_REGIONS); an explicit SAR_INK_MAX_REGIONS always wins.
    ink_max_regions: int = field(default_factory=lambda: _env_optional_pos_int("SAR_INK_MAX_REGIONS"))
    # Specks this close to a symbol (decimal points, minus signs, exponents) join its region; 0 turns it off.
    ink_small_merge_px: int = field(default_factory=lambda: _env_nonneg_int("SAR_INK_SMALL_MERGE_PX", 24))
    host: str = field(default_factory=_env_host)
    # 5050, not Flask's 5000: macOS AirPlay Receiver listens on 5000.
    port: int = field(default_factory=lambda: _env_int("SAR_PORT", 5050))
    seed_knowledge: bool = field(default_factory=lambda: os.environ.get("SAR_SEED_KNOWLEDGE", "1") != "0")
    # The PIN gate for phones (docs/MOBILE.md). Never off for a bind other devices can reach: __post_init__ turns
    # it on for any host but 127.0.0.0/8, ::1 and localhost, so the server can't listen on the network ungated.
    lan: bool = field(default_factory=lan_requested)
    # Host names besides IP literals and localhost that the gate accepts (e.g. "my-laptop.local").
    lan_hostnames: tuple[str, ...] = field(default_factory=lambda: parse_hostnames(os.environ.get("SAR_LAN_HOSTNAMES")))

    def __post_init__(self) -> None:
        object.__setattr__(self, "lan", is_lan_mode(sar_lan=bool(self.lan), bind_host=self.host))
        object.__setattr__(self, "lan_hostnames", _hostnames(self.lan_hostnames))
        if self.ink_max_regions <= 0:
            object.__setattr__(self, "ink_max_regions", INK_MAX_REGIONS.get(self.profile, INK_MAX_REGIONS["math"]))
        if self.language not in ("auto", "en", "es"):
            object.__setattr__(self, "language", normalize_language(self.language))

    @property
    def ink_kwargs(self) -> dict[str, int]:
        """Keyword arguments for vision_loop.find_ink_regions."""
        return {
            "min_area": self.ink_min_area,
            "max_regions": self.ink_max_regions,
            "block_size": self.ink_block_size,
            "c": self.ink_c,
            "small_merge_px": self.ink_small_merge_px,
        }

    @property
    def projector_size(self) -> tuple[int, int]:
        return (self.projector_width, self.projector_height)

    @property
    def knowledge_dir(self) -> Path:
        """Bundled seed documents for the current profile."""
        return PACKAGE_DIR / "knowledge" / self.profile

    @property
    def seed_marker_file(self) -> Path:
        # Next to the store it describes: deleting the store to reset it also brings the seed docs back.
        return self.chroma_dir / ".seeded"
