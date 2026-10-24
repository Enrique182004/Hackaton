from __future__ import annotations

import math
import os
from dataclasses import dataclass, field
from pathlib import Path

PACKAGE_DIR = Path(__file__).resolve().parent
DATA_DIR = Path(os.environ.get("SAR_DATA_DIR", PACKAGE_DIR.parent / "data"))


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


def _env_ink_c(name: str, default: int) -> int:
    value = _env_int(name, default)
    return value if value >= 0 else default


def _env_block_size(name: str, default: int) -> int:
    return normalize_block_size(_env_int(name, default), default)


def normalize_block_size(value: int, default: int = 31) -> int:
    """Adaptive-threshold block size must be odd and >= 3."""
    if value < 1:
        return default
    value = max(value, 3)
    return value if value % 2 == 1 else value + 1


@dataclass(frozen=True)
class Settings:
    camera_index: int = field(default_factory=lambda: _env_int("SAR_CAMERA_INDEX", 0))
    camera_width: int = field(default_factory=lambda: _env_int("SAR_CAMERA_WIDTH", 1280))
    camera_height: int = field(default_factory=lambda: _env_int("SAR_CAMERA_HEIGHT", 720))
    projector_width: int = field(default_factory=lambda: _env_int("SAR_PROJECTOR_WIDTH", 1920))
    projector_height: int = field(default_factory=lambda: _env_int("SAR_PROJECTOR_HEIGHT", 1080))
    checker_cols: int = field(default_factory=lambda: _env_int("SAR_CHECKER_COLS", 9))
    checker_rows: int = field(default_factory=lambda: _env_int("SAR_CHECKER_ROWS", 6))
    calibration_file: Path = field(
        default_factory=lambda: Path(os.environ.get("SAR_CALIBRATION_FILE", DATA_DIR / "calibration.json"))
    )
    chroma_dir: Path = field(default_factory=lambda: Path(os.environ.get("SAR_CHROMA_DIR", DATA_DIR / "chroma_db")))
    upload_dir: Path = field(default_factory=lambda: Path(os.environ.get("SAR_UPLOAD_DIR", DATA_DIR / "uploads")))
    tts_dir: Path = field(default_factory=lambda: Path(os.environ.get("SAR_TTS_DIR", DATA_DIR / "tts")))
    embed_model: str = field(default_factory=lambda: os.environ.get("SAR_EMBED_MODEL", "all-MiniLM-L6-v2"))
    rag_top_k: int = field(default_factory=lambda: _env_int("SAR_RAG_TOP_K", 4))
    # ollama | openai | mock
    vlm_backend: str = field(default_factory=lambda: os.environ.get("SAR_VLM_BACKEND", "ollama"))
    vlm_model: str = field(default_factory=lambda: os.environ.get("SAR_VLM_MODEL", "qwen2.5vl:3b"))
    vlm_base_url: str = field(default_factory=lambda: os.environ.get("SAR_VLM_BASE_URL", "http://localhost:11434"))
    vlm_api_key: str | None = field(default_factory=lambda: os.environ.get("SAR_VLM_API_KEY"))
    vlm_timeout: float = field(default_factory=lambda: float(os.environ.get("SAR_VLM_TIMEOUT", "90")))
    whisper_model: str = field(default_factory=lambda: os.environ.get("SAR_WHISPER_MODEL", "base"))
    wake_word: str = field(default_factory=lambda: os.environ.get("SAR_WAKE_WORD", "assistant"))
    # push (hotkey/button only) | wake (always listening for wake word)
    listen_mode: str = field(default_factory=lambda: os.environ.get("SAR_LISTEN_MODE", "push"))
    record_seconds: float = field(default_factory=lambda: float(os.environ.get("SAR_RECORD_SECONDS", "5")))
    # Wake-word chunks quieter than this RMS skip Whisper; push-to-talk only skips digital silence.
    silence_rms: float = field(default_factory=lambda: _env_nonneg_float("SAR_SILENCE_RMS", 0.01))
    # edge | pyttsx3 | say | none
    tts_backend: str = field(default_factory=lambda: os.environ.get("SAR_TTS_BACKEND", "edge"))
    tts_voice: str = field(default_factory=lambda: os.environ.get("SAR_TTS_VOICE", "en-US-AriaNeural"))
    ink_min_area: int = field(default_factory=lambda: _env_pos_int("SAR_INK_MIN_AREA", 150))
    ink_block_size: int = field(default_factory=lambda: _env_block_size("SAR_INK_BLOCK_SIZE", 31))
    ink_c: int = field(default_factory=lambda: _env_ink_c("SAR_INK_C", 15))
    ink_max_regions: int = field(default_factory=lambda: _env_pos_int("SAR_INK_MAX_REGIONS", 10))
    host: str = field(default_factory=lambda: os.environ.get("SAR_HOST", "127.0.0.1"))
    # 5050, not Flask's 5000: macOS AirPlay Receiver listens on 5000.
    port: int = field(default_factory=lambda: _env_int("SAR_PORT", 5050))
    seed_knowledge: bool = field(default_factory=lambda: os.environ.get("SAR_SEED_KNOWLEDGE", "1") != "0")

    @property
    def ink_kwargs(self) -> dict[str, int]:
        """Keyword arguments for vision_loop.find_ink_regions."""
        return {
            "min_area": self.ink_min_area,
            "max_regions": self.ink_max_regions,
            "block_size": self.ink_block_size,
            "c": self.ink_c,
        }

    @property
    def projector_size(self) -> tuple[int, int]:
        return (self.projector_width, self.projector_height)
