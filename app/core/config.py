from __future__ import annotations

import math
import os
import re
from pathlib import Path
from typing import Literal

from pydantic import Field, SecretStr, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RVC_ASSET_ROOT = PROJECT_ROOT / "assets" / "rvc"


def _default_rvc_asset(directory: str, pattern: str, fallback: str) -> Path:
    root = RVC_ASSET_ROOT / directory
    matches = sorted(root.glob(pattern)) if root.is_dir() else []
    return matches[0] if len(matches) == 1 else root / fallback


def _resolve_project_path(value: object) -> Path:
    path = Path(str(value)).expanduser()
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path.resolve()


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    host: str = "0.0.0.0"
    port: int = Field(default=8010, ge=1, le=65535)
    public_base_url: str = ""
    output_dir: Path = PROJECT_ROOT / "output"
    song_retention_enabled: bool = False
    song_retention_dry_run: bool = True
    song_retention_days: int = Field(default=3, ge=1, le=3650)

    prompt_max_chars: int = Field(default=2000, ge=100, le=20_000)
    request_max_bytes: int = Field(default=16_384, ge=1024, le=1_048_576)
    default_duration_minutes: int = 2
    min_duration_minutes: int = Field(default=1, ge=1)
    max_duration_minutes: int = Field(default=6, ge=1, le=6)
    max_concurrent_generations: int = Field(default=1, ge=1)

    llm_api_key: SecretStr | None = None
    llm_base_url: str = "https://api.openai.com/v1"
    llm_chat_completions_url: str = ""
    llm_model: str = "gpt-4o-mini"
    llm_timeout_seconds: float = Field(default=120, gt=0)
    llm_temperature: float = Field(default=0.2, ge=0, le=2)
    llm_max_tokens: int = Field(default=2048, ge=1, le=4096)
    llm_disable_thinking: bool = True

    music_api_mode: Literal["mock", "real"] = "mock"
    music_provider: Literal["generic", "suno_api", "elevenlabs_music", "minimax_music"] = (
        "minimax_music"
    )
    mock_full_song_path: Path = PROJECT_ROOT / "output" / "mock_full.mp3"
    music_api_key: SecretStr | None = None
    music_api_base_url: str = ""
    music_generate_path: str = "/generate"
    music_status_path: str = "/generate/status"
    music_api_timeout_seconds: float = Field(default=240, gt=0)
    music_download_timeout_seconds: float = Field(default=120, gt=0)
    music_poll_interval_seconds: float = Field(default=5, ge=0)
    music_max_poll_attempts: int = Field(default=60, ge=1)
    music_model: str = ""
    music_callback_url: str = ""

    elevenlabs_api_key: SecretStr | None = None
    elevenlabs_music_base_url: str = "https://api.elevenlabs.io"
    elevenlabs_music_model_id: str = "music_v2_5"
    elevenlabs_music_output_format: str = "pcm_44100"
    elevenlabs_clear_chinese_vocal_mode: bool = True
    elevenlabs_force_instrumental: bool = False
    elevenlabs_bypass_global_proxy: bool = True
    # Audio reference (conditioning_ref) is documented for music_v2 / music_v2_5 only.
    elevenlabs_reference_condition_strength: Literal["low", "medium", "high", "xhigh"] = "medium"
    elevenlabs_reference_max_upload_mb: int = Field(default=30, ge=1, le=200)

    suno_create_timeout_seconds: float = Field(default=180, gt=0)
    suno_make_instrumental: bool = False
    suno_model: str = "v4.5-all"
    suno_model_fallbacks: str = ""
    suno_use_default_model_fallbacks: bool = False

    minimax_base_url: str = "http://127.0.0.1:8111"
    minimax_model: str = "MiniMaxAI/MiniMax-Music3"
    minimax_seed: int = 42
    minimax_num_inference_steps: int = Field(default=30, ge=1, le=100)
    minimax_timeout_seconds: float = Field(default=7200, gt=0)

    split_profile: Literal["fast", "balanced", "quality"] = "fast"
    split_timeout_seconds: float = Field(default=1800, gt=0)
    split_keep_workdir: bool = False
    demucs_model: str = ""
    demucs_device: str = ""
    demucs_jobs: str = ""
    demucs_shifts: str = ""
    demucs_overlap: str = ""
    demucs_segment: str = ""

    rvc_model_path: Path = Field(
        default_factory=lambda: _default_rvc_asset("weights", "*.pth", "model.pth")
    )
    rvc_index_path: Path | None = Field(
        default_factory=lambda: _default_rvc_asset("indices", "*.index", "model.index")
    )
    rvc_base_model_dir: Path | None = RVC_ASSET_ROOT / "base_model"
    rvc_model_version: Literal["v1", "v2"] = "v2"
    rvc_device: str = "cuda:0"
    rvc_cpu_threads: int = Field(default_factory=lambda: min(6, os.cpu_count() or 1), ge=1)
    rvc_max_upload_mb: int = Field(default=50, ge=1)
    rvc_conversion_timeout_seconds: float = Field(default=1800, gt=0)
    rvc_mix_timeout_seconds: float = Field(default=180, gt=0, allow_inf_nan=False)

    task_backend: Literal["inline"] = "inline"
    cache_backend: Literal["none"] = "none"
    event_backend: Literal["none"] = "none"
    redis_url: SecretStr | None = None
    queue_name: str = "ai-music-generation"
    cache_default_ttl_seconds: int = Field(default=3600, ge=1)
    cors_origins: str = "*"

    @field_validator(
        "output_dir",
        "mock_full_song_path",
        "rvc_model_path",
        mode="before",
    )
    @classmethod
    def resolve_project_path(cls, value: object) -> Path:
        return _resolve_project_path(value)

    @field_validator("rvc_index_path", "rvc_base_model_dir", mode="before")
    @classmethod
    def resolve_optional_project_path(cls, value: object) -> Path | None:
        if value in (None, ""):
            return None
        return _resolve_project_path(value)

    @field_validator(
        "public_base_url",
        "llm_base_url",
        "elevenlabs_music_base_url",
        "minimax_base_url",
        "music_api_base_url",
    )
    @classmethod
    def strip_trailing_slash(cls, value: str) -> str:
        return value.rstrip("/")

    @field_validator("elevenlabs_music_output_format")
    @classmethod
    def require_elevenlabs_pcm(cls, value: str) -> str:
        value = value.strip().lower()
        if value == "auto":
            return "pcm_44100"
        if not re.fullmatch(r"pcm_(8000|16000|22050|24000|32000|44100|48000)", value):
            raise ValueError("ELEVENLABS_MUSIC_OUTPUT_FORMAT 必须为受支持的 pcm_* 格式")
        return value

    @model_validator(mode="after")
    def validate_duration_range(self) -> Settings:
        if self.max_duration_minutes < self.min_duration_minutes:
            raise ValueError("MAX_DURATION_MINUTES must be >= MIN_DURATION_MINUTES")
        if (
            not self.min_duration_minutes
            <= self.default_duration_minutes
            <= self.max_duration_minutes
        ):
            raise ValueError("DEFAULT_DURATION_MINUTES must be within the configured range")
        return self

    @property
    def llm_url(self) -> str:
        return self.llm_chat_completions_url or f"{self.llm_base_url}/chat/completions"

    @property
    def cors_origin_list(self) -> list[str]:
        if self.cors_origins.strip() == "*":
            return ["*"]
        return [item.strip() for item in self.cors_origins.split(",") if item.strip()]

    @property
    def elevenlabs_reference_max_upload_bytes(self) -> int:
        return self.elevenlabs_reference_max_upload_mb * 1024 * 1024

    @property
    def rvc_max_upload_bytes(self) -> int:
        return self.rvc_max_upload_mb * 1024 * 1024

    def normalize_duration(self, value: object) -> int:
        try:
            parsed = float(value) if value is not None else float(self.default_duration_minutes)
        except (TypeError, ValueError):
            parsed = float(self.default_duration_minutes)
        if not math.isfinite(parsed):
            parsed = float(self.default_duration_minutes)
        rounded = int(parsed + 0.5) if parsed >= 0 else int(parsed - 0.5)
        return min(self.max_duration_minutes, max(self.min_duration_minutes, rounded))
