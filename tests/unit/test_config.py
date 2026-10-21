from pathlib import Path

import pytest
from pydantic import ValidationError

from app.core.config import Settings
from tests.helpers import make_settings


def test_default_port_and_prompt_limit(tmp_path: Path) -> None:
    settings = make_settings(tmp_path)
    assert settings.port == 8010
    assert settings.prompt_max_chars == 2000
    assert settings.llm_timeout_seconds == 120
    assert settings.llm_max_tokens == 2048
    assert settings.rvc_conversion_timeout_seconds == 1800
    assert settings.llm_disable_thinking is True
    assert settings.task_backend == "inline"
    assert settings.cache_backend == "none"
    assert settings.music_provider == "minimax_music"
    assert settings.elevenlabs_music_model_id == "music_v2_5"


def test_minimax_music_defaults_and_provider(tmp_path: Path) -> None:
    settings = make_settings(tmp_path, music_provider="minimax_music")
    assert settings.minimax_base_url == "http://127.0.0.1:8111"
    assert settings.minimax_model == "MiniMaxAI/MiniMax-Music3"
    assert settings.minimax_seed == 42
    assert settings.minimax_num_inference_steps == 30
    assert settings.minimax_timeout_seconds == 7200


@pytest.mark.parametrize("value", ["auto", " AUTO ", " PCM_44100 "])
def test_elevenlabs_legacy_auto_output_is_migrated(tmp_path: Path, value: str) -> None:
    settings = make_settings(tmp_path, elevenlabs_music_output_format=value)
    assert settings.elevenlabs_music_output_format == "pcm_44100"


def test_elevenlabs_non_pcm_output_is_rejected_at_config_load(tmp_path: Path) -> None:
    with pytest.raises(ValidationError, match="必须为受支持的 pcm"):
        make_settings(tmp_path, elevenlabs_music_output_format="mp3_44100_128")


def test_reference_upload_limit_defaults_to_30_mb_and_is_configurable(tmp_path: Path) -> None:
    assert make_settings(tmp_path).audio_reference_max_upload_mb == 30
    configured = make_settings(tmp_path, audio_reference_max_upload_mb=31)
    assert configured.audio_reference_max_upload_mb == 31


def test_reference_upload_limit_uses_global_environment_variable(monkeypatch) -> None:
    monkeypatch.setenv("ELEVENLABS_REFERENCE_MAX_UPLOAD_MB", "29")
    monkeypatch.setenv("AUDIO_REFERENCE_MAX_UPLOAD_MB", "31")
    assert Settings(_env_file=None).audio_reference_max_upload_mb == 31


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (None, 2),
        ("bad", 2),
        ("1e400", 2),
        ("Infinity", 2),
        (0, 1),
        (1.4, 1),
        (1.5, 2),
        (4.8, 5),
        (6, 6),
        (20, 6),
    ],
)
def test_duration_matches_legacy_clamping(tmp_path: Path, value: object, expected: int) -> None:
    assert make_settings(tmp_path).normalize_duration(value) == expected


def test_invalid_duration_range_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(ValidationError):
        Settings(
            _env_file=None,
            output_dir=tmp_path,
            mock_full_song_path=tmp_path / "mock.mp3",
            min_duration_minutes=5,
            max_duration_minutes=2,
        )
    with pytest.raises(ValidationError):
        make_settings(tmp_path, max_duration_minutes=7)


def test_llm_output_budget_is_capped(tmp_path: Path) -> None:
    with pytest.raises(ValidationError):
        make_settings(tmp_path, llm_max_tokens=4097)


def test_retention_days_defaults_and_boundaries(tmp_path: Path) -> None:
    assert make_settings(tmp_path).song_retention_days == 3
    for days in (1, 3650):
        assert make_settings(tmp_path, song_retention_days=days).song_retention_days == days


@pytest.mark.parametrize("days", ["0", "3651", "1000000000"])
def test_invalid_retention_days_rejected_from_environment(monkeypatch, days):
    monkeypatch.setenv("SONG_RETENTION_DAYS", days)
    with pytest.raises(ValidationError, match="song_retention_days"):
        Settings(_env_file=None)


@pytest.mark.parametrize("days", [0, -1, 3651, 1_000_000_000])
def test_song_retention_days_are_bounded(tmp_path, days):
    with pytest.raises(ValidationError):
        make_settings(tmp_path, song_retention_days=days)

def test_style_llm_falls_back_to_main_llm(tmp_path: Path) -> None:
    settings = make_settings(
        tmp_path, llm_model="gpt-test", llm_timeout_seconds=99, llm_temperature=0.3
    )
    assert settings.style_llm_model == "gpt-test"
    assert settings.style_llm_timeout_seconds == 99
    assert settings.style_llm_temperature == 0.3
    assert settings.style_llm_base_url == settings.llm_base_url
    assert settings.style_llm_api_key == settings.llm_api_key
    assert settings.style_llm_inherits_main is True


def test_style_llm_blank_key_falls_back_only_without_independent_endpoint(
    tmp_path: Path,
) -> None:
    fallback = make_settings(tmp_path, llm_api_key="main-key", style_llm_api_key="")
    assert fallback.style_llm_api_key.get_secret_value() == "main-key"

    independent = make_settings(
        tmp_path,
        llm_api_key="main-key",
        style_llm_api_key="",
        style_llm_base_url="http://style.test/v1",
    )
    assert independent.style_llm_api_key.get_secret_value() == ""


def test_style_llm_url_inherits_main_chat_completions_url(tmp_path: Path) -> None:
    settings = make_settings(
        tmp_path,
        llm_chat_completions_url="https://main.test/v1/chat/completions",
    )
    assert settings.style_llm_url == settings.llm_url


def test_style_llm_base_url_ignores_main_chat_completions_url(tmp_path: Path) -> None:
    settings = make_settings(
        tmp_path,
        llm_chat_completions_url="https://main.test/v1/chat/completions",
        style_llm_base_url="http://style.test/v1",
        style_llm_model="qwen-test",
    )
    assert settings.style_llm_url == "http://style.test/v1/chat/completions"


def test_style_llm_overrides_main_llm_independently(tmp_path: Path) -> None:
    settings = make_settings(
        tmp_path,
        llm_model="gpt-test",
        style_llm_model="qwen-test",
        style_llm_base_url="http://style.test/v1",
        style_llm_temperature=0.7,
    )
    assert settings.llm_model == "gpt-test"
    assert settings.style_llm_model == "qwen-test"
    assert settings.style_llm_temperature == 0.7
    assert settings.style_llm_url == "http://style.test/v1/chat/completions"
    assert settings.style_llm_inherits_main is False
