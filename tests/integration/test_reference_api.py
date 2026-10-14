from __future__ import annotations

import math
import struct
import wave
from pathlib import Path

import pytest

from app.core.errors import GenerationError
from app.main import create_app
from app.services.providers import MusicResult
from tests.helpers import client, make_orchestrator, make_settings


def sine_wav(path: Path, seconds: float) -> bytes:
    rate = 8_000
    frames = b"".join(
        struct.pack("<h", int(8_000 * math.sin(2 * math.pi * 440 * index / rate)))
        for index in range(int(rate * seconds))
    )
    with wave.open(str(path), "wb") as audio:
        audio.setparams((1, 2, rate, 0, "NONE", "not compressed"))
        audio.writeframes(frames)
    return path.read_bytes()


class FakeElevenLabs:
    def __init__(self, source: Path, *, upload_error: Exception | None = None) -> None:
        self.source = source
        self.upload_error = upload_error
        self.uploads: list[tuple[str, float]] = []
        self.references = []

    async def upload_reference(self, path: Path, filename: str) -> str:
        if self.upload_error:
            raise self.upload_error
        self.uploads.append((filename, path.stat().st_size))
        return f"song_{len(self.uploads)}"

    async def generate(self, *args, reference=None, **kwargs) -> MusicResult:
        self.references.append(reference)
        return MusicResult(self.source, {"provider": "elevenlabs_music", "durationSeconds": 3})


def reference_app(tmp_path: Path, **updates):
    settings = make_settings(
        tmp_path, music_api_mode="real", elevenlabs_reference_max_upload_mb=1, **updates
    )
    orchestrator = make_orchestrator(settings)
    fake = FakeElevenLabs(settings.mock_full_song_path)
    orchestrator.music_providers["elevenlabs_music"] = fake
    return create_app(settings, orchestrator), settings, fake


async def test_upload_trims_lists_previews_and_deletes(tmp_path: Path) -> None:
    app, settings, fake = reference_app(tmp_path)
    audio = sine_wav(tmp_path / "long.wav", 40)
    async with client(app) as http:
        uploaded = await http.post(
            "/api/references", files={"file": ("我的 参考.wav", audio, "audio/wav")}
        )
        listed = await http.get("/api/references")
        preview = await http.get(uploaded.json()["audioUrl"].removeprefix("http://testserver"))
        index = await http.get("/output/references/index.json")
        deleted = await http.delete("/api/references/song_1")
        deleted_again = await http.delete("/api/references/song_1")
        after = await http.get("/api/references")

    assert uploaded.status_code == 201, uploaded.text
    body = uploaded.json()
    assert body["id"] == "song_1"
    assert body["name"] == "我的 参考.wav"
    # Trimmed locally to the documented 30 s conditioning limit before the billed upload.
    assert 29 <= body["durationSeconds"] <= 30
    assert fake.uploads[0][0] == "我的 参考.mp3"
    assert listed.json()["references"] == [body]
    assert listed.json()["maxSeconds"] == 30
    assert preview.status_code == 200 and preview.content[:3] in {b"ID3", b"\xff\xfb", b"\xff\xf3"}
    assert index.status_code == 404
    assert deleted.status_code == 204
    assert deleted_again.status_code == 404
    assert after.json()["references"] == []
    assert not list((settings.output_dir / "references").glob("*.mp3"))


@pytest.mark.parametrize(
    ("content", "status", "message"),
    [
        (b"", 400, "为空"),
        (b"definitely not audio", 400, "无法解析"),
        ("short", 400, "至少需要 3 秒"),
    ],
)
async def test_upload_rejects_unusable_files_without_billing(
    tmp_path: Path, content, status, message
) -> None:
    app, settings, fake = reference_app(tmp_path)
    if content == "short":
        content = sine_wav(tmp_path / "short.wav", 1)
    async with client(app) as http:
        response = await http.post("/api/references", files={"file": ("a.wav", content)})
    assert response.status_code == status
    assert message in response.json()["message"]
    assert fake.uploads == []
    assert not [p for p in (settings.output_dir / "references").iterdir() if p.is_file()]


@pytest.mark.parametrize("size", [1024 * 1024 + 10, 3 * 1024 * 1024])
async def test_upload_rejects_oversized_file(tmp_path: Path, size: int) -> None:
    app, _, fake = reference_app(tmp_path)
    async with client(app) as http:
        response = await http.post("/api/references", files={"file": ("big.wav", b"\x00" * size)})
    assert response.status_code == 413
    assert fake.uploads == []


async def test_upload_provider_failure_keeps_no_files(tmp_path: Path) -> None:
    app, settings, fake = reference_app(tmp_path)
    fake.upload_error = GenerationError("ElevenLabs 参考音频上传失败：HTTP 422: copyright")
    async with client(app) as http:
        response = await http.post(
            "/api/references", files={"file": ("a.wav", sine_wav(tmp_path / "a.wav", 4))}
        )
    assert response.status_code == 502
    assert "copyright" in response.json()["message"]
    assert not list((settings.output_dir / "references").glob("*.mp3"))


async def test_upload_requires_real_mode(tmp_path: Path) -> None:
    settings = make_settings(tmp_path)
    app = create_app(settings, make_orchestrator(settings))
    async with client(app) as http:
        response = await http.post("/api/references", files={"file": ("a.wav", b"RIFF")})
    assert response.status_code == 503


async def test_generation_passes_reference_and_validates_it(tmp_path: Path) -> None:
    app, _, fake = reference_app(tmp_path)
    async with client(app) as http:
        await http.post(
            "/api/references", files={"file": ("a.wav", sine_wav(tmp_path / "a.wav", 5))}
        )
        ok = await http.post(
            "/api/generate",
            json={"prompt": "流行", "provider": "elevenlabs_music", "referenceId": "song_1"},
        )
        missing = await http.post(
            "/api/jobs",
            json={"prompt": "流行", "provider": "elevenlabs_music", "referenceId": "song_9"},
        )
        minimax = await http.post(
            "/api/jobs",
            json={"prompt": "流行", "provider": "minimax_music", "referenceId": "song_1"},
        )
        invalid = await http.post(
            "/api/jobs",
            json={"prompt": "流行", "provider": "elevenlabs_music", "referenceId": "../x"},
        )

    assert ok.status_code == 200, ok.text
    reference = fake.references[0]
    assert reference.song_id == "song_1"
    assert 4_900 <= reference.duration_ms <= 5_100
    assert reference.strength == "medium"
    assert missing.status_code == 400 and "不存在" in missing.json()["message"]
    assert minimax.status_code == 400 and "MiniMax" in minimax.json()["message"]
    assert invalid.status_code == 422


async def test_generation_without_reference_never_passes_one(tmp_path: Path) -> None:
    app, _, fake = reference_app(tmp_path)
    async with client(app) as http:
        response = await http.post(
            "/api/generate", json={"prompt": "流行", "provider": "elevenlabs_music"}
        )
    assert response.status_code == 200, response.text
    assert fake.references == [None]


async def test_upload_reports_missing_audio_tools(tmp_path: Path, monkeypatch) -> None:
    import app.services.stems as stems

    def missing():
        raise GenerationError("音频处理需要 ffmpeg 和 ffprobe")

    monkeypatch.setattr(stems, "prepare_ffmpeg_environment", missing)
    app, settings, fake = reference_app(tmp_path)
    async with client(app) as http:
        response = await http.post("/api/references", files={"file": ("a.wav", b"RIFF-data")})
    assert response.status_code == 500
    assert "ffmpeg" in response.json()["message"]
    assert fake.uploads == []
    assert not list((settings.output_dir / "references").glob("*.mp3"))


async def test_mureka_generation_receives_local_reference_id(tmp_path: Path) -> None:
    app, settings, fake = reference_app(tmp_path)
    mureka = FakeElevenLabs(settings.mock_full_song_path)
    app.state.orchestrator.music_providers["mureka_music"] = mureka
    async with client(app) as http:
        await http.post(
            "/api/references", files={"file": ("a.wav", sine_wav(tmp_path / "a.wav", 5))}
        )
        ok = await http.post(
            "/api/generate",
            json={"prompt": "流行", "provider": "mureka_music", "referenceId": "song_1"},
        )
    assert ok.status_code == 200, ok.text
    assert mureka.references[0].song_id == "song_1"
    assert fake.references == []
