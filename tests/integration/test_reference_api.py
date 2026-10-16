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
        tmp_path, music_api_mode="real", audio_reference_max_upload_mb=1, **updates
    )
    orchestrator = make_orchestrator(settings)
    fake = FakeElevenLabs(settings.mock_full_song_path)
    orchestrator.music_providers["elevenlabs_music"] = fake
    return create_app(settings, orchestrator), settings, fake


async def test_reference_library_identifies_provider_even_when_empty(tmp_path: Path) -> None:
    app, _, _ = reference_app(tmp_path)
    async with client(app) as http:
        for provider in ("mureka_music", "elevenlabs_music"):
            response = await http.get("/api/references", params={"provider": provider})
            assert response.status_code == 200
            assert response.json().get("provider") == provider
            assert response.json()["references"] == []


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
        deleted = await http.delete(f"/api/references/{uploaded.json()['id']}")
        deleted_again = await http.delete(f"/api/references/{uploaded.json()['id']}")
        after = await http.get("/api/references")

    assert uploaded.status_code == 201, uploaded.text
    body = uploaded.json()
    assert len(body["id"]) == 32
    assert body["name"] == "我的 参考.wav"
    # Trimmed locally to the documented 30 s conditioning limit.
    assert 29 <= body["durationSeconds"] <= 30
    assert fake.uploads == []
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


async def test_upload_stays_local_when_provider_is_unavailable(tmp_path: Path) -> None:
    app, settings, fake = reference_app(tmp_path)
    fake.upload_error = GenerationError("ElevenLabs 参考音频上传失败：HTTP 422: copyright")
    async with client(app) as http:
        response = await http.post(
            "/api/references", files={"file": ("a.wav", sine_wav(tmp_path / "a.wav", 4))}
        )
    assert response.status_code == 201
    assert fake.uploads == []
    assert list((settings.output_dir / "references").glob("*.mp3"))


async def test_upload_does_not_require_real_mode(tmp_path: Path) -> None:
    settings = make_settings(tmp_path)
    app = create_app(settings, make_orchestrator(settings))
    async with client(app) as http:
        response = await http.post(
            "/api/references", files={"file": ("a.wav", sine_wav(tmp_path / "a.wav", 4))}
        )
    assert response.status_code == 201


async def test_generation_passes_reference_and_validates_it(tmp_path: Path) -> None:
    app, _, fake = reference_app(tmp_path)
    async with client(app) as http:
        uploaded = await http.post(
            "/api/references", files={"file": ("a.wav", sine_wav(tmp_path / "a.wav", 5))}
        )
        reference_id = uploaded.json()["id"]
        ok = await http.post(
            "/api/generate",
            json={"prompt": "流行", "provider": "elevenlabs_music", "referenceId": reference_id},
        )
        missing = await http.post(
            "/api/jobs",
            json={"prompt": "流行", "provider": "elevenlabs_music", "referenceId": "song_9"},
        )
        minimax = await http.post(
            "/api/jobs",
            json={"prompt": "流行", "provider": "minimax_music", "referenceId": reference_id},
        )
        invalid = await http.post(
            "/api/jobs",
            json={"prompt": "流行", "provider": "elevenlabs_music", "referenceId": "../x"},
        )

    assert ok.status_code == 200, ok.text
    reference = fake.references[0]
    assert reference.song_id == reference_id
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
        uploaded = await http.post(
            "/api/references?provider=mureka_music",
            files={"file": ("a.wav", sine_wav(tmp_path / "a.wav", 5))},
        )
        reference_id = uploaded.json()["id"]
        ok = await http.post(
            "/api/generate",
            json={"prompt": "流行", "provider": "mureka_music", "referenceId": reference_id},
        )
    assert ok.status_code == 200, ok.text
    assert mureka.references[0].song_id == reference_id
    assert fake.references == []


async def test_reference_libraries_and_generation_are_isolated_by_provider(tmp_path: Path) -> None:
    app, settings, _ = reference_app(tmp_path)
    audio = sine_wav(tmp_path / "a.wav", 5)
    async with client(app) as http:
        references = {}
        for provider in ("mureka_music", "elevenlabs_music"):
            uploaded = await http.post(
                f"/api/references?provider={provider}",
                files={"file": ("a.wav", audio)},
            )
            assert uploaded.status_code == 201, uploaded.text
            references[provider] = uploaded.json()
            assert uploaded.json()["provider"] == provider

        for provider, reference in references.items():
            listed = await http.get(f"/api/references?provider={provider}")
            assert listed.json()["references"] == [reference]
            other = next(value for key, value in references.items() if key != provider)
            generation = await http.post(
                "/api/jobs",
                json={"prompt": "pop", "provider": provider, "referenceId": other["id"]},
            )
            assert generation.status_code == 400
            assert "不属于当前模型" in generation.json()["message"]
            deleted = await http.delete(f"/api/references/{other['id']}?provider={provider}")
            assert deleted.status_code == 404

        for provider, reference in references.items():
            deleted = await http.delete(f"/api/references/{reference['id']}?provider={provider}")
            assert deleted.status_code == 204

    assert not list((settings.output_dir / "references").glob("*.mp3"))


async def test_unassigned_legacy_reference_is_preserved_but_cannot_cross_providers(
    tmp_path: Path,
) -> None:
    from app.services.references import load_references, new_reference_entry, save_references

    app, settings, _ = reference_app(tmp_path)
    legacy = new_reference_entry("legacy", "legacy.wav", 5, "legacy.mp3")
    save_references(settings.output_dir, [legacy])
    async with client(app) as http:
        for provider in ("mureka_music", "elevenlabs_music"):
            listed = await http.get(f"/api/references?provider={provider}")
            assert listed.json()["references"] == []
            response = await http.post(
                "/api/jobs",
                json={"prompt": "pop", "provider": provider, "referenceId": "legacy"},
            )
            assert response.status_code == 400
            assert "重新上传" in response.json()["message"]
    assert load_references(settings.output_dir) == [legacy]


@pytest.mark.parametrize("provider", ["minimax_music", "invalid", "mureka_music&other=true"])
async def test_reference_operations_reject_invalid_provider(tmp_path: Path, provider: str) -> None:
    app, _, _ = reference_app(tmp_path)
    async with client(app) as http:
        query = {"provider": provider}
        assert (await http.get("/api/references", params=query)).status_code == 422
        assert (
            await http.post(
                "/api/references",
                params=query,
                files={"file": ("a.mp3", b"fake")},
            )
        ).status_code == 422
        assert (await http.delete("/api/references/ref", params=query)).status_code == 422
