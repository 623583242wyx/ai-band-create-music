from __future__ import annotations

import json
import wave
from pathlib import Path

import httpx
import pytest

from app.core.errors import GenerationError, ProviderGlobalError
from app.services.providers import AudioReference, ElevenLabsMusicProvider
from app.services.references import (
    clean_reference_name,
    new_reference_entry,
    references_dir,
    save_references,
)
from tests.helpers import make_settings

PLAN = {
    "chunks": [
        {"text": "[Verse]\n第一句", "duration_ms": 2_000, "positive_styles": ["pop"]},
        {"text": "[Chorus]\n第二句", "duration_ms": 1_000, "positive_styles": ["pop"]},
    ]
}
REFERENCE = AudioReference("song_abc", 30_000, "high")


def test_reference_filename_never_keeps_a_client_path() -> None:
    assert clean_reference_name(r"..\\..\\evil.mp3") == "evil.mp3"


def eleven_settings(tmp_path: Path, **updates: object):
    return make_settings(
        tmp_path,
        music_api_mode="real",
        music_provider="elevenlabs_music",
        elevenlabs_api_key="secret",
        elevenlabs_music_base_url="https://eleven.test",
        elevenlabs_music_output_format="pcm_8000",
        **updates,
    )


async def test_reference_conditions_first_chunk_of_free_plan(tmp_path: Path) -> None:
    requests: list[httpx.Request] = []
    pcm = b"\x00\x00\x01\x00" * 8_000 * 3

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        if request.url.path == "/v1/music/plan":
            return httpx.Response(200, json=PLAN)
        return httpx.Response(200, content=pcm)

    settings = eleven_settings(tmp_path)
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        result = await ElevenLabsMusicProvider(settings, client).generate(
            "[Genre: Pop]",
            None,
            "[歌词与创作内容]\n第一句\n第二句",
            job_id="job-1",
            reference=REFERENCE,
        )

    assert [request.url.path for request in requests] == ["/v1/music/plan", "/v1/music/stream"]
    plan_body = json.loads(requests[0].content)
    assert plan_body["model_id"] == "music_v2_5"
    assert "第一句\n第二句" in plan_body["prompt"]
    assert "music_length_ms" not in plan_body
    music_body = json.loads(requests[1].content)
    assert set(music_body) == {"composition_plan", "model_id"}
    assert music_body["model_id"] == "music_v2_5"
    first, second = music_body["composition_plan"]["chunks"]
    assert first["conditioning_ref"] == {
        "song_id": "song_abc",
        "range": {"start_ms": 0, "end_ms": 30_000},
    }
    assert first["condition_strength"] == "high"
    assert "conditioning_ref" not in second
    assert result.debug["mode"] == "reference_plan_pcm_wav"
    assert result.debug["audioReferenceId"] == "song_abc"
    with wave.open(str(result.audio_path), "rb") as audio:
        assert audio.getnframes() == 8_000 * 3


async def test_reference_plan_receives_fixed_length(tmp_path: Path) -> None:
    bodies = []

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/v1/music/plan":
            bodies.append(json.loads(request.content))
            return httpx.Response(200, json=PLAN)
        return httpx.Response(200, content=b"\x00\x00\x01\x00" * 8_000 * 3)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        await ElevenLabsMusicProvider(eleven_settings(tmp_path), client).generate(
            "[Genre: Pop]", 3, "[歌词与创作内容]\n第一句", reference=REFERENCE
        )
    assert bodies[0]["music_length_ms"] == 3_000


@pytest.mark.parametrize(
    "plan",
    [{}, {"chunks": []}, {"chunks": [{"song_id": "x"}]}, {"chunks": [{"text": "a"}]}, []],
)
async def test_invalid_plan_never_starts_billed_generation(tmp_path: Path, plan) -> None:
    paths = []

    def handler(request: httpx.Request) -> httpx.Response:
        paths.append(request.url.path)
        return httpx.Response(200, json=plan)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        with pytest.raises(GenerationError, match="composition plan"):
            await ElevenLabsMusicProvider(eleven_settings(tmp_path), client).generate(
                "[Genre: Pop]", None, "[歌词与创作内容]\n第一句", reference=REFERENCE
            )
    assert paths == ["/v1/music/plan"]


async def test_plan_rate_limit_is_provider_global(tmp_path: Path) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(429, json={"detail": {"message": "rate limited"}})

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        with pytest.raises(ProviderGlobalError, match="429"):
            await ElevenLabsMusicProvider(eleven_settings(tmp_path), client).generate(
                "[Genre: Pop]", None, "[歌词与创作内容]\n第一句", reference=REFERENCE
            )


@pytest.mark.parametrize(
    ("updates", "message"),
    [
        ({"elevenlabs_force_instrumental": True}, "FORCE_INSTRUMENTAL"),
        ({"elevenlabs_music_model_id": "music_v1"}, "music_v2"),
    ],
)
async def test_reference_rejects_unsupported_settings_before_any_request(
    tmp_path: Path, updates, message
) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise AssertionError("no request expected")

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        with pytest.raises(GenerationError, match=message):
            await ElevenLabsMusicProvider(eleven_settings(tmp_path, **updates), client).generate(
                "[Genre: Pop]", None, "[歌词与创作内容]\n第一句", reference=REFERENCE
            )


async def test_upload_reference_returns_song_id(tmp_path: Path) -> None:
    requests = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(200, json={"song_id": "vVtPM1Sas70E2LIhQFch"})

    clip = tmp_path / "clip.mp3"
    clip.write_bytes(b"ID3-clip")
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        song_id = await ElevenLabsMusicProvider(eleven_settings(tmp_path), client).upload_reference(
            clip, "我的参考.mp3"
        )

    assert song_id == "vVtPM1Sas70E2LIhQFch"
    assert requests[0].url == "https://eleven.test/v1/music/upload"
    assert requests[0].headers["xi-api-key"] == "secret"
    assert requests[0].headers["content-type"].startswith("multipart/form-data")
    assert b"ID3-clip" in requests[0].content


async def test_local_reference_is_uploaded_only_when_generated(tmp_path: Path) -> None:
    paths: list[str] = []
    directory = references_dir(tmp_path / "output")
    directory.mkdir(parents=True)
    clip = directory / "local.mp3"
    clip.write_bytes(b"ID3-clip")
    entry = new_reference_entry("local_ref", "reference.mp3", 30, clip.name)
    entry["localOnly"] = True
    save_references(tmp_path / "output", [entry])

    def handler(request: httpx.Request) -> httpx.Response:
        paths.append(request.url.path)
        if request.url.path == "/v1/music/upload":
            return httpx.Response(200, json={"song_id": "remote_ref"})
        if request.url.path == "/v1/music/plan":
            return httpx.Response(200, json=PLAN)
        return httpx.Response(200, content=b"\x00\x00\x01\x00" * 8_000 * 3)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        await ElevenLabsMusicProvider(eleven_settings(tmp_path), client).generate(
            "[Genre: Pop]",
            None,
            "[歌词与创作内容]\n第一句",
            reference=AudioReference("local_ref", 30_000, "high"),
        )

    assert paths == ["/v1/music/upload", "/v1/music/plan", "/v1/music/stream"]


@pytest.mark.parametrize(
    ("response", "error"),
    [
        (httpx.Response(200, json={}), GenerationError),
        (httpx.Response(200, json={"song_id": "../evil"}), GenerationError),
        (httpx.Response(422, json={"detail": "copyright"}), GenerationError),
        (httpx.Response(401, json={"detail": "bad key"}), ProviderGlobalError),
    ],
)
async def test_upload_reference_failures(tmp_path: Path, response, error) -> None:
    clip = tmp_path / "clip.mp3"
    clip.write_bytes(b"ID3-clip")
    async with httpx.AsyncClient(transport=httpx.MockTransport(lambda _: response)) as client:
        with pytest.raises(error, match="参考音频上传失败"):
            await ElevenLabsMusicProvider(eleven_settings(tmp_path), client).upload_reference(
                clip, "clip.mp3"
            )


@pytest.mark.parametrize(
    ("status", "message", "is_global"),
    [
        (401, "鉴权或权限不足", True),
        (413, "超过服务限制", False),
        (422, "格式或时长不被接受", False),
        (500, "服务暂时不可用", False),
    ],
)
async def test_elevenlabs_reference_upload_reports_specific_error(
    tmp_path: Path, status: int, message: str, is_global: bool
) -> None:
    clip = tmp_path / "clip.mp3"
    clip.write_bytes(b"ID3-clip")

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/v1/music/upload"
        return httpx.Response(status, json={"detail": "provider detail"})

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        with pytest.raises(GenerationError, match=message) as caught:
            await ElevenLabsMusicProvider(eleven_settings(tmp_path), client).upload_reference(
                clip, "clip.mp3"
            )

    assert isinstance(caught.value, ProviderGlobalError) is is_global
    assert "Mureka" not in str(caught.value)
