from __future__ import annotations

import json
import wave
from pathlib import Path

import httpx
import pytest

from app.core.errors import GenerationError, ProviderGlobalError
from app.services.providers import AudioReference, ElevenLabsMusicProvider
from tests.helpers import make_settings

PLAN = {
    "chunks": [
        {"text": "[Verse]\n第一句", "duration_ms": 2_000, "positive_styles": ["pop"]},
        {"text": "[Chorus]\n第二句", "duration_ms": 1_000, "positive_styles": ["pop"]},
    ]
}
REFERENCE = AudioReference("song_abc", 30_000, "high")


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
    assert plan_body["model_id"] == "music_v2"
    assert "第一句\n第二句" in plan_body["prompt"]
    assert "music_length_ms" not in plan_body
    music_body = json.loads(requests[1].content)
    assert set(music_body) == {"composition_plan", "model_id"}
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
