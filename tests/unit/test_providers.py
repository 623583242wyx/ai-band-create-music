from __future__ import annotations

import asyncio
import io
import json
import time
import wave
from pathlib import Path

import httpx
import pytest

from app.core.errors import GenerationError, ProviderGlobalError
from app.services.providers import (
    AudioReference,
    ElevenLabsMusicProvider,
    GenericMusicProvider,
    MiniMaxMusicProvider,
    MurekaMusicProvider,
    _detailed_music_text,
    _wrap_long_lyric_lines,
    create_music_provider,
    extract_generated_audio_url,
    extract_suno_audio_url,
    extract_task_id,
)
from app.services.references import new_reference_entry, references_dir, save_references
from tests.helpers import make_settings


class StreamingErrorBody(httpx.AsyncByteStream):
    async def __aiter__(self):
        yield b'{"detail":{"message":"unauthorized"}}'


class ChunkedBody(httpx.AsyncByteStream):
    def __init__(self, *chunks: bytes) -> None:
        self.chunks = chunks

    async def __aiter__(self):
        for chunk in self.chunks:
            yield chunk


class Clock:
    now = 0

    def monotonic(self):
        return self.now

    def __getattr__(self, name):
        return getattr(time, name)


MINIMAX_USER_PROMPT = "[歌词与创作内容]\n[Verse]\ntest lyrics\n\n[风格要求]\nrock"


def wav_bytes(seconds: float = 1) -> bytes:
    output = io.BytesIO()
    with wave.open(output, "wb") as audio:
        audio.setparams((2, 2, 8000, 0, "NONE", "not compressed"))
        audio.writeframes(b"\0" * round(seconds * 8000 * 4))
    return output.getvalue()


def test_provider_response_extractors_cover_legacy_shapes() -> None:
    assert extract_task_id({"data": {"task_id": "task-1"}}) == "task-1"
    assert (
        extract_generated_audio_url({"data": [{"audio_url": "https://audio"}]}) == "https://audio"
    )
    assert (
        extract_suno_audio_url([{"id": "clip", "status": "streaming", "audio_url": "https://suno"}])
        == "https://suno"
    )


def test_minimax_wraps_long_lyrics_without_empty_lines_or_lost_text() -> None:
    samples = (
        "短句一，" + "啊" * 40 + "。",
        "啊" * 80,
        "Mr. Smith walked down the long and winding road to nowhere at all tonight.",
    )
    for lyrics in samples:
        wrapped = _wrap_long_lyric_lines(lyrics)
        assert not wrapped.endswith("\n")
        assert max(map(len, wrapped.splitlines())) <= 32
        assert wrapped.replace("\n", "") == lyrics


async def test_request_provider_overrides_configured_default(tmp_path: Path) -> None:
    settings = make_settings(
        tmp_path,
        music_api_mode="real",
        music_provider="elevenlabs_music",
    )
    async with httpx.AsyncClient() as client:
        provider = create_music_provider(settings, client, provider="minimax_music")
        assert isinstance(provider, MiniMaxMusicProvider)


async def test_minimax_provider_uses_direct_client(tmp_path: Path) -> None:
    proxied_requests: list[httpx.Request] = []
    direct_requests: list[httpx.Request] = []

    def proxied_handler(request: httpx.Request) -> httpx.Response:
        proxied_requests.append(request)
        raise AssertionError("MiniMax must not use the proxy-aware client")

    def direct_handler(request: httpx.Request) -> httpx.Response:
        direct_requests.append(request)
        return minimax_response(request)

    settings = make_settings(tmp_path, music_api_mode="real")
    async with (
        httpx.AsyncClient(transport=httpx.MockTransport(proxied_handler)) as client,
        httpx.AsyncClient(transport=httpx.MockTransport(direct_handler)) as direct_client,
    ):
        provider = create_music_provider(
            settings,
            client,
            direct_client,
            provider="minimax_music",
        )
        await provider.generate(
            "[Genre: Rock]",
            60,
            "[歌词与创作内容]\n[Verse]\ntest lyrics",
        )

    assert not proxied_requests
    assert direct_requests[0].url == "http://127.0.0.1:8111/v1/audio/jobs"


async def test_elevenlabs_sends_complete_prompt_and_streams_audio(tmp_path: Path) -> None:
    requests: list[httpx.Request] = []
    pcm = b"\x00\x00\x01\x00" * 24_000

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(
            200,
            stream=ChunkedBody(pcm[:3], pcm[3:1001], pcm[1001:]),
        )

    settings = make_settings(
        tmp_path,
        music_api_mode="real",
        music_provider="elevenlabs_music",
        elevenlabs_api_key="secret",
        elevenlabs_music_base_url="https://eleven.test",
        elevenlabs_music_output_format="pcm_8000",
    )
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        result = await ElevenLabsMusicProvider(settings, client).generate(
            "[Genre: Rock]",
            3,
            "[歌词与创作内容]\n第一句原歌词\n第二句原歌词\n\n[风格要求]\n普通话摇滚",
            job_id="job-1",
        )

    with wave.open(str(result.audio_path), "rb") as audio:
        assert audio.getparams()[:3] == (2, 2, 8_000)
        assert audio.readframes(audio.getnframes()) == pcm
    assert result.audio_path == settings.output_dir / "jobs/job-1/song_1/full_song_job-1_1.wav"
    assert not list(settings.output_dir.glob("full_song_*"))
    assert result.debug["mode"] == "prompt_pcm_wav"
    assert result.debug["durationSeconds"] == 3.0
    assert len(requests) == 1
    assert requests[0].url.path == "/v1/music/stream"
    assert requests[0].url.params["output_format"] == "pcm_8000"
    music_body = json.loads(requests[0].content)
    assert music_body["model_id"] == "music_v2_5"
    assert "[Genre: Rock]" in music_body["prompt"]
    assert "第一句原歌词\n第二句原歌词" in music_body["prompt"]
    assert "Mandarin Chinese lead vocals" in music_body["prompt"]
    assert music_body["music_length_ms"] == 3_000
    assert "composition_plan" not in music_body


def test_elevenlabs_section_only_plan_is_not_lyrics() -> None:
    metadata = {
        "song_metadata": {"title": "After the Rain"},
        "composition_plan": {
            "chunks": [
                {"text": "[Intro]"},
                {"text": "[Build]"},
                {"text": "{filtered rain texture}"},
            ]
        },
    }
    assert _detailed_music_text(metadata) == ("After the Rain", None)


async def test_elevenlabs_direct_prompt_reads_detailed_metadata(tmp_path: Path) -> None:
    requests: list[httpx.Request] = []
    pcm = b"\x00\x00\x01\x00" * 24_000
    boundary = "music-boundary"
    metadata = {
        "song_metadata": {"title": "雨夜"},
        "composition_plan": {"chunks": [{"text": "[Verse]\n第一句"}, {"text": "[Chorus]\n第二句"}]},
    }
    body = (
        f"--{boundary}\r\nContent-Type: application/json\r\n\r\n"
        f"{json.dumps(metadata, ensure_ascii=False)}\r\n"
        f"--{boundary}\r\nContent-Type: audio/pcm\r\n\r\n"
    ).encode() + pcm + f"\r\n--{boundary}--\r\n".encode()

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(
            200,
            stream=ChunkedBody(body[:37], body[37:51], body[51:200], body[200:]),
            headers={"content-type": f"multipart/mixed; boundary={boundary}"},
        )

    settings = make_settings(
        tmp_path,
        elevenlabs_api_key="secret",
        elevenlabs_music_base_url="https://eleven.test",
        elevenlabs_music_output_format="pcm_8000",
    )
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        result = await ElevenLabsMusicProvider(settings, client).generate(
            "[Genre: Folk]", 3, "温暖民谣", direct_prompt=True, job_id="direct-job"
        )

    assert requests[0].url.path == "/v1/music/detailed"
    assert "Write and sing original lyrics with a clear lead vocal" in json.loads(
        requests[0].content
    )["prompt"]
    assert result.title == "雨夜"
    assert result.lyrics == "[Verse]\n第一句\n[Chorus]\n第二句"
    with wave.open(str(result.audio_path), "rb") as audio:
        assert audio.readframes(audio.getnframes()) == pcm


@pytest.mark.parametrize("duration_seconds", [3, None])
async def test_elevenlabs_progress_throttles_and_flushes_final_bytes(
    tmp_path, monkeypatch, duration_seconds
):
    reports = []

    clock = Clock()
    monkeypatch.setattr("app.services.providers.time", clock)

    async def progress(stage, step, total, **values):
        assert step is total is None
        reports.append((stage, values))

    pcm = b"\0" * 32_000

    async def chunks():
        for now in (0, 0.5, 1, 1.5):
            clock.now = now
            yield pcm

    target = tmp_path / "stream.wav"
    duration = await ElevenLabsMusicProvider._write_pcm_wav(
        chunks(),
        target,
        8_000,
        duration_seconds,
        progress=progress,
    )
    assert duration == 4
    assert reports == [
        (
            "receiving_audio",
            {
                "received_audio_seconds": seconds,
                "expected_audio_seconds": duration_seconds,
            },
        )
        for seconds in (1, 3, 4)
    ]
    with wave.open(str(target), "rb") as audio:
        assert audio.readframes(audio.getnframes()) == pcm * 4


async def test_elevenlabs_cancellation_during_final_report_preserves_complete_audio(
    tmp_path, monkeypatch
):
    monkeypatch.setattr("app.services.providers.time", Clock())
    target = tmp_path / "cancelled.wav"
    reports = 0

    async def progress(*args, **kwargs):
        nonlocal reports
        reports += 1
        if reports == 2:
            raise asyncio.CancelledError

    with pytest.raises(asyncio.CancelledError):
        await ElevenLabsMusicProvider._write_pcm_wav(
            ChunkedBody(b"\0" * 32_000, b"\0" * 64_000).__aiter__(),
            target,
            8_000,
            3,
            progress=progress,
        )
    with wave.open(str(target), "rb") as audio:
        assert audio.getnframes() == 24_000
    assert not list(tmp_path.glob("*.part"))


@pytest.mark.parametrize("status", [402, 403, 404, 405])
@pytest.mark.parametrize("duration", [3, None])
async def test_elevenlabs_rejected_stream_falls_back_once_to_compose(
    tmp_path, monkeypatch, caplog, status, duration
):
    monkeypatch.setattr("app.services.providers.time", Clock())
    requests = []
    reports = []
    pcm = b"\0" * 96_000

    async def progress(stage, step, total, **values):
        assert step is total is None
        reports.append((stage, values))

    def handler(request):
        requests.append(request)
        if request.url.path.endswith("/stream"):
            return httpx.Response(
                status, json={"detail": "stream unavailable"}, headers={"x-request-id": "rejected"}
            )
        return httpx.Response(200, stream=ChunkedBody(pcm[:32_000], pcm[32_000:]))

    settings = make_settings(
        tmp_path, elevenlabs_api_key="test", elevenlabs_music_output_format="pcm_8000"
    )
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        result = await ElevenLabsMusicProvider(settings, client).generate(
            "rock", duration, "", job_id="fallback", progress=progress
        )
    assert [request.url.path for request in requests] == ["/v1/music/stream", "/v1/music"]
    assert requests[0].content == requests[1].content
    assert json.loads(requests[1].content)["model_id"] == "music_v2_5"
    assert requests[0].url.params == requests[1].url.params
    assert requests[0].headers["xi-api-key"] == requests[1].headers["xi-api-key"]
    assert result.debug["endpoint"].endswith("/v1/music")
    assert [stage for stage, _ in reports] == [
        "generating_music",
        "receiving_audio",
        "receiving_audio",
    ]
    assert [values["received_audio_seconds"] for _, values in reports] == [None, 1, 3]
    assert all(values["expected_audio_seconds"] == duration for _, values in reports)
    assert "falling back to compose" in caplog.text
    assert f"status={status} job_id=fallback" in caplog.text
    diagnostics = json.loads((settings.output_dir / "jobs/fallback/prompts.json").read_text())
    assert diagnostics["providerRequests"][0]["streamFallbackStatus"] == status
    rejection = diagnostics["providerRequests"][0]["streamFallbackResponse"]
    assert rejection["statusCode"] == status
    assert rejection["headers"]["x-request-id"] == "rejected"
    assert rejection["body"] == {"detail": "stream unavailable"}
    assert diagnostics["providerRequests"][0]["request"]["url"].endswith("/v1/music")
    with wave.open(str(result.audio_path), "rb") as audio:
        assert audio.readframes(audio.getnframes()) == pcm


@pytest.mark.parametrize("status", [401, 422, 429, 500])
async def test_elevenlabs_other_errors_do_not_retry(tmp_path, status):
    requests = []

    def handler(request):
        requests.append(request)
        return httpx.Response(status, json={"detail": "rejected"})

    settings = make_settings(tmp_path, elevenlabs_api_key="test")
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        with pytest.raises(GenerationError):
            await ElevenLabsMusicProvider(settings, client).generate("rock", 3, "")
    assert len(requests) == 1


async def test_elevenlabs_compose_fallback_failure_does_not_retry(tmp_path):
    requests = []

    def handler(request):
        requests.append(request)
        return httpx.Response(403, json={"detail": "rejected"})

    settings = make_settings(tmp_path, elevenlabs_api_key="test")
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        with pytest.raises(GenerationError, match="Music API 权限"):
            await ElevenLabsMusicProvider(settings, client).generate("rock", 3, "")
    assert len(requests) == 2


@pytest.mark.parametrize("provider_class", [ElevenLabsMusicProvider, MiniMaxMusicProvider])
@pytest.mark.parametrize("status", [401, 402, 403, 429, 422, 500])
async def test_providers_classify_global_http_errors(tmp_path, provider_class, status):
    requests = []

    def handler(request):
        requests.append(request)
        return httpx.Response(status, json={"detail": "rejected"})

    settings = make_settings(tmp_path, elevenlabs_api_key="test")
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        with pytest.raises(GenerationError) as caught:
            await provider_class(settings, client).generate("rock", 3, MINIMAX_USER_PROMPT)
    assert isinstance(caught.value, ProviderGlobalError) == (status in {401, 402, 403, 429})
    assert len(requests) == (
        2 if provider_class is ElevenLabsMusicProvider and status in {402, 403} else 1
    )


async def test_elevenlabs_progress_failure_never_discards_audio(tmp_path, caplog, monkeypatch):
    monkeypatch.setattr("app.services.providers.time", Clock())
    pcm = b"\0" * 96_000
    stages = []

    async def progress(stage, *args, **kwargs):
        stages.append(stage)
        raise OSError("metadata write failed")

    settings = make_settings(
        tmp_path, elevenlabs_api_key="test", elevenlabs_music_output_format="pcm_8000"
    )
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(
            lambda request: httpx.Response(200, stream=ChunkedBody(pcm[:32_000], pcm[32_000:]))
        )
    ) as client:
        result = await ElevenLabsMusicProvider(settings, client).generate(
            "rock", 3, "", progress=progress
        )
    assert stages == ["generating_music", "receiving_audio", "receiving_audio"]
    assert "progress report failed" in caplog.text
    with wave.open(str(result.audio_path), "rb") as audio:
        assert audio.readframes(audio.getnframes()) == pcm
    assert not list(settings.output_dir.rglob("*.part"))


@pytest.mark.parametrize("duration", [0, -1])
async def test_elevenlabs_rejects_nonpositive_duration_before_request(tmp_path, duration):
    settings = make_settings(tmp_path, elevenlabs_api_key="test")

    def handler(request):
        raise AssertionError("invalid duration must not be sent upstream")

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        with pytest.raises(GenerationError, match="目标时长必须大于 0"):
            await ElevenLabsMusicProvider(settings, client).generate("rock", duration, "")
    with pytest.raises(GenerationError, match="目标时长必须大于 0"):
        await ElevenLabsMusicProvider._write_pcm_wav(
            ChunkedBody(b"\0" * 96_000).__aiter__(), tmp_path / "invalid.wav", 8000, duration
        )
    assert not list(tmp_path.rglob("*.part"))


async def test_elevenlabs_auto_omits_fixed_duration(tmp_path: Path) -> None:
    requests: list[httpx.Request] = []
    pcm = b"\0" * (8_000 * 3 * 4)

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(200, content=pcm)

    settings = make_settings(
        tmp_path,
        elevenlabs_api_key="secret",
        elevenlabs_music_base_url="https://eleven.test",
        elevenlabs_music_output_format="pcm_8000",
    )
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        result = await ElevenLabsMusicProvider(settings, client).generate(
            "[Genre: Rock]", None, MINIMAX_USER_PROMPT, job_id="auto-job"
        )

    body = json.loads(requests[0].content)
    assert "music_length_ms" not in body
    assert "Target duration:" not in body["prompt"]
    assert "[Verse]\ntest lyrics" in body["prompt"]
    assert "Mandarin Chinese lead vocals" in body["prompt"]
    assert result.debug["durationSeconds"] == 3.0
    with wave.open(str(result.audio_path), "rb") as audio:
        assert audio.readframes(audio.getnframes()) == pcm
    diagnostics = json.loads((settings.output_dir / "jobs/auto-job/prompts.json").read_text())
    assert diagnostics["providerRequests"][0]["request"]["body"] == body


async def test_elevenlabs_variations_preserve_the_same_prompt(tmp_path: Path) -> None:
    requests: list[httpx.Request] = []
    pcm = b"\0" * (8_000 * 3 * 4)

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(200, content=pcm)

    settings = make_settings(
        tmp_path,
        music_api_mode="real",
        elevenlabs_api_key="secret",
        elevenlabs_music_base_url="https://eleven.test",
        elevenlabs_music_output_format="pcm_8000",
    )
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        provider = ElevenLabsMusicProvider(settings, client)
        await provider.generate("[Genre: Rock]", 3, "", variation=0)
        await provider.generate("[Genre: Rock]", 3, "", variation=1)

    bodies = [json.loads(request.content) for request in requests]
    assert bodies[0] == bodies[1]
    assert all("seed" not in body for body in bodies)


async def test_elevenlabs_rejects_oversized_complete_prompt_before_request(tmp_path: Path) -> None:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(200, content=b"unexpected")

    settings = make_settings(
        tmp_path,
        music_api_mode="real",
        music_provider="elevenlabs_music",
        elevenlabs_api_key="secret",
        elevenlabs_music_base_url="https://eleven.test",
    )
    lyrics = "[Verse]\n" + "很长的完整歌词" * 700
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        with pytest.raises(GenerationError, match="超过接口上限 4100"):
            await ElevenLabsMusicProvider(settings, client).generate(
                "[Genre: Rock]",
                120,
                f"[歌词与创作内容]\n{lyrics}",
                job_id="oversized-job",
            )

    assert not requests
    diagnostics = json.loads(
        (settings.output_dir / "jobs/oversized-job/prompts.json").read_text(encoding="utf-8")
    )
    provider_request = diagnostics["providerRequests"][0]
    assert provider_request["request"]["body"]["prompt"].endswith(lyrics)
    assert "超过接口上限 4100" in provider_request["validationError"]


async def test_elevenlabs_instrumental_mode_omits_chinese_vocal_direction(tmp_path: Path) -> None:
    requests: list[httpx.Request] = []
    pcm = b"\0" * (8_000 * 3 * 4)

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(200, content=pcm)

    settings = make_settings(
        tmp_path,
        music_api_mode="real",
        music_provider="elevenlabs_music",
        elevenlabs_api_key="secret",
        elevenlabs_music_base_url="https://eleven.test",
        elevenlabs_music_output_format="pcm_8000",
        elevenlabs_force_instrumental=True,
    )
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        result = await ElevenLabsMusicProvider(settings, client).generate(
            "[Genre: dark techno]", 3, ""
        )

    body = json.loads(requests[0].content)
    assert body["force_instrumental"] is True
    assert "Mandarin Chinese lead vocals" not in body["prompt"]
    assert result.debug["clearChineseVocalMode"] is False


@pytest.mark.parametrize("duration_seconds", [3, None])
@pytest.mark.parametrize(("pcm", "message"), [(b"", "返回空音频"), (b"\0", "不完整的 PCM 音频帧")])
async def test_elevenlabs_invalid_pcm_removes_temporary_file(
    tmp_path: Path, duration_seconds: int | None, pcm: bytes, message: str
) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=pcm)

    settings = make_settings(
        tmp_path,
        music_api_mode="real",
        elevenlabs_api_key="secret",
        elevenlabs_music_base_url="https://eleven.test",
        elevenlabs_music_output_format="pcm_8000",
    )
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        with pytest.raises(GenerationError, match=message):
            await ElevenLabsMusicProvider(settings, client).generate(
                "[Genre: Rock]", duration_seconds, "", job_id="empty-job"
            )

    song_dir = settings.output_dir / "jobs/empty-job/song_1"
    assert not list(song_dir.glob("*.part"))
    assert not list(song_dir.glob("*.wav"))


@pytest.mark.parametrize("frame_count", [1, 8_000 * 3 - 1])
async def test_elevenlabs_auto_rejects_complete_but_short_pcm(
    tmp_path: Path, frame_count: int
) -> None:
    target = tmp_path / "short.wav"
    chunks = ChunkedBody(b"\0" * (frame_count * 4))
    with pytest.raises(GenerationError, match="自动时长音频过短"):
        await ElevenLabsMusicProvider._write_pcm_wav(chunks.__aiter__(), target, 8_000, None)
    assert not target.exists()
    assert not list(tmp_path.glob("*.part"))


async def test_elevenlabs_accepts_duration_variance_and_sample_rate(tmp_path: Path) -> None:
    pcm = b"\0" * (16_000 * 9 * 2)

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=pcm)

    settings = make_settings(
        tmp_path,
        music_api_mode="real",
        elevenlabs_api_key="secret",
        elevenlabs_music_base_url="https://eleven.test",
        elevenlabs_music_output_format="pcm_16000",
    )
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        result = await ElevenLabsMusicProvider(settings, client).generate("[Genre: Rock]", 3, "")

    with wave.open(str(result.audio_path), "rb") as audio:
        assert audio.getparams()[:3] == (2, 2, 16_000)
    assert result.debug["durationSeconds"] == 4.5


async def test_elevenlabs_rejects_unexpected_pcm_duration(tmp_path: Path) -> None:
    pcm = b"\0" * (8_000 * 3 * 2)

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=pcm)

    settings = make_settings(
        tmp_path,
        music_api_mode="real",
        elevenlabs_api_key="secret",
        elevenlabs_music_base_url="https://eleven.test",
        elevenlabs_music_output_format="pcm_8000",
    )
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        with pytest.raises(GenerationError, match="PCM 时长异常"):
            await ElevenLabsMusicProvider(settings, client).generate(
                "[Genre: Rock]", 3, "", job_id="truncated-job"
            )

    song_dir = settings.output_dir / "jobs/truncated-job/song_1"
    assert not list(song_dir.glob("*.part"))
    assert not list(song_dir.glob("*.wav"))


async def test_elevenlabs_reads_streaming_error_before_building_message(tmp_path: Path) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(401, stream=StreamingErrorBody())

    settings = make_settings(
        tmp_path,
        music_api_mode="real",
        music_provider="elevenlabs_music",
        elevenlabs_api_key="invalid",
        elevenlabs_music_base_url="https://eleven.test",
    )
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        with pytest.raises(GenerationError, match="music_generation 权限"):
            await ElevenLabsMusicProvider(settings, client).generate("[Genre: Rock]", 120, "rock")


async def test_generic_provider_accepts_direct_audio_url(tmp_path: Path) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "POST":
            return httpx.Response(200, json={"audioUrl": "https://provider.test/song.mp3"})
        return httpx.Response(200, content=b"ID3-generic")

    settings = make_settings(
        tmp_path,
        music_api_mode="real",
        music_provider="generic",
        music_api_key="secret",
        music_api_base_url="https://provider.test",
    )
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        result = await GenericMusicProvider(settings, client).generate(
            "[Genre: Folk]", 60, "folk", job_id="job-1"
        )
    assert result.audio_path.read_bytes() == b"ID3-generic"
    assert result.audio_path == settings.output_dir / "jobs/job-1/song_1/full_song_job-1_1.mp3"
    assert result.debug["mode"] == "direct_audio_url"


async def test_minimax_provider_requires_prepared_lyrics(tmp_path: Path) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        pytest.fail("MiniMax must not be called without prepared lyrics")

    settings = make_settings(tmp_path, minimax_base_url="https://minimax.test")
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        with pytest.raises(GenerationError, match="缺少歌词"):
            await MiniMaxMusicProvider(settings, client).generate("rock", None, "rock")


async def test_minimax_provider_omits_duration_for_auto(tmp_path: Path) -> None:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return minimax_response(request)

    settings = make_settings(
        tmp_path,
        music_api_mode="real",
        music_provider="minimax_music",
        minimax_base_url="https://minimax.test",
    )
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        result = await MiniMaxMusicProvider(settings, client).generate(
            "[Genre: Rock]", None, MINIMAX_USER_PROMPT
        )

    body = json.loads(requests[0].content)
    assert requests[0].url.path == "/v1/audio/jobs"
    assert "authorization" not in requests[0].headers
    assert body["model"] == "MiniMaxAI/MiniMax-Music3"
    assert "Choose the natural complete song duration" in body["instructions"]
    assert "Sing every supplied non-tag lyric line exactly once" in body["instructions"]
    assert "Never fill unused time with repeated or invented vocals" in body["instructions"]
    assert body["input"] == "[Verse]\ntest lyrics"
    assert body["seed"] == 42
    assert body["num_inference_steps"] == 30
    assert body["response_format"] == "wav"
    assert body["stream"] is False
    assert "audio_duration" not in body
    assert "auto_duration_hint" not in body
    assert "max_new_tokens" not in body
    assert result.audio_path.suffix == ".wav"
    assert result.audio_path.read_bytes()[:4] == b"RIFF"
    assert result.debug["durationSeconds"] == 1
    assert "requestedDurationSeconds" not in result.debug
    assert result.debug["mode"] == "self_hosted_wav"


async def test_minimax_provider_ignores_selected_duration(tmp_path: Path) -> None:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return minimax_response(request)

    settings = make_settings(tmp_path, minimax_base_url="https://minimax.test")
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        result = await MiniMaxMusicProvider(settings, client).generate(
            "[Genre: Rock]", 60, MINIMAX_USER_PROMPT, job_id="local-job"
        )

    body = json.loads(requests[0].content)
    assert "audio_duration" not in body
    assert "Choose the natural complete song duration" in body["instructions"]
    assert result.audio_path == (
        settings.output_dir / "jobs/local-job/song_1/full_song_local-job_1.wav"
    )
    diagnostics = json.loads(
        (settings.output_dir / "jobs/local-job/prompts.json").read_text(encoding="utf-8")
    )
    provider_request = diagnostics["providerRequests"][0]
    assert "requestedDurationSeconds" not in provider_request
    assert provider_request["actualDurationSeconds"] == 1
    assert provider_request["request"]["body"] == body
    assert provider_request["request"]["url"] == "https://minimax.test/v1/audio/jobs"
    assert provider_request["createResponse"]["body"] == {"jobId": "remote-job"}
    assert provider_request["statusResponse"]["body"]["status"] == "succeeded"
    assert provider_request["audioResponse"]["statusCode"] == 200


async def test_minimax_provider_offsets_seed_for_alternatives(tmp_path: Path) -> None:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return minimax_response(request)

    settings = make_settings(tmp_path, minimax_base_url="https://minimax.test")
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        await MiniMaxMusicProvider(settings, client).generate(
            "[Genre: Rock]", 60, MINIMAX_USER_PROMPT, variation=1
        )

    assert json.loads(requests[0].content)["seed"] == settings.minimax_seed + 1


async def test_minimax_provider_preserves_create_page_lyrics(tmp_path: Path) -> None:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return minimax_response(request)

    settings = make_settings(tmp_path, minimax_base_url="https://minimax.test")
    lyrics = "夜色落进空荡站台，最后一班车没有回来，月光沿着铁轨沉默地延伸，我仍在原地等待。"
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        await MiniMaxMusicProvider(settings, client).generate(
            "[Genre: Dream Pop]",
            60,
            f"[歌词与创作内容]\n[Verse]\n{lyrics}\n\n[风格要求]\n梦幻流行、空灵女声",
        )

    body = json.loads(requests[0].content)
    assert body["input"].splitlines()[0] == "[Verse]"
    assert len(body["input"].splitlines()) > 2
    assert body["input"].replace("\n", "") == f"[Verse]{lyrics}"
    assert "梦幻流行、空灵女声" not in body["instructions"]


async def test_minimax_provider_reports_server_error(tmp_path: Path) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(422, json={"detail": "invalid request"})

    settings = make_settings(tmp_path, minimax_base_url="https://minimax.test")
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        with pytest.raises(GenerationError, match="422.*invalid request"):
            await MiniMaxMusicProvider(settings, client).generate(
                "[Genre: Rock]", 120, MINIMAX_USER_PROMPT, job_id="failed-job"
            )

    diagnostics = json.loads(
        (settings.output_dir / "jobs/failed-job/prompts.json").read_text(encoding="utf-8")
    )["providerRequests"][0]
    assert "audio_duration" not in diagnostics["request"]["body"]
    assert diagnostics["createResponse"]["statusCode"] == 422
    assert diagnostics["createResponse"]["body"] == {"detail": "invalid request"}


async def test_minimax_provider_reports_direct_connection_target(tmp_path: Path) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused", request=request)

    settings = make_settings(tmp_path, minimax_base_url="http://192.168.1.4:8111")
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        with pytest.raises(
            GenerationError,
            match=r"192\.168\.1\.4:8111/v1/audio/jobs.*绕过系统代理",
        ):
            await MiniMaxMusicProvider(settings, client).generate(
                "[Genre: Rock]", 60, MINIMAX_USER_PROMPT
            )


def minimax_response(request):
    if request.method == "POST":
        return httpx.Response(202, json={"jobId": "remote-job"})
    if request.url.path.endswith("/audio"):
        return httpx.Response(200, content=wav_bytes())
    return httpx.Response(
        200, json={"status": "succeeded", "stage": "completed", "step": None, "totalSteps": None}
    )


async def test_mureka_uploads_reference_once_polls_and_downloads_wav(tmp_path: Path) -> None:
    settings = make_settings(tmp_path, mureka_api_key="test", music_poll_interval_seconds=0)
    directory = references_dir(settings.output_dir)
    directory.mkdir(parents=True)
    (directory / "clip.mp3").write_bytes(b"mp3")
    save_references(settings.output_dir, [new_reference_entry("ref1", "a.mp3", 30, "clip.mp3")])
    requests: list[httpx.Request] = []
    polls = {"count": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        path = request.url.path
        if request.url.host == "api.mureka.ai":
            assert request.headers["Authorization"] == "Bearer test"
        if path == "/v1/files/upload":
            assert b'name="purpose"' in request.content and b"reference" in request.content
            return httpx.Response(200, json={"id": "file-1"})
        if path == "/v1/song/generate":
            return httpx.Response(200, json={"id": "task-1", "status": "preparing"})
        if path == "/v1/song/query/task-1":
            polls["count"] += 1
            if polls["count"] == 1:
                return httpx.Response(200, json={"id": "task-1", "status": "running"})
            return httpx.Response(
                200,
                json={
                    "id": "task-1",
                    "status": "succeeded",
                    "model": "mureka-9.5",
                    "choices": [
                        {
                            "id": "song-1",
                            "url": "https://cdn.test/a.mp3",
                            "wav_url": "https://cdn.test/a.wav",
                            "duration": 61000,
                        }
                    ],
                },
            )
        assert str(request.url) == "https://cdn.test/a.wav"
        return httpx.Response(200, content=b"wav-bytes")

    reference = AudioReference("ref1", 30000, "")
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        provider = MurekaMusicProvider(settings, client)
        reports = []

        async def progress(stage, step, total_steps):
            reports.append((stage, step, total_steps))

        result = await provider.generate(
            "x" * 2000,
            120,
            MINIMAX_USER_PROMPT,
            job_id="job_1",
            reference=reference,
            progress=progress,
        )
        polls["count"] = 0
        await provider.generate(
            "rock", None, MINIMAX_USER_PROMPT, variation=1, job_id="job_1", reference=reference
        )

    assert reports == [
        ("preparing_reference", None, None),
        ("waiting_music", None, None),
        ("generating_music", None, None),
        ("generating_music", None, None),
    ]
    assert result.audio_path.read_bytes() == b"wav-bytes"
    assert result.audio_path.suffix == ".wav"
    assert result.debug["durationSeconds"] == 61
    paths = [request.url.path for request in requests]
    assert paths.count("/v1/files/upload") == 1
    body = json.loads(next(r for r in requests if r.url.path == "/v1/song/generate").content)
    assert body == {
        "lyrics": "[Verse]\ntest lyrics",
        "model": "mureka-9.5",
        "prompt": "x" * 1024,
        "n": 1,
        "reference_id": "file-1",
    }


async def test_mureka_direct_prompt_generates_lyrics_and_title(tmp_path: Path) -> None:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        path = request.url.path
        if path == "/v1/lyrics/generate":
            return httpx.Response(200, json={"title": "雨夜", "lyrics": "[Verse]\n第一句"})
        if path == "/v1/song/generate":
            return httpx.Response(200, json={"id": "task-1", "status": "preparing"})
        if path == "/v1/song/query/task-1":
            return httpx.Response(
                200,
                json={
                    "id": "task-1",
                    "status": "succeeded",
                    "choices": [
                        {
                            "id": "song-1",
                            "url": "https://cdn.test/a.mp3",
                            "duration": 1000,
                        }
                    ],
                },
            )
        return httpx.Response(200, content=b"mp3-bytes")

    settings = make_settings(tmp_path, mureka_api_key="test", music_poll_interval_seconds=0)
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        result = await MurekaMusicProvider(settings, client).generate(
            "温暖民谣", None, "温暖民谣"
        )

    assert result.title == "雨夜"
    assert result.lyrics == "[Verse]\n第一句"
    lyrics_request = next(r for r in requests if r.url.path == "/v1/lyrics/generate")
    assert json.loads(lyrics_request.content) == {"prompt": "温暖民谣"}
    song_request = next(r for r in requests if r.url.path == "/v1/song/generate")
    body = json.loads(song_request.content)
    assert body["lyrics"] == "[Verse]\n第一句"
    assert body["prompt"] == "温暖民谣"


@pytest.mark.parametrize("status", [401, 429, 500])
async def test_mureka_classifies_http_errors(tmp_path: Path, status: int) -> None:
    settings = make_settings(tmp_path, mureka_api_key="test")
    handler = lambda request: httpx.Response(status, json={"error": {"message": "no"}})  # noqa: E731
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        with pytest.raises(GenerationError) as caught:
            await MurekaMusicProvider(settings, client).generate("rock", None, MINIMAX_USER_PROMPT)
    assert isinstance(caught.value, ProviderGlobalError) == (status in {401, 429})


@pytest.mark.parametrize(
    ("status", "message", "is_global"),
    [
        (401, "鉴权或权限不足", True),
        (413, "超过服务限制", False),
        (422, "格式或时长不被接受", False),
        (500, "服务暂时不可用", False),
    ],
)
async def test_mureka_reference_upload_reports_specific_error(
    tmp_path: Path, status: int, message: str, is_global: bool
) -> None:
    settings = make_settings(tmp_path, mureka_api_key="test")
    directory = references_dir(settings.output_dir)
    directory.mkdir(parents=True)
    (directory / "clip.mp3").write_bytes(b"mp3")
    save_references(settings.output_dir, [new_reference_entry("ref", "a.mp3", 30, "clip.mp3")])

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/v1/files/upload"
        return httpx.Response(status, json={"error": {"message": "provider detail"}})

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        with pytest.raises(GenerationError, match=message) as caught:
            await MurekaMusicProvider(settings, client).generate(
                "rock", None, MINIMAX_USER_PROMPT, reference=AudioReference("ref", 30_000, "")
            )

    assert isinstance(caught.value, ProviderGlobalError) is is_global
    assert "ElevenLabs" not in str(caught.value)


async def test_mureka_failed_task_reports_reason(tmp_path: Path) -> None:
    settings = make_settings(tmp_path, mureka_api_key="test")
    handler = lambda request: httpx.Response(  # noqa: E731
        200, json={"id": "t", "status": "failed", "failed_reason": "content policy"}
    )
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        with pytest.raises(GenerationError, match="content policy"):
            await MurekaMusicProvider(settings, client).generate("rock", None, MINIMAX_USER_PROMPT)


async def test_mureka_loops_short_reference_to_30_seconds(tmp_path: Path) -> None:
    from app.services.references import prepare_reference_audio, reference_audio_name

    settings = make_settings(tmp_path, mureka_api_key="test")
    source = tmp_path / "short.wav"
    source.write_bytes(wav_bytes(5))
    clip = references_dir(settings.output_dir) / reference_audio_name()
    duration = await prepare_reference_audio(source, clip)
    save_references(
        settings.output_dir, [new_reference_entry("ref5", "s.wav", duration, clip.name)]
    )
    uploads: list[int] = []

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/v1/files/upload":
            uploads.append(len(request.content))
            return httpx.Response(200, json={"id": "file-5"})
        return httpx.Response(200, json={"id": "t", "status": "failed", "failed_reason": "stop"})

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        with pytest.raises(GenerationError, match="stop"):
            await MurekaMusicProvider(settings, client).generate(
                "rock", None, MINIMAX_USER_PROMPT, reference=AudioReference("ref5", 5000, "")
            )

    # 30 s at 192 kbps is about 720 kB; the original 5 s clip is about 120 kB.
    assert len(uploads) == 1 and uploads[0] > 600_000
    assert sorted(path.name for path in clip.parent.iterdir() if path.suffix != ".lock") == sorted(
        [clip.name, "index.json"]
    )

async def test_elevenlabs_detailed_multipart_handles_preamble_and_extra_parts(
    tmp_path: Path,
) -> None:
    pcm = b"\x00\x00\x01\x00" * 24_000
    boundary = "music-boundary"
    metadata = {
        "song_metadata": {"title": "雨后"},
        "composition_plan": {"chunks": [{"text": "[Verse]\n第一句"}]},
    }
    extra = {"words_timestamps": [{"word": "雨", "start_ms": 0}]}
    body = (
        b"preamble before the first boundary\r\n"
        + f"--{boundary}\r\nContent-Type: audio/pcm\r\n\r\n".encode()
        + pcm
        + b"\r\n"
        + f"--{boundary}\r\nContent-Type: application/json\r\n\r\n".encode()
        + json.dumps(metadata, ensure_ascii=False).encode()
        + b"\r\n"
        + f"--{boundary}\r\nContent-Type: application/json\r\n\r\n".encode()
        + json.dumps(extra, ensure_ascii=False).encode()
        + b"\r\n"
        + f"--{boundary}\r\nContent-Type: text/plain\r\n\r\n".encode()
        + b"ignored extra part\r\n"
        + f"--{boundary}--\r\n".encode()
    )

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            stream=ChunkedBody(body[:1], body[1:41], body[41:333], body[333:]),
            headers={"content-type": f"multipart/mixed; boundary={boundary}"},
        )

    settings = make_settings(
        tmp_path,
        elevenlabs_api_key="secret",
        elevenlabs_music_base_url="https://eleven.test",
        elevenlabs_music_output_format="pcm_8000",
    )
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        result = await ElevenLabsMusicProvider(settings, client).generate(
            "[Genre: Folk]", 3, "温暖民谣", direct_prompt=True, job_id="multipart-job"
        )

    assert result.title == "雨后"
    assert result.lyrics == "[Verse]\n第一句"
    with wave.open(str(result.audio_path), "rb") as audio:
        assert audio.readframes(audio.getnframes()) == pcm
