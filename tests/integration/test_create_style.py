import json
from pathlib import Path
from unittest.mock import AsyncMock

import httpx
import pytest

from app.main import create_app
from app.services.prompt import (
    CHINESE_STYLE_CATEGORIES,
    OpenAICompatiblePromptExpander,
    validate_chinese_style,
)
from app.services.providers import MusicResult
from tests.helpers import make_orchestrator, make_settings

RAW_CHINESE_STYLE = ", ".join(
    f"[{name}: 温暖清晰的民谣制作要求]" for name in CHINESE_STYLE_CATEGORIES
)
CHINESE_STYLE = "\n".join(f"{name}：温暖清晰的民谣制作要求" for name in CHINESE_STYLE_CATEGORIES)


async def test_style_expansion_only_returns_chinese_style_and_retries(tmp_path: Path):
    calls = []

    def handler(request):
        calls.append(json.loads(request.content))
        style = "English only" if len(calls) == 1 else RAW_CHINESE_STYLE
        return httpx.Response(
            200, json={"choices": [{"message": {"content": json.dumps({"styleTags": style})}}]}
        )

    settings = make_settings(
        tmp_path, llm_api_key=None, llm_temperature=0.7, llm_model="Qwen/Qwen3.5-9B-FP8"
    )
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        expanded = await OpenAICompatiblePromptExpander(settings, client).expand_style("雨夜民谣")
    assert expanded == CHINESE_STYLE
    assert "[" not in expanded and "]" not in expanded
    assert len(expanded.splitlines()) == len(CHINESE_STYLE_CATEGORIES)
    assert len(calls) == 2
    assert "不生成歌词正文" in calls[0]["messages"][0]["content"]
    assert "可演唱的普通话主唱" in calls[0]["messages"][0]["content"]
    assert "无歌词演唱" in calls[0]["messages"][0]["content"]
    assert "音阶" in calls[0]["messages"][0]["content"]
    assert "音程" in calls[0]["messages"][0]["content"]
    assert calls[0]["temperature"] == 0.7
    assert calls[0]["chat_template_kwargs"] == {"enable_thinking": False}
    assert calls[0]["response_format"]["type"] == "json_schema"
    assert "上次响应" in calls[1]["messages"][-1]["content"]


async def test_style_expansion_endpoint_validates_and_does_not_create_job(tmp_path: Path):
    settings = make_settings(tmp_path)
    orchestrator = make_orchestrator(settings)
    orchestrator.prompt_expander.expand_style = AsyncMock(return_value=CHINESE_STYLE)
    app = create_app(settings, orchestrator)
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        for prompt in ["", " ", "文" * 2001]:
            assert (
                await client.post("/api/styles/expand", json={"prompt": prompt})
            ).status_code in (400, 422)
        response = await client.post("/api/styles/expand", json={"prompt": "雨夜民谣"})
    assert response.json() == {"style": CHINESE_STYLE}
    assert not app.state.jobs
    orchestrator.prompt_expander.expand_style.assert_awaited_once_with("雨夜民谣")


@pytest.mark.parametrize("provider", ["mureka_music", "elevenlabs_music"])
@pytest.mark.parametrize("prompt", ["  雨夜民谣  ", CHINESE_STYLE])
async def test_direct_style_skips_expansion_and_llm_calls(
    tmp_path: Path, provider: str, prompt: str
):
    settings = make_settings(tmp_path)
    orchestrator = make_orchestrator(settings)
    orchestrator.prompt_expander.prepare = AsyncMock(side_effect=AssertionError("must not expand"))
    supplied = []
    original = orchestrator.music_provider.generate

    async def capture(style, duration, raw_prompt, **kwargs):
        supplied.append((style, raw_prompt))
        return await original(style, duration, raw_prompt, **kwargs)

    orchestrator.music_provider.generate = capture
    app = create_app(settings, orchestrator)
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        response = await client.post(
            "/api/generate", json={"prompt": prompt, "provider": provider, "directPrompt": True}
        )
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["prompt"] == body["structuredPrompt"] == prompt
    assert body["directPrompt"] is True
    assert supplied == [(prompt, prompt)]
    orchestrator.prompt_expander.prepare.assert_not_awaited()


async def test_fullwidth_chinese_colons_do_not_trigger_retry(tmp_path: Path):
    calls = []

    def handler(request):
        calls.append(request)
        content = json.dumps({"styleTags": RAW_CHINESE_STYLE.replace(":", "：")})
        return httpx.Response(200, json={"choices": [{"message": {"content": content}}]})

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        style = await OpenAICompatiblePromptExpander(make_settings(tmp_path), client).expand_style(
            "古风"
        )
    assert style == CHINESE_STYLE
    assert len(calls) == 1


@pytest.mark.parametrize("payload", [{"provider": "minimax_music"}, {}])
async def test_direct_minimax_is_temporarily_unavailable(
    tmp_path: Path, payload: dict[str, str]
):
    settings = make_settings(tmp_path)
    app = create_app(settings, make_orchestrator(settings))
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        response = await client.post(
            "/api/generate", json={**payload, "prompt": "雨夜民谣", "directPrompt": True}
        )
    assert response.status_code == 400
    assert "暂时停止维护" in response.json()["message"]


async def test_mureka_truncates_long_raw_style_and_reports_it(tmp_path: Path):
    from app.services.providers import MurekaMusicProvider

    requests = []

    def handler(request):
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
                        {"id": "song-1", "url": "https://cdn.test/a.mp3", "duration": 1000}
                    ],
                },
            )
        return httpx.Response(200, content=b"mp3-bytes")

    settings = make_settings(
        tmp_path, mureka_api_key="test", music_poll_interval_seconds=0
    )
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        result = await MurekaMusicProvider(settings, client).generate(
            "风" * 1025, None, "风" * 1025
        )

    lyrics_request = next(r for r in requests if r.url.path == "/v1/lyrics/generate")
    assert len(json.loads(lyrics_request.content)["prompt"]) == 1024
    song_request = next(r for r in requests if r.url.path == "/v1/song/generate")
    assert len(json.loads(song_request.content)["prompt"]) == 1024
    assert result.debug["styleTruncated"] is True
    assert result.lyrics == "[Verse]\n第一句"

async def test_direct_provider_metadata_is_saved_to_job_json(tmp_path: Path):
    settings = make_settings(tmp_path)
    orchestrator = make_orchestrator(settings)

    async def with_metadata(style, duration, raw_prompt, **kwargs):
        return MusicResult(
            settings.mock_full_song_path,
            {"provider": "mock"},
            title="雨夜",
            lyrics="[Verse]\n第一句",
        )

    orchestrator.music_provider.generate = with_metadata
    app = create_app(settings, orchestrator)
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        response = await client.post(
            "/api/generate",
            json={"prompt": "温暖民谣", "provider": "mureka_music", "directPrompt": True},
        )
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["title"] == "雨夜"
    assert body["lyrics"] == "[Verse]\n第一句"
    assert body["directPrompt"] is True
    assert body["alternatives"] == []
    job = json.loads(
        (settings.output_dir / f"jobs/{body['jobId']}/job.json").read_text(encoding="utf-8")
    )
    assert job["title"] == "雨夜"
    assert job["lyrics"] == "[Verse]\n第一句"

def test_chinese_style_rejects_no_vocal_human_tag():
    no_vocal = RAW_CHINESE_STYLE.replace(
        "[人声: 温暖清晰的民谣制作要求]",
        "[人声: 无歌词演唱，仅使用女性哼唱（Vocal Chops）作为点缀]",
    )
    with pytest.raises(ValueError, match="人声标签"):
        validate_chinese_style(no_vocal)

async def test_mureka_style_truncation_is_returned_as_warning(tmp_path: Path):
    settings = make_settings(tmp_path)
    orchestrator = make_orchestrator(settings)

    async def truncated(style, duration, raw_prompt, **kwargs):
        return MusicResult(
            settings.mock_full_song_path,
            {"provider": "mureka_music", "styleTruncated": True},
            title="雨夜",
            lyrics="[Verse]\n第一句",
        )

    orchestrator.music_provider.generate = truncated
    app = create_app(settings, orchestrator)
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        response = await client.post(
            "/api/generate",
            json={"prompt": "风" * 1025, "provider": "mureka_music", "directPrompt": True},
        )
    assert response.status_code == 200, response.text
    assert "1024" in response.json()["warning"]
