from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
from unittest.mock import AsyncMock

import httpx
import pytest

from app.main import create_app
from app.services.prompt import OpenAICompatiblePromptExpander, PreparedPrompt
from tests.helpers import (
    BlockingPromptExpander,
    StubMusicProvider,
    make_orchestrator,
    make_settings,
)


async def _client(app) -> httpx.AsyncClient:
    return httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app),
        base_url="http://testserver",
    )


def _clear_proxy_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    """Unset every proxy variable inherited from the host shell.

    Keeps the test deterministic regardless of the machine running it: a
    developer's shell often exports ``http_proxy``/``all_proxy`` etc., and the
    uppercase/lowercase variants can shadow the variables a test sets.
    """
    for name in list(os.environ):
        if name.lower().endswith("_proxy"):
            monkeypatch.delenv(name, raising=False)


async def test_health_preserves_legacy_contract(tmp_path: Path) -> None:
    settings = make_settings(tmp_path)
    app = create_app(settings, make_orchestrator(settings))
    async with await _client(app) as client:
        response = await client.get("/api/health")
    assert response.status_code == 200
    body = response.json()
    assert body["ok"] is True
    assert body["maxConcurrentGenerations"] == 1
    assert body["outputUrl"] == "http://testserver/output"
    assert body["llm"]["timeoutSeconds"] == 120
    assert body["llm"]["maxTokens"] == 2048
    assert body["llm"]["disableThinking"] is True
    assert body["splitting"]["enabled"] is True
    assert body["splitting"]["mode"] == "on_demand"
    assert body["splitting"]["device"] == "auto"
    assert body["splitting"]["engine"] == "bs_roformer"
    assert body["splitting"]["model"] == "roformer-model-bs-roformer-sw-by-jarredou"
    assert body["splitting"]["stems"] == ["vocal", "drums", "bass", "guitar", "piano", "other"]
    assert body["splitting"]["timeoutSeconds"] == 1800
    assert body["replacement"] == {
        "conversionTimeoutSeconds": 1800,
        "workersHoldingCapacityAfterTerminal": 0,
    }
    assert body["infrastructure"] == {
        "taskBackend": "inline",
        "cacheBackend": "none",
        "eventBackend": "none",
    }


async def test_cors_wildcard_never_allows_credentials(tmp_path: Path) -> None:
    settings = make_settings(tmp_path, cors_origins="*,https://ui.test")
    app = create_app(settings, make_orchestrator(settings))
    async with await _client(app) as client:
        response = await client.get("/api/health", headers={"Origin": "https://evil.test"})
    assert response.headers["access-control-allow-origin"] == "*"
    assert "access-control-allow-credentials" not in response.headers


async def test_generate_validates_prompt_boundaries(tmp_path: Path) -> None:
    settings = make_settings(tmp_path, prompt_max_chars=2000)
    app = create_app(settings, make_orchestrator(settings))
    async with await _client(app) as client:
        blank = await client.post("/api/generate", json={"prompt": "   "})
        maximum = await client.post("/api/generate", json={"prompt": "乐" * 2000})
        too_long = await client.post("/api/generate", json={"prompt": "乐" * 2001})
    assert blank.status_code == 400
    assert maximum.status_code == 200
    assert too_long.status_code == 400
    assert "2000" in too_long.json()["message"]


@pytest.mark.parametrize("endpoint", ["/api/generate", "/api/jobs"])
@pytest.mark.parametrize(
    "prompt",
    [
        "[歌词与创作内容]\n\n[风格要求]",
        "[歌词与创作内容]\n\n[风格要求]\n",
        "[歌词与创作内容]\r\n \t\r\n[风格要求]\r\n \t",
        "[歌词与创作内容]\n \t",
        "[风格要求]\n \t",
    ],
)
async def test_generation_rejects_empty_lyrics_and_style_before_creating_job(
    tmp_path: Path, endpoint: str, prompt: str
) -> None:
    settings = make_settings(tmp_path)
    orchestrator = make_orchestrator(settings)
    orchestrator.prompt_expander.prepare = AsyncMock()
    app = create_app(settings, orchestrator)
    async with await _client(app) as client:
        response = await client.post(endpoint, json={"prompt": prompt})

    assert response.status_code == 400
    assert response.json() == {"success": False, "message": "歌词和风格至少需要提供一项。"}
    orchestrator.prompt_expander.prepare.assert_not_awaited()
    assert app.state.jobs == {}
    assert not list((settings.output_dir / "jobs").glob("*/job.json"))


@pytest.mark.parametrize("prompt", ["[风格要求]\n摇滚", "[歌词与创作内容]\n\n[风格要求]\n摇滚"])
async def test_generate_accepts_style_with_empty_lyrics(tmp_path: Path, prompt: str) -> None:
    settings = make_settings(tmp_path)
    orchestrator = make_orchestrator(settings)
    orchestrator.prompt_expander.prepare = AsyncMock(
        return_value=PreparedPrompt("[Genre: Rock]", "[verse]\n自动生成的测试歌词", None)
    )
    app = create_app(settings, orchestrator)
    async with await _client(app) as client:
        response = await client.post("/api/generate", json={"prompt": prompt})

    assert response.status_code == 200
    assert orchestrator.prompt_expander.prepare.await_args.args[0] == prompt


async def test_generate_preserves_not_ready_response(tmp_path: Path) -> None:
    app = create_app(make_settings(tmp_path))
    async with await _client(app) as client:
        response = await client.post("/api/generate", json={"prompt": "ambient"})
    assert response.status_code == 503
    assert response.json()["detail"] == "Application is not ready"


async def test_ready_accepts_music_api_key_for_elevenlabs(tmp_path: Path) -> None:
    settings = make_settings(
        tmp_path,
        music_provider="elevenlabs_music",
        music_api_key="generic-key",
    )
    app = create_app(settings)
    async with await _client(app) as client:
        response = await client.get("/api/health/ready")
    assert response.status_code == 200
    assert response.json()["checks"]["provider"] is True


async def test_generate_returns_full_track_without_splitting(tmp_path: Path) -> None:
    settings = make_settings(
        tmp_path,
        music_provider="elevenlabs_music",
    )
    orchestrator = make_orchestrator(settings)
    orchestrator.stem_separator.split = AsyncMock(side_effect=AssertionError("no splitting"))
    app = create_app(settings, orchestrator)
    async with await _client(app) as client:
        response = await client.post(
            "/api/generate",
            json={"prompt": "明亮的普通话摇滚", "durationMinutes": 3},
        )
        assert response.status_code == 200
        body = response.json()
        assert body["title"] == "测试歌名"
        assert body["durationMinutes"] == 3
        assert body["requestedDurationSeconds"] == 180
        assert body["debug"]["music"]["durationSeconds"] == 180
        assert body["splitEnabled"] is False
        assert body["stems"] == {}
        assert body["waveforms"] == {}
        assert "splitterDurationMs" not in body["debug"]
        audio = await client.get(body["fullTrack"])
    assert audio.status_code == 200
    assert audio.headers["content-type"].startswith("audio/mpeg")
    assert audio.headers["cache-control"] == "no-store"
    assert audio.content == b"ID3-full-audio"
    assert orchestrator.music_provider.requested_durations == [180]
    orchestrator.stem_separator.split.assert_not_called()


async def test_generate_accepts_exact_duration_seconds(tmp_path: Path) -> None:
    settings = make_settings(
        tmp_path, enable_audio_splitting=False, music_provider="elevenlabs_music"
    )
    orchestrator = make_orchestrator(settings)
    app = create_app(settings, orchestrator)
    async with await _client(app) as client:
        response = await client.post(
            "/api/generate",
            json={"prompt": "三十秒试听", "durationMinutes": 3, "durationSeconds": 30},
        )

    assert response.status_code == 200
    body = response.json()
    assert body["durationMinutes"] == 0.5
    assert body["requestedDurationSeconds"] == 30
    assert orchestrator.music_provider.requested_durations == [30]


@pytest.mark.parametrize(
    "duration_field",
    [{"durationMinutes": "auto"}, {"durationMinutes": None}, {}],
)
@pytest.mark.parametrize("provider", ["minimax_music", "elevenlabs_music"])
async def test_generate_lets_provider_choose_duration_for_auto(
    tmp_path: Path, duration_field: dict[str, object], provider: str
) -> None:
    settings = make_settings(tmp_path)
    orchestrator = make_orchestrator(settings)
    app = create_app(settings, orchestrator)
    async with await _client(app) as client:
        response = await client.post(
            "/api/generate",
            json={"prompt": "自动长度普通话歌曲", "provider": provider, **duration_field},
        )

    assert response.status_code == 200
    body = response.json()
    assert body["durationMinutes"] == "auto"
    assert "requestedDurationSeconds" not in body
    assert body["debug"]["music"]["durationSeconds"] == 150
    assert orchestrator.music_provider.requested_durations == [None]
    diagnostics = json.loads(
        (settings.output_dir / "jobs" / body["jobId"] / "prompts.json").read_text()
    )
    assert diagnostics["provider"] == provider
    assert diagnostics["durationSource"] == "provider"
    assert "effectiveDurationSeconds" not in diagnostics
    assert "effectiveDurationMinutes" not in diagnostics


async def test_generate_selects_provider_per_request(tmp_path: Path) -> None:
    settings = make_settings(tmp_path)
    orchestrator = make_orchestrator(settings)
    orchestrator.music_providers = {
        name: StubMusicProvider(settings.mock_full_song_path, name)
        for name in ("minimax_music", "elevenlabs_music")
    }
    app = create_app(settings, orchestrator)

    async with await _client(app) as client:
        for provider in orchestrator.music_providers:
            response = await client.post(
                "/api/generate", json={"prompt": "rock", "provider": provider}
            )
            assert response.status_code == 200
            assert response.json()["debug"]["music"]["provider"] == provider


async def test_minimax_count_two_returns_two_results(tmp_path: Path) -> None:
    settings = make_settings(tmp_path)
    orchestrator = make_orchestrator(settings)
    provider = StubMusicProvider(settings.mock_full_song_path, "minimax_music")
    orchestrator.music_providers = {"minimax_music": provider}
    app = create_app(settings, orchestrator)

    async with await _client(app) as client:
        response = await client.post(
            "/api/generate",
            json={"prompt": "两首摇滚", "provider": "minimax_music", "count": 2},
        )

    body = response.json()
    assert response.status_code == 200
    assert body["count"] == 2
    assert len(body["alternatives"]) == 1
    assert provider.variations == [0, 1]


async def test_elevenlabs_count_two_returns_two_results(tmp_path: Path) -> None:
    settings = make_settings(tmp_path)
    orchestrator = make_orchestrator(settings)
    provider = StubMusicProvider(settings.mock_full_song_path, "elevenlabs_music")
    orchestrator.music_providers = {"elevenlabs_music": provider}
    app = create_app(settings, orchestrator)

    async with await _client(app) as client:
        response = await client.post(
            "/api/generate",
            json={"prompt": "两首摇滚", "provider": "elevenlabs_music", "count": 2},
        )

    body = response.json()
    assert response.status_code == 200
    assert body["count"] == 2
    assert len(body["alternatives"]) == 1
    assert body["warning"] is None
    assert provider.variations == [0, 1]


async def test_ai_talk_lyrics_are_tagged_before_music_generation(tmp_path: Path) -> None:
    settings = make_settings(tmp_path)
    orchestrator = make_orchestrator(settings)
    provider = StubMusicProvider(settings.mock_full_song_path)
    orchestrator.music_provider = provider
    app = create_app(settings, orchestrator)

    async with await _client(app) as client:
        response = await client.post(
            "/api/generate",
            json={"prompt": ("[歌词与创作内容]\n第一句\n第二句\n\n[风格要求]\n梦幻流行")},
        )

    assert response.status_code == 200
    assert provider.user_prompt == (
        "[歌词与创作内容]\n[Verse]\n第一句\n第二句\n\n[风格要求]\n梦幻流行"
    )


@pytest.mark.parametrize("provider_name", ["minimax_music", "elevenlabs_music"])
async def test_lyrics_only_inferred_style_reaches_music_provider(
    tmp_path: Path, provider_name: str
) -> None:
    settings = make_settings(tmp_path, llm_api_key="secret", llm_base_url="https://llm.test/v1")
    style = (
        "[Genre: Mandarin rock], [Tempo: 132 BPM in 4/4], [Mood: Uplifting], "
        "[Instrumentation: Electric guitars and drums], [Vocal: Clear Mandarin lead], "
        "[Arrangement: Verse and chorus], [Production: Clean stereo mix], "
        "[Negative Constraints: No buried vocals]"
    )

    def handler(request: httpx.Request) -> httpx.Response:
        messages = json.loads(request.content)["messages"]
        assert "根据歌词的主题、情绪、意象和语言节奏推断" in messages[1]["content"]
        content = json.dumps({"taggedLyrics": "[Verse]\n第一句\n第二句", "styleTags": style})
        return httpx.Response(200, json={"choices": [{"message": {"content": content}}]})

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as llm_client:
        orchestrator = make_orchestrator(
            settings, prompt_expander=OpenAICompatiblePromptExpander(settings, llm_client)
        )
        provider = StubMusicProvider(settings.mock_full_song_path, provider_name)
        provider.generate = AsyncMock(wraps=provider.generate)
        orchestrator.music_providers = {provider_name: provider}
        app = create_app(settings, orchestrator)
        async with await _client(app) as client:
            response = await client.post(
                "/api/generate",
                json={"prompt": "[歌词与创作内容]\n第一句\n第二句", "provider": provider_name},
            )

    assert response.status_code == 200
    assert response.json()["lyrics"] == "[verse]\n第一句\n第二句"
    assert response.json()["structuredPrompt"] == style
    assert provider.user_prompt == "[歌词与创作内容]\n[verse]\n第一句\n第二句"
    assert provider.generate.await_args.args[0] == style


async def test_minimax_ignores_fixed_duration_and_preserves_user_lyrics(tmp_path: Path) -> None:
    settings = make_settings(tmp_path)
    orchestrator = make_orchestrator(settings)
    provider = StubMusicProvider(settings.mock_full_song_path, "minimax_music")
    orchestrator.music_providers = {"minimax_music": provider}
    app = create_app(settings, orchestrator)
    original = "\n".join(f"第 {index} 句用户歌词" for index in range(15))

    async with await _client(app) as client:
        response = await client.post(
            "/api/generate",
            json={
                "prompt": f"[歌词与创作内容]\n{original}\n\n[风格要求]\n梦幻流行",
                "durationMinutes": 1,
                "provider": "minimax_music",
            },
        )

    expected = f"[Verse]\n{original}"
    assert response.status_code == 200
    assert response.json()["durationMinutes"] == 1
    assert "requestedDurationSeconds" not in response.json()
    assert response.json()["lyrics"] == expected
    assert provider.user_prompt == f"[歌词与创作内容]\n{expected}\n\n[风格要求]\n梦幻流行"
    assert provider.requested_durations == [None]
    diagnostics = json.loads(
        (settings.output_dir / f"jobs/{response.json()['jobId']}/prompts.json").read_text(
            encoding="utf-8"
        )
    )
    assert diagnostics["requestedDurationMinutes"] == 1
    assert diagnostics["durationSource"] == "provider_override"


async def test_async_job_reports_real_stage_and_result(tmp_path: Path) -> None:
    settings = make_settings(tmp_path)
    blocker = BlockingPromptExpander()
    app = create_app(settings, make_orchestrator(settings, prompt_expander=blocker))
    async with await _client(app) as client:
        created = await client.post("/api/jobs", json={"prompt": "真实进度"})
        assert created.status_code == 202
        job_id = created.json()["jobId"]
        await blocker.started.wait()

        running = await client.get(f"/api/jobs/{job_id}")
        assert running.json()["status"] == "running"
        assert running.json()["stage"] == "expanding_prompt"
        assert running.json()["progress"] is None
        assert running.json()["step"] is None
        assert running.json()["totalSteps"] is None
        assert running.json()["prompt"] == "真实进度"
        assert running.json()["structuredPrompt"] is None
        prompt_file = settings.output_dir / "jobs" / job_id / "prompts.json"
        initial_diagnostics = json.loads(prompt_file.read_text(encoding="utf-8"))
        assert initial_diagnostics["jobId"] == job_id
        assert initial_diagnostics["prompt"] == "真实进度"
        assert initial_diagnostics["structuredPrompt"] is None
        assert initial_diagnostics["lyrics"] is None
        assert initial_diagnostics["provider"] == "minimax_music"
        assert initial_diagnostics["requestedDurationMinutes"] == "auto"
        assert initial_diagnostics["requestedCount"] == 1
        assert initial_diagnostics["effectiveCount"] == 1
        assert initial_diagnostics["requestId"]

        blocker.release.set()
        for _ in range(20):
            completed = await client.get(f"/api/jobs/{job_id}")
            if completed.json()["status"] == "succeeded":
                break
            await asyncio.sleep(0)

    body = completed.json()
    assert body["progress"] == 100
    assert body["prompt"] == "真实进度"
    assert body["structuredPrompt"] == "[Genre: Test]"
    assert body["lyrics"] == "[Verse]\n自动生成的测试歌词"
    final_diagnostics = json.loads(prompt_file.read_text(encoding="utf-8"))
    assert final_diagnostics["structuredPrompt"] == "[Genre: Test]"
    assert final_diagnostics["lyrics"] == "[Verse]\n自动生成的测试歌词"
    assert final_diagnostics["style"] == "真实进度"
    assert final_diagnostics["providerPrompt"].startswith("[歌词与创作内容]")
    assert final_diagnostics["durationSource"] == "provider"
    assert "effectiveDurationSeconds" not in final_diagnostics
    assert "effectiveDurationMinutes" not in final_diagnostics
    assert body["result"]["splitEnabled"] is False
    assert body["result"]["stems"] == {}


async def test_async_job_can_be_cancelled(tmp_path: Path) -> None:
    settings = make_settings(tmp_path)
    blocker = BlockingPromptExpander()
    orchestrator = make_orchestrator(settings, prompt_expander=blocker)
    app = create_app(settings, orchestrator)
    async with await _client(app) as client:
        created = await client.post("/api/jobs", json={"prompt": "取消任务"})
        job_id = created.json()["jobId"]
        await blocker.started.wait()
        cancelled = await client.patch(f"/api/jobs/{job_id}", json={"status": "cancelled"})
        await asyncio.sleep(0)

    assert cancelled.status_code == 200
    assert cancelled.json()["status"] == "cancelled"
    assert orchestrator.capacity.active == 0


async def test_async_job_deleted_stem_stays_deleted(tmp_path: Path) -> None:
    settings = make_settings(tmp_path)
    app = create_app(settings, make_orchestrator(settings))
    async with await _client(app) as client:
        created = await client.post("/api/jobs", json={"prompt": "删除分轨"})
        job_id = created.json()["jobId"]
        for _ in range(20):
            completed = await client.get(f"/api/jobs/{job_id}")
            if completed.json()["status"] == "succeeded":
                break
            await asyncio.sleep(0)

        seed_legacy_stems(app, settings, job_id)
        completed = await client.get(f"/api/jobs/{job_id}")
        vocal_url = completed.json()["result"]["stems"]["vocal"]
        deleted = await client.delete(f"/api/jobs/{job_id}/stems/vocal")
        refreshed = await client.get(f"/api/jobs/{job_id}")
        missing_file = await client.get(vocal_url)
        hidden_trash = await client.get(f"/output/.trash/{job_id}/{Path(vocal_url).name}")
        restored = await client.put(f"/api/jobs/{job_id}/stems/vocal")
        restored_file = await client.get(vocal_url)

    assert deleted.status_code == 204
    assert deleted.content == b""
    assert "vocal" not in refreshed.json()["result"]["stems"]
    assert missing_file.status_code == 404
    assert hidden_trash.status_code == 404
    assert restored.status_code == 200
    assert "vocal" in restored.json()["result"]["stems"]
    assert restored_file.status_code == 200


async def test_async_second_song_stem_can_be_deleted_and_restored(tmp_path: Path) -> None:
    settings = make_settings(tmp_path)
    orchestrator = make_orchestrator(settings)
    provider = StubMusicProvider(settings.mock_full_song_path, "minimax_music")
    orchestrator.music_providers = {"minimax_music": provider}
    app = create_app(settings, orchestrator)

    async with await _client(app) as client:
        created = await client.post(
            "/api/jobs",
            json={"prompt": "两首摇滚", "provider": "minimax_music", "count": 2},
        )
        job_id = created.json()["jobId"]
        for _ in range(20):
            completed = await client.get(f"/api/jobs/{job_id}")
            if completed.json()["status"] == "succeeded":
                break
            await asyncio.sleep(0)

        seed_legacy_stems(app, settings, job_id)
        deleted = await client.delete(f"/api/jobs/{job_id}/stems/vocal?song=1")
        after_delete = (await client.get(f"/api/jobs/{job_id}")).json()
        restored = await client.put(f"/api/jobs/{job_id}/stems/vocal?song=1")

    assert deleted.status_code == 204
    assert "vocal" in after_delete["result"]["stems"]
    assert "vocal" not in after_delete["result"]["alternatives"][0]["stems"]
    assert restored.status_code == 200
    assert "vocal" in restored.json()["result"]["alternatives"][0]["stems"]


async def test_restore_forgets_missing_trash_file(tmp_path: Path) -> None:
    settings = make_settings(tmp_path)
    app = create_app(settings, make_orchestrator(settings))
    async with await _client(app) as client:
        created = await client.post("/api/jobs", json={"prompt": "永久删除分轨"})
        job_id = created.json()["jobId"]
        for _ in range(20):
            completed = await client.get(f"/api/jobs/{job_id}")
            if completed.json()["status"] == "succeeded":
                break
            await asyncio.sleep(0)

        seed_legacy_stems(app, settings, job_id)
        completed = await client.get(f"/api/jobs/{job_id}")
        vocal_url = completed.json()["result"]["stems"]["vocal"]
        await client.delete(f"/api/jobs/{job_id}/stems/vocal")
        trash = settings.output_dir / ".trash" / job_id / Path(vocal_url).name
        trash.unlink()
        missing = await client.put(f"/api/jobs/{job_id}/stems/vocal")
        retried = await client.put(f"/api/jobs/{job_id}/stems/vocal")

    assert missing.status_code == 404
    assert "已不存在" in missing.json()["message"]
    assert retried.status_code == 404
    assert "没有可恢复" in retried.json()["message"]


async def test_generation_returns_full_track_without_stems(tmp_path: Path) -> None:
    settings = make_settings(tmp_path)
    app = create_app(settings, make_orchestrator(settings))
    async with await _client(app) as client:
        response = await client.post("/api/generate", json={"prompt": "ambient"})
    body = response.json()
    assert body["splitEnabled"] is False
    assert body["stems"] == {}


async def test_request_size_limit_returns_413(tmp_path: Path) -> None:
    settings = make_settings(tmp_path, request_max_bytes=1024)
    app = create_app(settings, make_orchestrator(settings))
    async with await _client(app) as client:
        response = await client.post(
            "/api/generate",
            content=b"x" * 1025,
            headers={"content-type": "application/json"},
        )
    assert response.status_code == 413, response.text
    assert response.json()["success"] is False


async def test_streamed_request_size_limit_returns_413(tmp_path: Path) -> None:
    async def oversized_body():
        yield b'{"prompt":"' + b"x" * 600
        yield b"x" * 600 + b'"}'

    settings = make_settings(tmp_path, request_max_bytes=1024)
    app = create_app(settings, make_orchestrator(settings))
    async with await _client(app) as client:
        response = await client.post(
            "/api/generate",
            content=oversized_body(),
            headers={"content-type": "application/json"},
        )
    assert response.status_code == 413, response.text
    assert response.json()["success"] is False


async def test_invalid_content_length_returns_400(tmp_path: Path) -> None:
    settings = make_settings(tmp_path)
    app = create_app(settings, make_orchestrator(settings))
    async with await _client(app) as client:
        response = await client.post(
            "/api/generate",
            content=b"{" + b'"prompt":"ambient"}',
            headers={"content-type": "application/json", "content-length": "invalid"},
        )
    assert response.status_code == 400
    assert "Content-Length" in response.json()["message"]


async def test_concurrent_generation_returns_429_without_queueing(tmp_path: Path) -> None:
    settings = make_settings(tmp_path, max_concurrent_generations=1)
    blocker = BlockingPromptExpander()
    orchestrator = make_orchestrator(settings, prompt_expander=blocker)
    app = create_app(settings, orchestrator)
    async with await _client(app) as client:
        first = asyncio.create_task(client.post("/api/generate", json={"prompt": "first"}))
        await blocker.started.wait()
        second = await client.post("/api/generate", json={"prompt": "second"})
        blocker.release.set()
        first_response = await first
    assert second.status_code == 429
    assert first_response.status_code == 200
    assert orchestrator.capacity.active == 0


async def test_sync_generation_blocks_job_admission(tmp_path: Path) -> None:
    settings = make_settings(tmp_path, max_concurrent_generations=1)
    blocker = BlockingPromptExpander()
    orchestrator = make_orchestrator(settings, prompt_expander=blocker)
    app = create_app(settings, orchestrator)
    async with await _client(app) as client:
        generation = asyncio.create_task(client.post("/api/generate", json={"prompt": "sync"}))
        await blocker.started.wait()
        job = await client.post("/api/jobs", json={"prompt": "job"})
        blocker.release.set()
        generated = await generation
    assert job.status_code == 429
    assert generated.status_code == 200
    assert orchestrator.capacity.active == 0


async def test_audio_route_rejects_missing_file(tmp_path: Path) -> None:
    settings = make_settings(tmp_path)
    app = create_app(settings, make_orchestrator(settings))
    async with await _client(app) as client:
        response = await client.get("/output/not-found.mp3")
    assert response.status_code == 404


async def test_audio_route_supports_byte_ranges(tmp_path: Path) -> None:
    settings = make_settings(tmp_path)
    app = create_app(settings, make_orchestrator(settings))
    async with await _client(app) as client:
        generated = await client.post("/api/generate", json={"prompt": "range test"})
        response = await client.get(generated.json()["fullTrack"], headers={"Range": "bytes=0-2"})
    assert response.status_code == 206
    assert response.content == b"ID3"
    assert response.headers["content-range"].startswith("bytes 0-2/")


async def test_audio_route_returns_empty_file_with_zero_length(tmp_path: Path) -> None:
    settings = make_settings(tmp_path)
    app = create_app(settings, make_orchestrator(settings))
    (settings.output_dir / "empty.mp3").touch()
    async with await _client(app) as client:
        response = await client.get("/output/empty.mp3")
    assert response.status_code == 200
    assert response.headers["content-length"] == "0"
    assert response.content == b""


@pytest.mark.parametrize(
    "proxy_variables",
    [
        {},
        {"ALL_PROXY": "socks5://127.0.0.1:7897"},
        {"ALL_PROXY": "socks://127.0.0.1:7897"},
        {"all_proxy": "socks://127.0.0.1:7897"},
        {"ALL_PROXY": "socks5://127.0.0.1:7897", "all_proxy": "socks://127.0.0.1:7897"},
        {"HTTP_PROXY": "http://127.0.0.1:7897", "HTTPS_PROXY": "http://127.0.0.1:7897"},
        {"ALL_PROXY": "http://127.0.0.1:7897"},
        {"ALL_PROXY": "socks4://127.0.0.1:7897"},
    ],
    ids=[
        "no-proxy",
        "socks5",
        "socks-alias",
        "lowercase-socks-alias",
        "mixed-case-socks",
        "http-https",
        "http-all",
        "unsupported-socks4-dropped",
    ],
)
async def test_application_lifespan_accepts_proxy_environment(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    proxy_variables: dict[str, str],
) -> None:
    settings = make_settings(tmp_path)
    _clear_proxy_environment(monkeypatch)
    for name, value in proxy_variables.items():
        monkeypatch.setenv(name, value)
    app = create_app(settings)

    async with app.router.lifespan_context(app):
        assert app.state.orchestrator is not None


def seed_legacy_stems(app, settings, job_id):
    job = app.state.jobs[job_id]
    for index, result in enumerate([job.result, *job.result.get("alternatives", [])]):
        path = settings.output_dir / "jobs" / job_id / f"legacy_{index}_vocal.mp3"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"ID3-legacy-stem")
        url = f"http://testserver/output/{path.relative_to(settings.output_dir)}"
        result.update(stems={"vocal": url}, stemUrls=[url], splitEnabled=True)
    job.save(settings.output_dir)
