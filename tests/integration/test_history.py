import asyncio
import json
from unittest.mock import AsyncMock

import httpx
import pytest
from starlette.requests import Request

from app.core.errors import CapacityExceededError
from app.main import GenerationJob, create_app, load_jobs
from app.services.prompt import PreparedPrompt
from app.services.stems import STEM_NAMES
from tests.helpers import make_orchestrator, make_settings


def client(app):
    return httpx.AsyncClient(transport=httpx.ASGITransport(app), base_url="http://testserver")


def events_request(app, path: str = "/api/jobs/active/events") -> Request:
    return Request(
        {
            "type": "http",
            "app": app,
            "headers": [],
            "scheme": "http",
            "server": ("testserver", 80),
            "path": path,
            "root_path": "",
            "query_string": b"",
            "method": "GET",
        }
    )


async def time_out(awaitable, *_args, **_kwargs):
    """让 SSE 循环立刻走一次 15 秒保活超时分支，测试不必真的等待。"""
    awaitable.close()
    raise asyncio.TimeoutError


def split_waveforms() -> dict[str, list[float]]:
    """拆轨会连同 "full" 一起重新提取波形，桩函数必须返回同样的音轨集合。"""
    return {"full": [0.25], **{name: [0.5] for name in STEM_NAMES}}


async def test_direct_history_restart(tmp_path):
    settings = make_settings(tmp_path)
    orchestrator = make_orchestrator(settings)
    # 生成路径只产出完整混音。
    orchestrator.stem_separator.split = AsyncMock(side_effect=AssertionError("no splitting"))
    app = create_app(settings, orchestrator)
    async with client(app) as http:
        result = (await http.post("/api/generate", json={"prompt": "rock"})).json()
        history = (await http.get("/api/jobs")).json()["jobs"]
    assert len(history) == 1
    assert history[0]["createdAt"] == result["createdAt"]
    assert history[0]["result"]["fullTrack"] == result["fullTrack"]
    stored = json.loads((settings.output_dir / "jobs" / result["jobId"] / "job.json").read_text())
    assert stored["diagnostics"]["prompt"] == "rock"
    assert stored["diagnostics"]["provider"] == "minimax_music"
    assert stored["diagnostics"]["durationSource"] == "provider"
    assert "effectiveDurationSeconds" not in stored["diagnostics"]
    assert "effectiveDurationMinutes" not in stored["diagnostics"]
    assert list((settings.output_dir / "jobs" / result["jobId"] / "song_1").glob("full_song_*"))
    orchestrator.stem_separator.split.assert_not_called()
    restarted = create_app(settings, orchestrator)
    async with client(restarted) as http:
        restored = (await http.get("/api/jobs")).json()["jobs"]
        assert len(restored) == 1
        assert (await http.get(result["fullTrack"])).content == b"ID3-full-audio"


async def test_history_listing_omits_waveform_bins(tmp_path):
    settings = make_settings(tmp_path)
    orchestrator = make_orchestrator(settings)
    app = create_app(settings, orchestrator)
    bins = [0.5] * 640

    async with client(app) as http:
        job_id = (await http.post("/api/generate", json={"prompt": "rock", "count": 2})).json()[
            "jobId"
        ]

    # 改写落盘记录，模拟带波形的历史任务；重启后按 job.json 重新加载。
    record = settings.output_dir / "jobs" / job_id / "job.json"
    stored = json.loads(record.read_text(encoding="utf-8"))
    stored["result"]["waveforms"] = {"full": bins}
    stored["result"]["alternatives"][0]["waveforms"] = {"full": bins}
    record.write_text(json.dumps(stored, ensure_ascii=False), encoding="utf-8")

    restarted = create_app(settings, orchestrator)
    async with client(restarted) as http:
        history = (await http.get("/api/jobs")).json()["jobs"][0]
        detail = (await http.get(f"/api/jobs/{job_id}")).json()

    # 历史列表只做投影，不携带波形数据；单任务详情仍然返回真实波形。
    assert history["result"]["waveforms"] == {}
    assert history["result"]["alternatives"][0]["waveforms"] == {}
    assert detail["result"]["waveforms"]["full"] == bins
    assert detail["result"]["alternatives"][0]["waveforms"]["full"] == bins
    # 投影只是响应层裁剪，落盘的波形不能被改写。
    assert json.loads(record.read_text(encoding="utf-8"))["result"]["waveforms"]["full"] == bins


async def test_history_listing_skips_rvc_fingerprint(tmp_path, monkeypatch):
    settings = make_settings(tmp_path)
    app = create_app(settings, make_orchestrator(settings))
    job = GenerationJob(job_id="cached", prompt="rock", status="succeeded", stage="completed")
    job.result = {
        "replacedVocal": "jobs/cached/song_1/vocal_rvc_vocal.wav",
        "_replacedVocalModel": "v1:old",
    }
    app.state.jobs[job.job_id] = job

    def fail(_settings):
        raise AssertionError("history listing must not hash RVC assets")

    monkeypatch.setattr("app.main._rvc_model_fingerprint", fail)
    async with client(app) as http:
        response = await http.get("/api/jobs")

    assert response.status_code == 200


async def test_job_detail_reuses_startup_rvc_fingerprint(tmp_path, monkeypatch):
    settings = make_settings(tmp_path)
    calls = []

    def fingerprint(_settings):
        calls.append(_settings)
        return "fingerprint"

    monkeypatch.setattr("app.main._rvc_model_fingerprint", fingerprint)
    app = create_app(settings, make_orchestrator(settings))
    job = GenerationJob(job_id="cached", prompt="rock", status="succeeded", stage="completed")
    job.result = {
        "success": True,
        "jobId": job.job_id,
        "prompt": job.prompt,
        "durationMinutes": "auto",
        "structuredPrompt": "rock",
        "lyrics": "test",
        "fullTrack": "jobs/cached/song_1/full.wav",
        "stems": {},
        "stemUrls": [],
        "waveforms": {},
        "splitEnabled": False,
        "debug": {},
        "count": 1,
        "alternatives": [],
        "replacedVocal": "jobs/cached/song_1/vocal_rvc_vocal.wav",
        "_replacedVocalModel": "fingerprint",
    }
    replaced = settings.output_dir / job.result["replacedVocal"]
    replaced.parent.mkdir(parents=True, exist_ok=True)
    replaced.write_bytes(b"RIFF-replaced")
    app.state.jobs[job.job_id] = job
    async with client(app) as http:
        first = await http.get(f"/api/jobs/{job.job_id}")
        second = await http.get(f"/api/jobs/{job.job_id}")

    assert first.json()["replaceStatus"] == second.json()["replaceStatus"] == "succeeded"
    assert calls == [settings]


async def test_split_refreshes_legacy_64_bin_full_waveform(tmp_path, monkeypatch):
    settings = make_settings(tmp_path)
    orchestrator = make_orchestrator(settings)
    app = create_app(settings, orchestrator)
    extract = AsyncMock(
        return_value={
            "full": [0.25] * 640,
            **{name: [0.5] * 640 for name in ("vocal", "drums", "bass", "other")},
        }
    )
    monkeypatch.setattr("app.main.extract_waveforms", extract)

    async with client(app) as http:
        job_id = (await http.post("/api/generate", json={"prompt": "rock"})).json()["jobId"]
        # 旧版本持久化的任务：只有 64 个 bin 的完整混音波形。
        app.state.jobs[job_id].result["waveforms"] = {"full": [0.25] * 64}
        assert (await http.post(f"/api/jobs/{job_id}/split?song=0")).status_code == 202
        await app.state.jobs[job_id].split_task
        completed = (await http.get(f"/api/jobs/{job_id}")).json()

    waveforms = completed["result"]["waveforms"]
    assert set(waveforms) == {"full", "vocal", "drums", "bass", "other"}
    # 所有车道必须共享同一个 bin 数量，否则客户端的 x 轴会错位。
    assert {len(values) for values in waveforms.values()} == {640}


async def test_split_keeps_stored_waveforms_when_extraction_fails(tmp_path, monkeypatch):
    settings = make_settings(tmp_path)
    orchestrator = make_orchestrator(settings)
    app = create_app(settings, orchestrator)
    monkeypatch.setattr("app.main.extract_waveforms", AsyncMock(return_value={}))

    async with client(app) as http:
        job_id = (await http.post("/api/generate", json={"prompt": "rock"})).json()["jobId"]
        app.state.jobs[job_id].result["waveforms"] = {"full": [0.25] * 64}
        assert (await http.post(f"/api/jobs/{job_id}/split?song=0")).status_code == 202
        await app.state.jobs[job_id].split_task
        completed = (await http.get(f"/api/jobs/{job_id}")).json()

    assert completed["result"]["splitEnabled"] is True
    assert completed["result"]["waveforms"] == {"full": [0.25] * 64}


async def test_history_song_starts_async_split_and_updates_result(tmp_path, monkeypatch):
    settings = make_settings(tmp_path)
    orchestrator = make_orchestrator(settings)
    app = create_app(settings, orchestrator)

    async with client(app) as http:
        generated = (await http.post("/api/generate", json={"prompt": "rock"})).json()
        job_id = generated["jobId"]
        original_split = orchestrator.stem_separator.split
        started, release = asyncio.Event(), asyncio.Event()

        async def split(*args, **kwargs):
            started.set()
            await release.wait()
            return await original_split(*args, **kwargs)

        monkeypatch.setattr(orchestrator.stem_separator, "split", split)
        extract = AsyncMock(return_value=split_waveforms())
        monkeypatch.setattr("app.main.extract_waveforms", extract)
        accepted = await http.post(f"/api/jobs/{job_id}/split?song=0")
        await asyncio.wait_for(started.wait(), 2)
        running = (await http.get(f"/api/jobs/{job_id}")).json()
        history_running = (await http.get("/api/jobs")).json()["jobs"][0]
        duplicate = await http.post(f"/api/jobs/{job_id}/split?song=0")
        release.set()
        await app.state.jobs[job_id].split_task
        completed = (await http.get(f"/api/jobs/{job_id}")).json()
        cache_guard = AsyncMock(side_effect=AssertionError("cached split must not run again"))
        monkeypatch.setattr(orchestrator.stem_separator, "split", cache_guard)
        cached = await http.post(f"/api/jobs/{job_id}/split?song=0")

        for stem in list(completed["result"]["stems"]):
            assert (
                await http.delete(
                    f"/api/jobs/{job_id}/stems/{stem}",
                    headers={"X-Audio-Revision": "1"},
                )
            ).status_code == 204
        assert (await http.get(f"/api/jobs/{job_id}")).json()["result"]["stems"] == {}
        resplit = await http.post(f"/api/jobs/{job_id}/split?song=0")
        await app.state.jobs[job_id].split_task
        resplit_result = (await http.get(f"/api/jobs/{job_id}")).json()

    assert accepted.status_code == duplicate.status_code == 202
    assert (
        running["operationStatus"],
        running["operationStage"],
        running["operationProgress"],
    ) == (
        "running",
        "splitting",
        25,
    )
    assert (running["status"], running["stage"], running["progress"]) == (
        "running",
        "splitting",
        25,
    )
    assert running["splitSong"] == running["operationSong"] == 0
    assert running["operationMessage"] == "正在分离音轨"
    assert history_running["status"] == "succeeded"
    assert "splitStatus" not in history_running
    assert history_running["result"]["waveforms"] == {}
    assert completed["status"] == "succeeded"
    assert completed["result"]["splitEnabled"] is True
    assert sorted(completed["result"]["stems"]) == sorted(STEM_NAMES)
    assert completed["result"]["waveforms"]["vocal"] == [0.5]
    # 拆轨必须重新提取 "full"，否则客户端会拿到长度不一致的波形车道。
    assert set(extract.await_args.args[0]) == {"full", *STEM_NAMES}
    assert set(completed["result"]["waveforms"]) == {"full", *STEM_NAMES}
    assert cached.status_code == 200
    assert cached.json()["status"] == "succeeded"
    cache_guard.assert_not_called()
    assert resplit.status_code == 200
    assert resplit_result["result"]["stems"] == {}
    # 首次拆轨推进文件版本；删除与缓存命中都不再改动它。
    assert resplit_result["result"]["audioRevision"] == 1
    assert len(app.state.jobs[job_id].deleted_stems) == len(STEM_NAMES)


async def test_split_reserves_job_before_capacity_await(tmp_path, monkeypatch):
    settings = make_settings(tmp_path, max_concurrent_generations=3)
    orchestrator = make_orchestrator(settings)
    app = create_app(settings, orchestrator)
    original_acquire = orchestrator.capacity.acquire
    entered, release = asyncio.Event(), asyncio.Event()
    acquire_calls = 0

    async def acquire():
        nonlocal acquire_calls
        acquire_calls += 1
        entered.set()
        await release.wait()
        await original_acquire()

    monkeypatch.setattr(
        "app.main.extract_waveforms",
        AsyncMock(return_value=split_waveforms()),
    )
    async with client(app) as http:
        job_id = (await http.post("/api/generate", json={"prompt": "rock", "count": 2})).json()[
            "jobId"
        ]
        monkeypatch.setattr(orchestrator.capacity, "acquire", acquire)
        first = asyncio.create_task(http.post(f"/api/jobs/{job_id}/split?song=0"))
        await asyncio.wait_for(entered.wait(), 2)
        duplicate = await http.post(f"/api/jobs/{job_id}/split?song=0")
        other_song = await http.post(f"/api/jobs/{job_id}/split?song=1")
        release.set()
        accepted = await first
        await app.state.jobs[job_id].split_task

    assert accepted.status_code == duplicate.status_code == 202
    assert acquire_calls == 1
    assert other_song.status_code == 409
    assert "第 1 首" in other_song.json()["message"]
    assert app.state.jobs[job_id].status == "succeeded"
    assert app.state.jobs[job_id].result["stems"]
    assert app.state.jobs[job_id].result["alternatives"][0]["stems"] == {}


async def test_split_failure_and_cancel_preserve_completed_generation(tmp_path, monkeypatch):
    settings = make_settings(tmp_path)
    orchestrator = make_orchestrator(settings)
    app = create_app(settings, orchestrator)

    async with client(app) as http:
        job_id = (await http.post("/api/generate", json={"prompt": "rock"})).json()["jobId"]
        monkeypatch.setattr(
            orchestrator.stem_separator,
            "split",
            AsyncMock(side_effect=RuntimeError("secret /tmp/model failed")),
        )
        assert (await http.post(f"/api/jobs/{job_id}/split?song=0")).status_code == 202
        await app.state.jobs[job_id].split_task
        failed = (await http.get(f"/api/jobs/{job_id}")).json()

    assert failed["status"] == "succeeded"
    assert failed["operationStatus"] == "failed"
    assert failed["splitStatus"] == "failed"
    assert failed["splitSong"] == 0
    assert failed["splitError"] == "音轨分离失败，请检查服务配置后重试。"
    assert "secret" not in json.dumps(failed)
    assert app.state.jobs[job_id].status == "succeeded"
    assert load_jobs(settings.output_dir)[job_id].status == "succeeded"
    async with client(app) as http:
        history_job = (await http.get("/api/jobs")).json()["jobs"][0]
    assert "splitError" not in history_job
    async with client(create_app(settings, orchestrator)) as http:
        assert (await http.get(f"/api/jobs/{job_id}")).json()["status"] == "succeeded"

    started = asyncio.Event()

    async def blocking_split(*_args, **_kwargs):
        started.set()
        await asyncio.Event().wait()

    monkeypatch.setattr(orchestrator.stem_separator, "split", blocking_split)
    async with client(app) as http:
        await http.post(f"/api/jobs/{job_id}/split?song=0")
        await asyncio.wait_for(started.wait(), 2)
        await http.patch(f"/api/jobs/{job_id}", json={"status": "cancelled"})
        assert app.state.jobs[job_id].split_status == "cancelled"
        await asyncio.sleep(0)
        cancelled = (await http.get(f"/api/jobs/{job_id}")).json()
        cancelled_task = app.state.jobs[job_id].split_task
        # cancelled 是操作状态；song=0 明确重试刚才取消的同一首歌。
        retry = await http.post(f"/api/jobs/{job_id}/split?song=0")
        assert retry.status_code == 202
        assert retry.json()["splitStatus"] == "pending"
        assert app.state.jobs[job_id].split_task is not cancelled_task
        assert not app.state.jobs[job_id].split_task.done()
        await http.patch(f"/api/jobs/{job_id}", json={"status": "cancelled"})
        await asyncio.sleep(0)

    assert cancelled["status"] == "succeeded"
    assert cancelled["operationStatus"] == "cancelled"
    assert cancelled["splitStatus"] == "cancelled"
    assert app.state.jobs[job_id].status == "succeeded"
    assert load_jobs(settings.output_dir)[job_id].status == "succeeded"


async def test_split_rejects_invalid_or_unavailable_jobs(tmp_path, monkeypatch):
    settings = make_settings(tmp_path)
    orchestrator = make_orchestrator(settings)
    app = create_app(settings, orchestrator)
    async with client(app) as http:
        assert (await http.post("/api/jobs/missing/split")).status_code == 404
        job_id = (await http.post("/api/generate", json={"prompt": "rock"})).json()["jobId"]
        job = app.state.jobs[job_id]
        full_track = job.result.pop("fullTrack")
        assert (await http.post(f"/api/jobs/{job_id}/split")).status_code == 409
        job.result["fullTrack"] = full_track
        full_file = next(settings.output_dir.rglob(f"full_song_{job_id}*"))
        full_file.unlink()
        assert (await http.post(f"/api/jobs/{job_id}/split")).status_code == 409
        full_file.write_bytes(b"ID3-full-audio")
        monkeypatch.setattr(
            orchestrator.capacity,
            "acquire",
            AsyncMock(side_effect=CapacityExceededError("full")),
        )
        limited = await http.post(f"/api/jobs/{job_id}/split")

    assert limited.status_code == 429
    assert job.split_status is None


async def test_history_urls_do_not_persist_request_host(tmp_path):
    settings = make_settings(tmp_path)
    app = create_app(settings, make_orchestrator(settings))
    async with client(app) as http:
        result = (
            await http.post(
                "/api/generate", json={"prompt": "rock"}, headers={"Host": "attacker.test"}
            )
        ).json()
        stored = json.loads(
            (settings.output_dir / "jobs" / result["jobId"] / "job.json").read_text()
        )
        history = (await http.get("/api/jobs")).json()["jobs"]

    assert result["fullTrack"].startswith("http://attacker.test/output/")
    assert stored["result"]["fullTrack"].endswith(f"{result['jobId']}_1.mp3")
    assert "://" not in stored["result"]["fullTrack"]
    assert history[0]["result"]["fullTrack"].startswith("http://testserver/output/")


async def test_invalid_persisted_output_url_does_not_break_history(tmp_path):
    settings = make_settings(tmp_path)
    app = create_app(settings, make_orchestrator(settings))
    async with client(app) as http:
        job_id = (await http.post("/api/generate", json={"prompt": "rock"})).json()["jobId"]

    job_file = settings.output_dir / "jobs" / job_id / "job.json"
    stored = json.loads(job_file.read_text())
    stored["result"]["fullTrack"] = "/etc/passwd"
    job_file.write_text(json.dumps(stored))

    async with client(create_app(settings, make_orchestrator(settings))) as http:
        history = await http.get("/api/jobs")
        detail = await http.get(f"/api/jobs/{job_id}")

    assert history.status_code == detail.status_code == 200
    assert detail.json()["result"]["fullTrack"] == "/etc/passwd"


def test_interrupted_and_corrupt_metadata(tmp_path):
    settings = make_settings(tmp_path)
    job = GenerationJob(
        job_id="interrupted", prompt="rock", status="running", step=4, total_steps=30
    )
    job.save(settings.output_dir)
    corrupt = settings.output_dir / "jobs" / "corrupt" / "job.json"
    corrupt.parent.mkdir(parents=True)
    corrupt.write_text("{broken")
    restored = load_jobs(settings.output_dir)
    assert restored["interrupted"].status == "failed"
    assert restored["interrupted"].step is None
    assert "重启" in restored["interrupted"].error
    assert (
        json.loads((corrupt.parent.parent / "interrupted" / "job.json").read_text())["status"]
        == "failed"
    )
    assert corrupt.read_text() == "{broken"
    assert not list(settings.output_dir.rglob("*.tmp"))


async def test_job_forwards_actual_provider_counts(tmp_path):
    settings = make_settings(tmp_path)
    orchestrator = make_orchestrator(settings)
    reached, release = asyncio.Event(), asyncio.Event()
    original = orchestrator.music_provider.generate

    async def generate(*args, progress=None, **kwargs):
        await progress("denoising", 7, 90)
        reached.set()
        await release.wait()
        return await original(*args, **kwargs)

    orchestrator.music_provider.generate = generate
    app = create_app(settings, orchestrator)
    async with client(app) as http:
        job_id = (await http.post("/api/jobs", json={"prompt": "rock"})).json()["jobId"]
        await asyncio.wait_for(reached.wait(), 2)
        row = (await http.get(f"/api/jobs/{job_id}")).json()
        assert (row["stage"], row["step"], row["totalSteps"], row["progress"]) == (
            "denoising",
            7,
            90,
            None,
        )
        assert row["title"] == "测试歌名"
        assert (await http.get("/api/jobs")).json()["jobs"][0]["title"] == "测试歌名"
        assert load_jobs(settings.output_dir)[job_id].title == "测试歌名"
        endpoint = next(
            route.endpoint for route in app.routes if route.path == "/api/jobs/{job_id}/events"
        )
        events = await endpoint(job_id, events_request(app, f"/api/jobs/{job_id}/events"))
        frame = await anext(events.body_iterator)
        assert json.loads(frame.removeprefix("data: "))["title"] == "测试歌名"
        await events.body_iterator.aclose()
        release.set()
        await app.state.jobs[job_id].task
        assert load_jobs(settings.output_dir)[job_id].status == "succeeded"


async def test_generated_title_survives_music_failure(tmp_path):
    settings = make_settings(tmp_path)
    orchestrator = make_orchestrator(settings)
    orchestrator.music_provider.generate = AsyncMock(side_effect=RuntimeError("provider died"))
    app = create_app(settings, orchestrator)
    async with client(app) as http:
        job_id = (await http.post("/api/jobs", json={"prompt": "rock"})).json()["jobId"]
        await app.state.jobs[job_id].task
        row = (await http.get(f"/api/jobs/{job_id}")).json()
    assert row["status"] == "failed"
    assert row["title"] == "测试歌名"
    assert load_jobs(settings.output_dir)[job_id].title == "测试歌名"


async def test_generated_title_survives_music_cancellation(tmp_path):
    settings = make_settings(tmp_path)
    orchestrator = make_orchestrator(settings)
    reached = asyncio.Event()

    async def generate(*args, **kwargs):
        reached.set()
        await asyncio.Event().wait()

    orchestrator.music_provider.generate = generate
    app = create_app(settings, orchestrator)
    async with client(app) as http:
        job_id = (await http.post("/api/jobs", json={"prompt": "rock"})).json()["jobId"]
        await asyncio.wait_for(reached.wait(), 2)
        cancelled = await http.patch(f"/api/jobs/{job_id}", json={"status": "cancelled"})
        await app.state.jobs[job_id].task
    assert cancelled.json()["status"] == "cancelled"
    assert cancelled.json()["title"] == "测试歌名"
    assert load_jobs(settings.output_dir)[job_id].title == "测试歌名"


async def test_history_preserves_failed_jobs_and_events_send_done(tmp_path):
    settings = make_settings(tmp_path)
    app = create_app(settings, make_orchestrator(settings))
    failed = GenerationJob(job_id="failed", prompt="bad", status="failed", stage="failed")
    failed.error = failed.message = "provider died"
    app.state.jobs[failed.job_id] = failed

    async with client(app) as http:
        assert (await http.get("/api/jobs")).json()["jobs"][0]["error"] == "provider died"
        assert (await http.get("/api/jobs/failed")).json()["error"] == "provider died"
        events = await http.get("/api/jobs/failed/events")

    assert events.headers["content-type"].startswith("text/event-stream")
    assert events.text.startswith("event: done\ndata: ")
    assert json.loads(events.text.split("data: ", 1)[1])["status"] == "failed"


async def test_unstarted_job_events_do_not_register_subscriber(tmp_path):
    settings = make_settings(tmp_path)
    app = create_app(settings, make_orchestrator(settings))
    app.state.jobs["active"] = GenerationJob(
        job_id="active", prompt="rock", status="running", stage="music"
    )
    endpoint = next(
        route.endpoint for route in app.routes if route.path == "/api/jobs/{job_id}/events"
    )
    request = Request({"type": "http", "app": app, "headers": []})

    await endpoint("active", request)

    assert app.state.job_subscribers == {}


async def test_job_events_keep_alive_after_timeout(tmp_path, monkeypatch):
    settings = make_settings(tmp_path)
    app = create_app(settings, make_orchestrator(settings))
    job = GenerationJob(job_id="active", prompt="rock", status="running", stage="music")
    app.state.jobs[job.job_id] = job
    endpoint = next(
        route.endpoint for route in app.routes if route.path == "/api/jobs/{job_id}/events"
    )
    monkeypatch.setattr(asyncio, "wait_for", time_out)
    response = await endpoint("active", events_request(app))
    assert (await anext(response.body_iterator)).startswith("data: ")
    assert await anext(response.body_iterator) == ": keep-alive\n\n"
    await response.body_iterator.aclose()


@pytest.mark.parametrize("stage", ["receiving_audio", "song_completed"])
async def test_job_events_only_publish_waveforms_on_completion(tmp_path, monkeypatch, stage):
    settings = make_settings(tmp_path)
    app = create_app(settings, make_orchestrator(settings))
    bins = [0.5] * 640
    job = GenerationJob(job_id="active", prompt="rock", status="running", stage=stage)
    job.result = {"fullTrack": "song_1/full.mp3", "waveforms": {"full": bins}}
    app.state.jobs[job.job_id] = job
    endpoint = next(
        route.endpoint for route in app.routes if route.path == "/api/jobs/{job_id}/events"
    )
    monkeypatch.setattr(asyncio, "wait_for", time_out)
    response = await endpoint("active", events_request(app))

    progress = await anext(response.body_iterator)
    assert await anext(response.body_iterator) == ": keep-alive\n\n"
    job.status = "succeeded"
    job.stage = "completed"
    done = await anext(response.body_iterator)
    await response.body_iterator.aclose()

    payload = json.loads(progress.removeprefix("data: "))
    assert payload["status"] == "running"
    assert payload["result"]["waveforms"] == ({"full": bins} if stage == "song_completed" else {})
    assert done.startswith("event: done\ndata: ")
    completed = json.loads(done.split("data: ", 1)[1])
    assert completed["status"] == "succeeded"
    assert completed["result"]["waveforms"] == {"full": bins}


async def test_job_events_omit_waveforms_while_split_runs(tmp_path, monkeypatch):
    """分轨会把 status 覆盖成 running，所以 SSE 要一直推流到分轨收尾为止。"""
    settings = make_settings(tmp_path)
    app = create_app(settings, make_orchestrator(settings))
    bins = [0.5] * 640
    job = GenerationJob(job_id="active", prompt="rock", status="succeeded", stage="completed")
    job.result = {"fullTrack": "song_1/full.mp3", "waveforms": {"full": bins}}
    job.split_status = "running"
    job.split_stage = "waveform"
    job.split_progress = 50
    job.split_message = "正在提取波形"
    app.state.jobs[job.job_id] = job
    endpoint = next(
        route.endpoint for route in app.routes if route.path == "/api/jobs/{job_id}/events"
    )
    monkeypatch.setattr(asyncio, "wait_for", time_out)
    response = await endpoint("active", events_request(app))

    splitting = await anext(response.body_iterator)
    assert await anext(response.body_iterator) == ": keep-alive\n\n"
    job.split_status = "succeeded"
    job.split_stage = "completed"
    job.split_progress = 100
    done = await anext(response.body_iterator)
    await response.body_iterator.aclose()

    payload = json.loads(splitting.removeprefix("data: "))
    assert payload["status"] == payload["operationStatus"] == "running"
    assert (payload["stage"], payload["progress"], payload["message"]) == (
        payload["operationStage"],
        payload["operationProgress"],
        payload["operationMessage"],
    )
    assert (payload["operationStage"], payload["operationProgress"]) == ("waveform", 50)
    assert payload["operationMessage"] == "正在提取波形"
    assert payload["splitStatus"] == "running"
    # 分轨期间 result 不会变化，波形只随 done 帧下发。
    assert payload["result"]["waveforms"] == {}
    completed = json.loads(done.split("data: ", 1)[1])
    assert completed["status"] == "succeeded"
    assert completed["result"]["waveforms"] == {"full": bins}


async def test_direct_failure_is_persisted(tmp_path):
    settings = make_settings(tmp_path)
    orchestrator = make_orchestrator(settings)
    orchestrator.music_provider.generate = AsyncMock(side_effect=RuntimeError("provider died"))
    app = create_app(settings, orchestrator)
    async with client(app) as http:
        assert (await http.post("/api/generate", json={"prompt": "rock"})).status_code == 500
    row = next(iter(load_jobs(settings.output_dir).values()))
    assert row.status == "failed"
    assert row.error == "provider died"


async def test_first_split_resets_previous_project_state(tmp_path, monkeypatch):
    """新工程不能恢复上一代工程的人声替换。"""
    settings = make_settings(tmp_path)
    orchestrator = make_orchestrator(settings)
    app = create_app(settings, orchestrator)
    monkeypatch.setattr("app.main.extract_waveforms", AsyncMock(return_value=split_waveforms()))
    async with client(app) as http:
        job_id = (await http.post("/api/generate", json={"prompt": "rock"})).json()["jobId"]
        result = app.state.jobs[job_id].result
        result["mixedTrack"] = f"jobs/{job_id}/song_1/demo_rvc_mix.wav"
        result["replacedVocal"] = f"jobs/{job_id}/song_1/demo_rvc_vocal.wav"
        replaced_path = settings.output_dir / result["replacedVocal"]
        replaced_path.write_bytes(b"RIFF" + b"\0" * 32)
        result["_replacedVocalModel"] = "model-v1"
        result["playback"] = {"mixedTrack": "old-mix.mp3", "replacedVocal": "old-vocal.mp3"}
        result["waveforms"] = {"full": [0.25], "mix": [0.75] * 640, "replaced": [0.4] * 640}

        assert (await http.post(f"/api/jobs/{job_id}/split?song=0")).status_code == 202
        await app.state.jobs[job_id].split_task
        completed = (await http.get(f"/api/jobs/{job_id}")).json()["result"]
        restored = await http.request(
            "PUT",
            "/api/voice/result",
            data={
                "filename": replaced_path.name,
                "job_id": job_id,
                "song": "0",
                # 首轮拆轨已经把文件版本推进到 1，旧版本号的撤回必须被拒。
                "audio_revision": "0",
            },
        )

    assert "mixedTrack" not in completed and "replacedVocal" not in completed
    assert "mixedTrack" not in completed["playback"]
    assert "replacedVocal" not in completed["playback"]
    assert set(completed["waveforms"]) == {"full", *STEM_NAMES}
    assert completed["audioRevision"] == 1
    # 旧的工程控制状态字段不再出现在任务结果里。
    assert "editorState" not in completed and "projectRevision" not in completed
    assert app.state.jobs[job_id].deleted_replaced_vocals == {}
    assert restored.status_code == 409


async def test_first_split_discards_previous_project_undo(tmp_path, monkeypatch):
    settings = make_settings(tmp_path)
    app = create_app(settings, make_orchestrator(settings))
    monkeypatch.setattr("app.main.extract_waveforms", AsyncMock(return_value=split_waveforms()))
    async with client(app) as http:
        job_id = (await http.post("/api/generate", json={"prompt": "rock"})).json()["jobId"]
        job = app.state.jobs[job_id]
        job.deleted_replaced_vocals["0"] = {"url": "previously-deleted.wav"}
        assert (await http.post(f"/api/jobs/{job_id}/split?song=0")).status_code == 202
        await job.split_task
    assert job.deleted_replaced_vocals == {}


async def test_job_title_is_listed_and_survives_restart(tmp_path):
    settings = make_settings(tmp_path)
    app = create_app(settings, make_orchestrator(settings))
    async with client(app) as http:
        job_id = (
            await http.post("/api/jobs", json={"prompt": "rock", "title": " 毕业后的狂响 "})
        ).json()["jobId"]
        await app.state.jobs[job_id].task
        history = (await http.get("/api/jobs")).json()["jobs"]
    assert history[0]["title"] == "毕业后的狂响"
    assert load_jobs(settings.output_dir)[job_id].title == "毕业后的狂响"


async def test_generated_title_is_listed_and_survives_restart(tmp_path):
    settings = make_settings(tmp_path)
    app = create_app(settings, make_orchestrator(settings))
    async with client(app) as http:
        job_id = (await http.post("/api/jobs", json={"prompt": "rock", "title": "   "})).json()[
            "jobId"
        ]
        await app.state.jobs[job_id].task
        history = (await http.get("/api/jobs")).json()["jobs"]
    assert history[0]["title"] == "测试歌名"
    assert load_jobs(settings.output_dir)[job_id].title == "测试歌名"


async def test_job_succeeds_when_expander_has_no_valid_title(tmp_path):
    settings = make_settings(tmp_path)
    orchestrator = make_orchestrator(settings)
    orchestrator.prompt_expander.prepare = AsyncMock(
        return_value=PreparedPrompt("[Genre: Test]", "[Verse]\n测试歌词", None)
    )
    app = create_app(settings, orchestrator)
    async with client(app) as http:
        job_id = (await http.post("/api/jobs", json={"prompt": "rock"})).json()["jobId"]
        await app.state.jobs[job_id].task
        job = (await http.get(f"/api/jobs/{job_id}")).json()
    assert job["status"] == "succeeded"
    assert job["title"] is None
    assert load_jobs(settings.output_dir)[job_id].title is None
