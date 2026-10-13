import asyncio
import json
import threading
from unittest.mock import AsyncMock

import pytest

from app.core.errors import GenerationError
from app.main import JOB_EVENT_QUEUE_SIZE, _publish_job, create_app
from tests.helpers import (
    BlockingVoiceEngine,
    ReadyVoiceEngine,
    client,
    events_request,
    make_orchestrator,
    make_settings,
    seed_job,
)

STAGES = {
    "split": [
        ("starting_split", "正在读取音频"),
        ("splitting", "正在分离音轨"),
        ("waveform", "正在提取波形"),
        ("preview", "正在导出分轨结果"),
    ],
    "replace": [
        ("preparing_vocal", "正在准备输入音频"),
        ("replacing_vocal", "正在转换人声音色"),
        ("creating_replacement", "正在创建替换音轨"),
        ("exporting_replacement", "正在导出替换结果"),
    ],
}


def operation_state(payload, operation):
    return tuple(
        payload.get(key)
        for key in (
            "status",
            "stage",
            "progress",
            "message",
            "operation",
            "operationStatus",
            "operationSong",
            "operationStage",
            "operationProgress",
            "operationMessage",
            f"{operation}Status",
            f"{operation}Song",
            f"{operation}Error",
        )
    )


async def stream(app, job):
    endpoint = next(
        route.endpoint for route in app.routes if route.path == "/api/jobs/{job_id}/events"
    )
    return (await endpoint(job.job_id, events_request(app, job.job_id))).body_iterator


async def next_payload(events):
    frame = await asyncio.wait_for(anext(events), 2)
    return frame, json.loads(frame.split("data: ", 1)[1])


@pytest.mark.parametrize("operation", STAGES)
async def test_four_stages_follow_work_and_post_get_sse_agree(tmp_path, monkeypatch, operation):
    settings = make_settings(tmp_path)
    orchestrator = make_orchestrator(settings)
    engine = BlockingVoiceEngine()
    app = create_app(settings, orchestrator, engine)
    job, _ = seed_job(app, settings)
    if operation == "split":
        job.result["stems"] = {}
        job.result["splitEnabled"] = False
    reached = [asyncio.Event() for _ in range(4)]
    release = [asyncio.Event() for _ in range(4)]
    read_release = threading.Event()
    loop = asyncio.get_running_loop()

    def read_input(_path):
        assert getattr(job, f"{operation}_stage") == STAGES[operation][0][0]
        loop.call_soon_threadsafe(reached[0].set)
        assert read_release.wait(2)

    async def wait_at(index):
        assert getattr(job, f"{operation}_stage") == STAGES[operation][index][0]
        reached[index].set()
        await release[index].wait()

    original_split = orchestrator.stem_separator.split

    async def split(*args):
        await wait_at(1)
        return await original_split(*args)

    async def convert(_input, output, **_params):
        await wait_at(1)
        output.write_bytes(b"RIFF-replaced")

    async def waveforms(paths, *, progress=None):
        await wait_at(2)
        if progress:
            for finished in range(1, len(paths) + 1):
                progress(finished, len(paths))
        return {name: [0.5] for name in paths}

    async def preview(*_args):
        await wait_at(3)
        return None

    monkeypatch.setattr("app.main._read_operation_input", read_input)
    monkeypatch.setattr(orchestrator.stem_separator, "split", split)
    monkeypatch.setattr(engine, "convert", convert)
    monkeypatch.setattr("app.main.extract_waveforms", waveforms)
    monkeypatch.setattr("app.main.make_playback_mp3", preview)
    async with client(app) as http:
        url = f"/api/jobs/{job.job_id}"
        post = await http.post(f"{url}/{operation}")
        initial = (await http.get(url)).json()
        assert operation_state(post.json(), operation) == operation_state(initial, operation)
        assert initial["operationProgress"] == 0
        assert (initial["status"], initial["stage"], initial["progress"]) == (
            "pending",
            STAGES[operation][0][0],
            0,
        )
        events = await stream(app, job)
        _, first = await next_payload(events)
        seen = []
        try:
            for index, (stage, message) in enumerate(STAGES[operation]):
                await asyncio.wait_for(reached[index].wait(), 2)
                detail = (await http.get(url)).json()
                duplicate = (await http.post(f"{url}/{operation}")).json()
                assert operation_state(duplicate, operation) == operation_state(detail, operation)
                payload = first if index == 0 else None
                while payload is None or operation_state(payload, operation) != operation_state(
                    detail, operation
                ):
                    _, payload = await next_payload(events)
                assert operation_state(payload, operation) == operation_state(detail, operation)
                assert (
                    detail["operationStatus"],
                    detail["operationMessage"],
                    detail[f"{operation}Song"],
                ) == (
                    "running",
                    message,
                    0,
                )
                assert (
                    detail["status"],
                    detail["stage"],
                    detail["progress"],
                    detail["message"],
                ) == (
                    detail["operationStatus"],
                    detail["operationStage"],
                    detail["operationProgress"],
                    detail["operationMessage"],
                )
                assert detail["operationSong"] == detail[f"{operation}Song"] == 0
                seen.append(stage)
                if index == 0:
                    read_release.set()
                else:
                    release[index].set()
            await getattr(job, f"{operation}_task")
            frames = []
            while True:
                frame, payload = await next_payload(events)
                frames.append(payload)
                if frame.startswith("event: done"):
                    break
            detail = (await http.get(url)).json()
            assert operation_state(payload, operation) == operation_state(detail, operation)
            assert detail["operationStatus"] == detail[f"{operation}Status"] == "succeeded"
            assert (detail["operationStage"], detail["operationProgress"]) == ("completed", 100)
            assert all(row["operationProgress"] < 100 for row in frames[:-1])
            if operation == "split":
                # 六轨导出按实际完成数量递增，最后一轨结束仍只能报 99。
                assert [
                    row["operationProgress"] for row in frames if row["operationStage"] == "preview"
                ] == [
                    79,
                    83,
                    87,
                    91,
                    95,
                    99,
                ]
            assert seen == [row[0] for row in STAGES[operation]]
            assert orchestrator.capacity.active == 0
            assert job.status == "succeeded"  # 主生成任务及历史仍保留成功。
        finally:
            read_release.set()
            for gate in release:
                gate.set()
            await events.aclose()


@pytest.mark.parametrize("operation", STAGES)
@pytest.mark.parametrize("outcome", ["succeeded", "failed", "cancelled"])
async def test_operation_terminal_and_cache_are_consistent(
    tmp_path, monkeypatch, operation, outcome
):
    settings = make_settings(tmp_path)
    orchestrator = make_orchestrator(settings)
    engine = BlockingVoiceEngine()
    app = create_app(settings, orchestrator, engine)
    job, _ = seed_job(app, settings)
    if operation == "split":
        job.result["stems"] = {}
        job.result["splitEnabled"] = False
    started, release = asyncio.Event(), asyncio.Event()
    original_split = orchestrator.stem_separator.split

    async def split(*args):
        started.set()
        await release.wait()
        if outcome == "failed":
            raise RuntimeError("split failed")
        return await original_split(*args)

    async def convert(_input, output, **_params):
        started.set()
        await release.wait()
        if outcome == "failed":
            raise RuntimeError("replacement failed")
        output.write_bytes(b"RIFF-replaced")

    monkeypatch.setattr(orchestrator.stem_separator, "split", split)
    monkeypatch.setattr(engine, "convert", convert)
    monkeypatch.setattr("app.main.extract_waveforms", AsyncMock(return_value={}))
    monkeypatch.setattr("app.main.make_playback_mp3", AsyncMock(return_value=None))
    async with client(app) as http:
        url = f"/api/jobs/{job.job_id}"
        await http.post(f"{url}/{operation}")
        await asyncio.wait_for(started.wait(), 2)
        events = await stream(app, job)
        await next_payload(events)
        if outcome == "cancelled":
            patch = (await http.patch(url, json={"status": "cancelled"})).json()
            frame, payload = await next_payload(events)
            assert frame.startswith("event: done")
            assert operation_state(patch, operation) == operation_state(payload, operation)
            if operation == "replace":
                assert not job.replace_task.done()
                assert orchestrator.capacity.active == 1
            release.set()
        else:
            release.set()
        await getattr(job, f"{operation}_task")
        if outcome != "cancelled":
            # 快速阶段即便已全部完成，已连接 SSE 的客户端仍能收到完整顺序。
            stages = []
            while True:
                frame, payload = await next_payload(events)
                if frame.startswith("event: done"):
                    break
                if not stages or payload["operationStage"] != stages[-1]:
                    stages.append(payload["operationStage"])
            if outcome == "succeeded":
                assert stages == [row[0] for row in STAGES[operation]][2:]
        detail = (await http.get(url)).json()
        assert operation_state(detail, operation) == operation_state(payload, operation)
        assert detail["operationStatus"] == detail[f"{operation}Status"] == outcome
        history = (await http.get("/api/jobs")).json()["jobs"][0]
        stored = json.loads((settings.output_dir / "jobs" / job.job_id / "job.json").read_text())
        restarted = create_app(settings, make_orchestrator(settings), ReadyVoiceEngine())
        async with client(restarted) as restarted_http:
            restored = (await restarted_http.get(url)).json()
        for row in (detail, payload, history, stored, restored):
            assert (row["status"], row["stage"], row["progress"], row["message"]) == (
                "succeeded",
                "completed",
                100,
                "音乐生成完成",
            )
        assert (await http.get(detail["result"]["fullTrack"])).status_code == 200
        assert detail["operationStage"] == ("completed" if outcome == "succeeded" else outcome)
        assert detail["operationProgress"] == (100 if outcome == "succeeded" else None)
        assert orchestrator.capacity.active == 0
        await events.aclose()
        if outcome == "succeeded":
            # 重启后的缓存也必须补齐操作成功状态与 song，不能依赖主生成任务成功。
            guard = AsyncMock(side_effect=AssertionError("cache must not perform work"))
            monkeypatch.setattr(restarted.state.voice_engine, "convert", guard)
            monkeypatch.setattr(restarted.state.orchestrator.stem_separator, "split", guard)
            async with client(restarted) as cached_http:
                cached = await cached_http.post(f"{url}/{operation}")
                cached_detail = (await cached_http.get(url)).json()
                cached_events = await cached_http.get(f"{url}/events")
            assert cached.status_code == 200
            assert cached_detail["operationSong"] == cached_detail[f"{operation}Song"] == 0
            assert cached_detail[f"{operation}Status"] == "succeeded"
            cached_payload = json.loads(cached_events.text.split("data: ", 1)[1])
            assert (
                operation_state(cached.json(), operation)
                == operation_state(cached_detail, operation)
                == operation_state(cached_payload, operation)
            )
            assert cached_detail["operationMessage"] == (
                "音轨分离完成" if operation == "split" else "人声替换完成"
            )
            guard.assert_not_called()


@pytest.mark.parametrize("stage", ["creating_replacement", "exporting_replacement"])
async def test_replace_cancel_during_postprocessing_stays_cancelled(tmp_path, monkeypatch, stage):
    settings = make_settings(tmp_path)
    orchestrator = make_orchestrator(settings)
    app = create_app(settings, orchestrator, ReadyVoiceEngine())
    job, _ = seed_job(app, settings)
    started, release = asyncio.Event(), asyncio.Event()

    async def blocked(*_args, **_kwargs):
        started.set()
        await release.wait()
        return {} if stage == "creating_replacement" else None

    monkeypatch.setattr("app.main.extract_waveforms", AsyncMock(return_value={}))
    monkeypatch.setattr("app.main.make_playback_mp3", AsyncMock(return_value=None))
    target = "extract_waveforms" if stage == "creating_replacement" else "make_playback_mp3"
    monkeypatch.setattr(f"app.main.{target}", blocked)
    async with client(app) as http:
        url = f"/api/jobs/{job.job_id}"
        await http.post(f"{url}/replace")
        await asyncio.wait_for(started.wait(), 2)
        assert job.replace_stage == stage
        events = await stream(app, job)
        await next_payload(events)
        patch = (await http.patch(url, json={"status": "cancelled"})).json()
        frame, payload = await next_payload(events)
        assert frame.startswith("event: done")
        assert operation_state(patch, "replace") == operation_state(payload, "replace")
        release.set()
        await job.replace_task
        detail = (await http.get(url)).json()
        assert detail["operationStatus"] == detail["replaceStatus"] == "cancelled"
        assert detail["operationProgress"] is None
        assert "replacedVocal" not in detail["result"]
        assert orchestrator.capacity.active == 0
        await events.aclose()


async def test_latest_operation_controls_terminal_response_without_losing_other_status(
    tmp_path, monkeypatch
):
    settings = make_settings(tmp_path)
    app = create_app(settings, make_orchestrator(settings), ReadyVoiceEngine())
    job, _ = seed_job(app, settings)
    monkeypatch.setattr("app.main.extract_waveforms", AsyncMock(return_value={}))
    monkeypatch.setattr("app.main.make_playback_mp3", AsyncMock(return_value=None))
    async with client(app) as http:
        url = f"/api/jobs/{job.job_id}"
        await http.post(f"{url}/replace")
        await job.replace_task
        replaced = (await http.get(url)).json()
        assert replaced["operationMessage"] == "人声替换完成"
        cached_split = (await http.post(f"{url}/split")).json()
        assert cached_split["operationMessage"] == "音轨分离完成"
        assert cached_split["splitStatus"] == cached_split["replaceStatus"] == "succeeded"
        cached_replace = (await http.post(f"{url}/replace")).json()
        assert cached_replace["operationMessage"] == "人声替换完成"
        assert cached_replace["splitSong"] == cached_replace["replaceSong"] == 0


async def test_job_publisher_bounds_snapshots_and_coalesces_generation_wakeups(tmp_path):
    settings = make_settings(tmp_path)
    app = create_app(settings, make_orchestrator(settings), ReadyVoiceEngine())
    job, _ = seed_job(app, settings)
    request = events_request(app, job.job_id)
    queue = asyncio.Queue(maxsize=JOB_EVENT_QUEUE_SIZE)
    app.state.job_subscribers[job.job_id] = {queue}
    job.last_operation = "replace"
    job.replace_status = "running"
    for progress in range(100):
        job.replace_progress = progress
        _publish_job(job, request, snapshot=True)
    assert queue.qsize() == JOB_EVENT_QUEUE_SIZE
    _publish_job(job, request)  # Already queued updates wake the consumer.
    assert queue.qsize() == JOB_EVENT_QUEUE_SIZE
    buffered = [queue.get_nowait() for _ in range(queue.qsize())]
    assert [event.replace_progress for event in buffered] == list(
        range(100 - JOB_EVENT_QUEUE_SIZE, 100)
    )
    _publish_job(job, request)
    _publish_job(job, request)
    assert queue.qsize() == 1
    assert queue.get_nowait() is None
    job.replace_status = job.replace_stage = "cancelled"
    job.replace_progress = None
    _publish_job(job, request, snapshot=True)
    assert queue.get_nowait().replace_status == "cancelled"


@pytest.mark.parametrize("error", [GenerationError("private /tmp/input"), OSError("secret")])
async def test_preparing_vocal_failure_reports_specific_safe_input_error(
    tmp_path, monkeypatch, error
):
    settings = make_settings(tmp_path)
    app = create_app(settings, make_orchestrator(settings), ReadyVoiceEngine())
    job, _ = seed_job(app, settings)
    guard = AsyncMock(side_effect=AssertionError("invalid input must not reach RVC"))
    monkeypatch.setattr(app.state.voice_engine, "convert", guard)

    def unreadable(_path):
        raise error

    monkeypatch.setattr("app.main._read_operation_input", unreadable)
    async with client(app) as http:
        url = f"/api/jobs/{job.job_id}"
        accepted = (await http.post(f"{url}/replace")).json()
        assert (accepted["operationStage"], accepted["operationMessage"]) == (
            "preparing_vocal",
            "正在准备输入音频",
        )
        await job.replace_task
        detail = (await http.get(url)).json()
        event = await http.get(f"{url}/events")
        payload = json.loads(event.text.split("data: ", 1)[1])
    assert operation_state(detail, "replace") == operation_state(payload, "replace")
    assert detail["replaceError"] == "人声输入音频不可读，请检查音轨文件后重试。"
    assert detail["operationStatus"] == detail["replaceStatus"] == "failed"
    assert detail["status"] == "succeeded"
    assert detail["operationProgress"] is None
    assert not any(private in event.text for private in ("private", "secret", "/tmp/input"))
    assert app.state.orchestrator.capacity.active == 0
    guard.assert_not_called()


@pytest.mark.parametrize("operation", ["split", "replace"])
async def test_second_song_operation_progress_and_cache_keep_song_number(
    tmp_path, monkeypatch, operation
):
    settings = make_settings(tmp_path)
    engine = BlockingVoiceEngine()
    app = create_app(settings, make_orchestrator(settings), engine)
    job, _ = seed_job(app, settings)
    alternative_job, _ = seed_job(app, settings, "alternative")
    job.result["count"] = 2
    job.result["alternatives"] = [alternative_job.result]
    if operation == "replace":
        # Put this song's input in the physical song_2 directory expected by RVC.
        source = settings.output_dir / alternative_job.result["stems"]["vocal"]
        target_dir = settings.output_dir / "jobs" / job.job_id / "song_2"
        target_dir.mkdir()
        target = target_dir / source.name
        target.write_bytes(source.read_bytes())
        alternative_job.result["stems"]["vocal"] = target.relative_to(
            settings.output_dir
        ).as_posix()
    monkeypatch.setattr("app.main.extract_waveforms", AsyncMock(return_value={}))
    monkeypatch.setattr("app.main.make_playback_mp3", AsyncMock(return_value=None))
    async with client(app) as http:
        url = f"/api/jobs/{job.job_id}"
        first = await http.post(f"{url}/{operation}?song=1")
        if operation == "replace":
            await asyncio.wait_for(engine.started.wait(), 2)
            running = (await http.get(url)).json()
            duplicate = (await http.post(f"{url}/{operation}?song=1")).json()
            events = await stream(app, job)
            _, payload = await next_payload(events)
            assert operation_state(running, operation) == operation_state(duplicate, operation)
            assert operation_state(running, operation) == operation_state(payload, operation)
            assert running["operationSong"] == running["replaceSong"] == 1
            assert running["status"] == "running"
            await events.aclose()
            engine.release.set()
            await job.replace_task
        else:
            assert first.status_code == 200
        cached = await http.post(f"{url}/{operation}?song=1")
        detail = (await http.get(url)).json()
        event = await http.get(f"{url}/events")
        payload = json.loads(event.text.split("data: ", 1)[1])
        assert cached.status_code == 200
        assert operation_state(cached.json(), operation) == operation_state(detail, operation)
        assert operation_state(detail, operation) == operation_state(payload, operation)
        assert detail["operationSong"] == detail[f"{operation}Song"] == 1
        assert detail["operationStatus"] == "succeeded"
        assert (detail["status"], detail["stage"], detail["progress"]) == (
            "succeeded",
            "completed",
            100,
        )
    if operation == "replace":
        restarted = create_app(settings, make_orchestrator(settings), ReadyVoiceEngine())
        async with client(restarted) as http:
            restored = (await http.get(url)).json()
            event = await http.get(f"{url}/events")
            payload = json.loads(event.text.split("data: ", 1)[1])
        for row in (restored, payload):
            assert row["operation"] == "replace"
            assert row["operationStatus"] == row["replaceStatus"] == "succeeded"
            assert row["operationSong"] == row["replaceSong"] == 1
