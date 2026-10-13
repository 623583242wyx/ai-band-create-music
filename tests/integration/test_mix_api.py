"""Contract tests for `POST /api/jobs/{jobId}/mix?song=N`.

Task-scoped like /split and /replace: 202 + job state + SSE + PATCH cancel. Mixing is
stubbed here so these tests pin request handling, job state, atomic publishing, the
invalidation rules and error reporting. The real FFmpeg pipeline lives in
tests/unit/test_mixing.py; the live-server path in tests/functional/test_http_server.py.

Tests covering the same path are merged on purpose: one test per behaviour, not one per
scenario.
"""

from __future__ import annotations

import asyncio
import copy
import json
from pathlib import Path
from urllib.parse import unquote, urlsplit

import httpx
import pytest
from starlette.requests import Request

from app.core.errors import GenerationError
from app.main import GenerationJob, create_app, load_jobs

# 试听文件名必须与真实编码器一致，否则响应层的 playback_matches 会过滤它。
from app.services.audio_files import _playback_name
from app.services.voice import RVCConversionError
from tests.helpers import make_orchestrator, make_settings

MIX_WAVEFORM = [0.5] * 640


def client(app):
    return httpx.AsyncClient(transport=httpx.ASGITransport(app), base_url="http://testserver")


def seed_job(app, settings, job_id="job-mix", *, count=1, prefix="demo"):
    """Real persisted job metadata; only media decoding is replaced by stubs."""
    songs = []
    for index in range(count):
        directory = settings.output_dir / "jobs" / job_id / f"song_{index + 1}"
        directory.mkdir(parents=True, exist_ok=True)
        names = {"fullTrack": "full_song.wav", "replacedVocal": f"{prefix}_rvc_vocal.wav"}
        names.update({name: f"{prefix}_{name}.mp3" for name in ("vocal", "drums", "bass", "other")})
        paths = {}
        for key, filename in names.items():
            path = directory / filename
            path.write_bytes(f"stub-{index}-{key}".encode())
            paths[key] = path.relative_to(settings.output_dir).as_posix()
        stems = {name: paths[name] for name in ("vocal", "drums", "bass", "other")}
        songs.append(
            {
                "fullTrack": paths["fullTrack"],
                "replacedVocal": paths["replacedVocal"],
                "_replacedVocalModel": app.state.rvc_model_fingerprint,
                "stems": stems,
                "stemUrls": list(stems.values()),
                "waveforms": {"full": [0.2] * 640, "vocal": [0.3] * 640},
                "splitEnabled": True,
                "debug": {},
            }
        )
    result = {
        **songs[0],
        "success": True,
        "jobId": job_id,
        "prompt": "rock",
        "durationMinutes": "auto",
        "structuredPrompt": "[Genre: Rock]",
        "lyrics": "[Verse]\ntest",
        "count": count,
        "alternatives": songs[1:],
    }
    job = GenerationJob(
        job_id=job_id,
        prompt="rock",
        structured_prompt="[Genre: Rock]",
        lyrics="[Verse]\ntest",
        status="succeeded",
        stage="completed",
        progress=100,
        message="音乐生成完成",
        result=result,
    )
    app.state.jobs[job_id] = job
    job.save(settings.output_dir)
    return job


def song_result(job, song=0):
    return job.result if song == 0 else job.result["alternatives"][song - 1]


class StubMixer:
    def __init__(self, *, blocking=False, error=None):
        self.started = asyncio.Event()
        self.release = asyncio.Event()
        if not blocking:
            self.release.set()
        self.error = error
        self.calls = []
        self.gains = []

    async def __call__(
        self,
        inputs,
        reference_path,
        output_path,
        timeout_seconds,
        *,
        gains=None,
        track_outputs=None,
    ):
        self.calls.append((list(inputs), reference_path, Path(output_path), timeout_seconds))
        self.gains.append(gains)
        self.started.set()
        await self.release.wait()
        Path(output_path).write_bytes(b"RIFF-mixed-stub")
        for path in track_outputs or []:
            Path(path).write_bytes(b"RIFF-retained-stub")
        if self.error:
            raise self.error


class StubReplaceEngine:
    """Enough of RVCEngine for /replace: writes a new vocal file and returns."""

    loaded = True

    async def convert(self, _input_path: Path, output_path: Path, **_params) -> None:
        output_path.write_bytes(b"RIFF-replaced-vocal")


class FailingReplaceEngine(StubReplaceEngine):
    async def convert(self, _input_path: Path, _output_path: Path, **_params) -> None:
        raise RuntimeError("engine down")


@pytest.fixture
def install_stubs(monkeypatch):
    def install(mixer=None, *, waveforms=None):
        active = mixer or StubMixer()
        monkeypatch.setattr("app.main.mix_tracks", active)

        async def extract(inputs, **_kwargs):
            assert "mix" in inputs
            assert all(Path(path).is_file() for path in inputs.values())
            return (
                {name: MIX_WAVEFORM.copy() for name in inputs} if waveforms is None else waveforms
            )

        monkeypatch.setattr("app.main.extract_waveforms", extract)
        return active

    return install


def assert_error(response, status):
    assert response.status_code == status, response.text
    body = response.json()
    assert body["success"] is False
    assert isinstance(body["message"], str) and body["message"]
    assert "private diagnostic" not in response.text
    return body


async def test_legacy_export_includes_guitar_and_piano(tmp_path, install_stubs):
    mixer = install_stubs()
    settings = make_settings(tmp_path)
    app = create_app(settings, make_orchestrator(settings))
    job = seed_job(app, settings)
    for name in ("guitar", "piano"):
        path = settings.output_dir / "jobs" / job.job_id / "song_1" / f"demo_{name}.wav"
        path.write_bytes(b"RIFF-stub")
        job.result["stems"][name] = path.relative_to(settings.output_dir).as_posix()
    async with client(app) as http:
        response = await http.post(f"/api/jobs/{job.job_id}/mix")
        assert response.status_code == 202, response.text
        await job.mix_task
    assert job.mix_status == "succeeded", job.mix_error
    assert [path.stem for path in mixer.calls[0][0]] == [
        "demo_rvc_vocal",
        "demo_drums",
        "demo_bass",
        "demo_other",
        "demo_guitar",
        "demo_piano",
    ]


def record_path(settings, job):
    return settings.output_dir / "jobs" / job.job_id / "job.json"


def mix_url(job, song=0):
    return f"/api/jobs/{job.job_id}/mix?song={song}"


async def run_job_mix(http, job, song=0):
    response = await http.post(mix_url(job, song))
    if job.mix_task is not None:
        await job.mix_task
    return response


def build_app(settings, engine=None):
    return create_app(settings, make_orchestrator(settings), engine)


# ------------------------------------------------------------------ happy path & state


@pytest.mark.parametrize("song", [0, 1])
async def test_job_route_mixes_song_and_reports_state(tmp_path, install_stubs, song):
    mixer = install_stubs()
    settings = make_settings(tmp_path, rvc_mix_timeout_seconds=17)
    orchestrator = make_orchestrator(settings)
    app = create_app(settings, orchestrator)
    job = seed_job(app, settings, count=2)
    before = copy.deepcopy(job.result)

    async with client(app) as http:
        accepted = await run_job_mix(http, job, song)
        detail = (await http.get(f"/api/jobs/{job.job_id}")).json()
        history = (await http.get("/api/jobs")).json()["jobs"][0]
        rendered = detail["result"] if song == 0 else detail["result"]["alternatives"][song - 1]
        audio = await http.get(rendered["mixedTrack"])

    assert accepted.status_code == 202, accepted.text
    selected = song_result(job, song)
    original = before if song == 0 else before["alternatives"][song - 1]
    # 成品落在该歌曲目录下、用原版固定名；原始音频与分轨都不动。
    assert selected["mixed"]["fullTrack"] == f"jobs/{job.job_id}/song_{song + 1}/demo_rvc_mix.wav"
    assert (settings.output_dir / selected["fullTrack"]).read_bytes() == (
        f"stub-{song}-fullTrack".encode()
    )
    assert selected["waveforms"] == original["waveforms"]
    assert selected["mixed"]["waveforms"] == {"full": MIX_WAVEFORM}
    assert (detail["mixStatus"], detail["mixSong"], detail["progress"]) == ("succeeded", song, 100)
    assert audio.content == b"RIFF-mixed-stub"
    assert rendered["mixedTrack"].startswith("http://testserver/output/")
    assert unquote(urlsplit(rendered["mixedTrack"]).path).endswith("/demo_rvc_mix.wav")
    # 历史投影带成品 URL，但按既有约定不带波形与运行态。
    history_song = history["result"] if song == 0 else history["result"]["alternatives"][song - 1]
    assert history_song["mixedTrack"] == rendered["mixedTrack"]
    assert history["result"]["waveforms"] == {} and "mixStatus" not in history
    assert orchestrator.capacity.active == 0 and len(mixer.calls) == 1
    inputs, reference, _output, timeout = mixer.calls[0]
    assert Path(reference) == settings.output_dir / selected["fullTrack"]
    assert inputs[0] == settings.output_dir / selected["replacedVocal"]
    assert [Path(path).name for path in inputs[1:]] == [
        "demo_drums.mp3",
        "demo_bass.mp3",
        "demo_other.mp3",
    ]
    assert timeout == 17
    assert json.loads(record_path(settings, job).read_text())["result"] == job.result
    # 另一首歌完全没被碰到。
    untouched = (
        job.result["alternatives"][0]
        if song == 0
        else {key: value for key, value in job.result.items() if key != "alternatives"}
    )
    previous = (
        before["alternatives"][0]
        if song == 0
        else {key: value for key, value in before.items() if key != "alternatives"}
    )
    assert untouched == previous


async def test_restart_derives_status_from_the_result(tmp_path, install_stubs):
    """mixStatus 不落盘：重启后由结果推导；唯一一首有成品才报序号，多首都有则报 null。"""
    install_stubs()
    settings = make_settings(tmp_path)
    seeded = build_app(settings)
    job = seed_job(seeded, settings, count=2)

    restarted = build_app(settings)
    restarted_job = restarted.state.jobs[job.job_id]
    async with client(restarted) as http:
        await run_job_mix(http, restarted_job, 1)
        single = (await http.get(f"/api/jobs/{job.job_id}")).json()
        # One completed mix after restart still identifies its song before any new POST.
        unique = build_app(settings)
        async with client(unique) as unique_http:
            unique_detail = (await unique_http.get(f"/api/jobs/{job.job_id}")).json()
            unique_event = await unique_http.get(f"/api/jobs/{job.job_id}/events")
        await run_job_mix(http, restarted_job, 0)

    # 再重启一次：两首歌都有成品、又没有运行态，此时不猜序号。
    fresh = build_app(settings)
    async with client(fresh) as http:
        both = (await http.get(f"/api/jobs/{job.job_id}")).json()
        both_event = await http.get(f"/api/jobs/{job.job_id}/events")

    for detail, event, expected_song in (
        (unique_detail, unique_event, 1),
        (both, both_event, None),
    ):
        payload = json.loads(event.text.split("data: ", 1)[1])
        assert event.text.startswith("event: done")
        for row in (detail, payload):
            assert (
                row["operation"],
                row["operationStatus"],
                row["operationStage"],
                row["operationProgress"],
                row["operationSong"],
                row["mixSong"],
            ) == ("mix", "succeeded", "completed", 100, expected_song, expected_song)

    assert (single["mixStatus"], single["mixSong"]) == ("succeeded", 1)
    assert (both["mixStatus"], both["mixSong"]) == ("succeeded", None)


@pytest.mark.parametrize("preview_outcome", ["ok", "none", "error"])
async def test_mix_streams_progress_over_sse(tmp_path, install_stubs, monkeypatch, preview_outcome):
    """试听成功或失败，终态帧都须在试听步骤结束后发出。"""
    mixer = install_stubs(StubMixer(blocking=True))
    settings = make_settings(tmp_path)
    app = build_app(settings)
    job = seed_job(app, settings)
    preview_started = asyncio.Event()
    preview_release = asyncio.Event()

    async def preview(source: Path, output_dir: Path) -> str | None:
        preview_started.set()
        await preview_release.wait()
        if preview_outcome == "none":
            return None
        if preview_outcome == "error":
            raise RuntimeError("preview unavailable")
        target = source.parent / "playtrack" / _playback_name(source)
        target.parent.mkdir(exist_ok=True)
        target.write_bytes(b"ID3-preview")
        return target.relative_to(output_dir).as_posix()

    monkeypatch.setattr("app.main.make_playback_mp3", preview)
    endpoint = next(
        route.endpoint for route in app.routes if route.path == "/api/jobs/{job_id}/events"
    )
    request = Request(
        {
            "type": "http",
            "app": app,
            "headers": [],
            "scheme": "http",
            "server": ("testserver", 80),
            "path": f"/api/jobs/{job.job_id}/events",
            "root_path": "",
            "query_string": b"",
            "method": "GET",
        }
    )
    async with client(app) as http:
        accepted = asyncio.create_task(http.post(mix_url(job)))
        stream = None
        try:
            await asyncio.wait_for(mixer.started.wait(), 2)
            stream = await endpoint(job.job_id, request)
            first = await anext(stream.body_iterator)
            running = json.loads(first.split("data: ", 1)[1])
            assert first.startswith("data: ")
            assert (running["status"], running["stage"], running["progress"]) == (
                "running",
                "mixing",
                76,
            )
            assert running["operationSong"] == running["mixSong"]
            assert running["operationStatus"] == running["mixStatus"] == "running"
            assert (running["operationStage"], running["operationProgress"]) == ("mixing", 76)
            assert running["result"]["waveforms"] == {}
            mixer.release.set()
            assert (await asyncio.wait_for(accepted, 2)).status_code == 202
            await asyncio.wait_for(preview_started.wait(), 2)
            waiting = (await http.get(f"/api/jobs/{job.job_id}")).json()
            assert (
                waiting["mixStatus"],
                waiting["operationStage"],
                waiting["operationProgress"],
            ) == (
                "running",
                "preview",
                95,
            )
            assert (waiting["status"], waiting["stage"], waiting["progress"]) == (
                "running",
                "preview",
                95,
            )
            preview_release.set()
            await job.mix_task
            done = await asyncio.wait_for(anext(stream.body_iterator), 2)
            while not done.startswith("event: done"):
                done = await asyncio.wait_for(anext(stream.body_iterator), 2)
            completed = json.loads(done.split("data: ", 1)[1])
            history = (await http.get("/api/jobs")).json()["jobs"][0]
        finally:
            mixer.release.set()
            preview_release.set()
            await asyncio.wait_for(accepted, 2)
            if stream is not None:
                await stream.body_iterator.aclose()
    assert done.startswith("event: done\ndata: ")
    assert (completed["mixStatus"], completed["progress"]) == ("succeeded", 100)
    assert (
        completed["operation"],
        completed["operationStatus"],
        completed["operationStage"],
        completed["operationProgress"],
        completed["operationMessage"],
    ) == ("mix", "succeeded", "completed", 100, "合轨完成")
    assert completed["result"]["waveforms"]["mix"] == MIX_WAVEFORM
    assert completed["result"]["mixedTrack"].endswith(".wav")
    if preview_outcome == "ok":
        preview_url = completed["result"]["playback"]["mixedTrack"]
        assert preview_url.endswith(".mp3")
        assert history["result"]["playback"]["mixedTrack"] == preview_url
    else:
        assert "mixedTrack" not in completed["result"].get("playback", {})
        assert "mixedTrack" not in history["result"].get("playback", {})
    assert job.job_id not in app.state.job_subscribers


# ------------------------------------------------------------------ publishing & rollback


async def test_failures_keep_the_previous_mix_intact(tmp_path, install_stubs, monkeypatch):
    """同名覆盖的三条失败路径都不得破坏上一版：混音失败、写盘失败、发布失败。"""
    mixer = install_stubs()
    settings = make_settings(tmp_path)
    app = build_app(settings)
    job = seed_job(app, settings)
    async with client(app) as http:
        await run_job_mix(http, job)
        published = settings.output_dir / job.result["mixed"]["fullTrack"]
        published.write_bytes(b"RIFF-previous-mix")
        stored = record_path(settings, job).read_bytes()
        previous = copy.deepcopy(job.result)

        mixer.error = RVCConversionError("private diagnostic")
        await run_job_mix(http, job)
        detail = (await http.get(f"/api/jobs/{job.job_id}")).json()
        assert (detail["mixStatus"], detail["status"]) == ("failed", "succeeded")
        assert (
            detail["operation"],
            detail["operationStatus"],
            detail["operationStage"],
            detail["operationProgress"],
        ) == ("mix", "failed", "failed", None)
        assert (await http.get(detail["result"]["mixedTrack"])).content == b"RIFF-previous-mix"

        mixer.error = None
        saved = GenerationJob.save
        monkeypatch.setattr(
            GenerationJob,
            "save",
            lambda self, out: (
                (_ for _ in ()).throw(OSError("private diagnostic"))
                if self is job
                else saved(self, out)
            ),
        )
        await run_job_mix(http, job)

    assert job.result == previous
    assert published.read_bytes() == b"RIFF-previous-mix"
    assert record_path(settings, job).read_bytes() == stored
    assert job.mix_status == "failed"


async def test_failed_publish_rolls_metadata_back(tmp_path, install_stubs, monkeypatch):
    """`output.replace()` 失败时元数据必须退回上一版：文件是旧的，车道也必须还是旧的。"""
    install_stubs()
    settings = make_settings(tmp_path)
    app = build_app(settings)
    job = seed_job(app, settings)
    async with client(app) as http:
        await run_job_mix(http, job)
        stored = record_path(settings, job).read_bytes()
        previous = copy.deepcopy(job.result)
        real_replace = Path.replace

        def failing_replace(self, target):
            if Path(target).name == "demo_rvc_mix.wav":
                raise IsADirectoryError("private diagnostic")
            return real_replace(self, target)

        monkeypatch.setattr(Path, "replace", failing_replace)
        await run_job_mix(http, job)

    assert job.mix_status == "failed" and job.result == previous
    assert record_path(settings, job).read_bytes() == stored


async def test_waveform_failure_is_non_fatal(tmp_path, install_stubs):
    """波形只是编辑器的绘制数据：提取失败不影响已经落盘的成品。"""
    install_stubs()
    settings = make_settings(tmp_path)
    app = build_app(settings)
    job = seed_job(app, settings)
    async with client(app) as http:
        await run_job_mix(http, job)
        install_stubs(waveforms={})
        await run_job_mix(http, job)
        detail = (await http.get(f"/api/jobs/{job.job_id}")).json()
        audio = await http.get(detail["result"]["mixedTrack"])

    assert detail["mixStatus"] == "succeeded"
    assert "mix" not in detail["result"]["waveforms"]
    assert detail["result"]["waveforms"]["full"] == [0.2] * 640
    assert audio.content == b"RIFF-mixed-stub"


async def test_history_load_drops_a_mix_reference_without_a_file(tmp_path, install_stubs):
    """元数据先落盘、文件后替换之间被杀进程：重启要摘掉悬空引用。"""
    install_stubs()
    settings = make_settings(tmp_path)
    app = build_app(settings)
    job = seed_job(app, settings)
    async with client(app) as http:
        await run_job_mix(http, job)
    (settings.output_dir / job.result["mixed"]["fullTrack"]).unlink()

    reloaded = load_jobs(settings.output_dir)[job.job_id]
    assert "mixed" not in reloaded.result
    assert "mix" not in reloaded.result["waveforms"]
    assert "mixed" not in json.loads(record_path(settings, job).read_text())["result"]


# ------------------------------------------------------------------------ admission


async def test_admission_rules(tmp_path, install_stubs):
    """额度用满 429；同一首歌并发只跑一次（额度足够时靠任务占位拦住第二个）。"""
    mixer = install_stubs(StubMixer(blocking=True))
    settings = make_settings(tmp_path, max_concurrent_generations=2)
    orchestrator = make_orchestrator(settings)
    app = create_app(settings, orchestrator)
    job = seed_job(app, settings)
    async with client(app) as http:
        first = asyncio.create_task(http.post(mix_url(job)))
        second = asyncio.create_task(http.post(mix_url(job)))
        try:
            await asyncio.wait_for(mixer.started.wait(), 2)
            assert (await second).status_code == 409
        finally:
            mixer.release.set()
            assert (await asyncio.wait_for(first, 3)).status_code == 202
            await job.mix_task

    for _ in range(2):
        await orchestrator.capacity.acquire()
    try:
        async with client(app) as http:
            assert_error(await http.post(mix_url(job)), 429)
    finally:
        for _ in range(2):
            await orchestrator.capacity.release()
    assert len(mixer.calls) == 1 and orchestrator.capacity.active == 0


@pytest.mark.parametrize("status", ["running", "cancelled"])
async def test_song_operations_block_mix_until_their_task_really_ends(
    tmp_path, install_stubs, status
):
    """拆轨/替换在跑时拒绝合轨；已进入终态但线程还在收尾（task 未结束）时同样拒绝。

    后者是替换超时/取消后的真实状态：状态字段已是 failed/cancelled，`replace_task` 仍在
    `_await_stuck_replacement_worker` 里等 RVC 线程退出——此时放行合轨就会去读那份人声。
    """
    mixer = install_stubs()
    settings = make_settings(tmp_path)
    app = build_app(settings)
    job = seed_job(app, settings)
    release = asyncio.Event()
    for operation in ("split", "replace"):
        task = asyncio.create_task(release.wait())
        setattr(job, f"{operation}_task", task)
        setattr(job, f"{operation}_status", status)
        setattr(job, f"{operation}_song", 0)
        async with client(app) as http:
            assert_error(await http.post(mix_url(job)), 409)
        release.set()
        await task
    assert not mixer.calls


async def test_mix_blocks_every_other_job_mutation_while_running(tmp_path, install_stubs):
    mixer = install_stubs(StubMixer(blocking=True))
    settings = make_settings(tmp_path)
    orchestrator = make_orchestrator(settings)
    app = create_app(settings, orchestrator)
    job = seed_job(app, settings, count=2)
    before = copy.deepcopy(job.result)
    async with client(app) as http:
        accepted = asyncio.create_task(http.post(mix_url(job)))
        try:
            await asyncio.wait_for(mixer.started.wait(), 2)
            assert (await accepted).status_code == 202
            assert orchestrator.capacity.active == 1
            assert_error(await http.post(mix_url(job)), 409)
            assert_error(await http.post(mix_url(job, 1)), 409)
            for method, url in [
                ("POST", f"/api/jobs/{job.job_id}/split"),
                ("POST", f"/api/jobs/{job.job_id}/replace"),
                ("DELETE", f"/api/jobs/{job.job_id}/stems/drums"),
                ("PUT", f"/api/jobs/{job.job_id}/stems/drums"),
            ]:
                blocked = await http.request(method, url)
                assert blocked.status_code == 409, (method, url, blocked.text)
            assert job.result == before
        finally:
            mixer.release.set()
            await asyncio.wait_for(accepted, 2)
            await job.mix_task
    assert len(mixer.calls) == 1 and orchestrator.capacity.active == 0


@pytest.mark.parametrize(
    "mutation,status",
    [
        ("job", 404),
        ("other song", 404),
        ("unfinished", 409),
        ("no stems", 409),
        ("no replacement", 409),
        ("stem traversal", 400),
        ("stem absolute", 400),
        ("stem other song", 400),
        ("stem in trash", 400),
        ("stem non media", 400),
        ("vocal traversal", 400),
        ("missing stem file", 404),
        ("malformed fullTrack", 409),
    ],
)
async def test_bad_inputs_are_refused(tmp_path, install_stubs, mutation, status):
    """畸形记录与缺失输入一律拒绝，且不启动混音。"""
    mixer = install_stubs()
    settings = make_settings(tmp_path)
    app = build_app(settings)
    job = seed_job(app, settings)
    song_dir = settings.output_dir / "jobs" / job.job_id / "song_1"
    (song_dir / "notes.txt").write_text("not audio")
    trash = settings.output_dir / ".trash" / job.job_id / "song_1"
    trash.mkdir(parents=True, exist_ok=True)
    (trash / "drums.wav").write_bytes(b"trashed")
    secret = tmp_path / "secret.wav"
    secret.write_bytes(b"secret")

    cases = {
        "unfinished": lambda: setattr(job, "status", "running"),
        "no stems": lambda: job.result.__setitem__("stems", {}),
        "no replacement": lambda: job.result.pop("replacedVocal"),
        "stem traversal": lambda: job.result["stems"].__setitem__(
            "drums", f"jobs/{job.job_id}/song_1/../../../secret.wav"
        ),
        "stem absolute": lambda: job.result["stems"].__setitem__("drums", str(secret)),
        "stem other song": lambda: job.result["stems"].__setitem__(
            "drums", f"jobs/{job.job_id}/song_2/drums.wav"
        ),
        "stem in trash": lambda: job.result["stems"].__setitem__(
            "drums", (trash / "drums.wav").relative_to(settings.output_dir).as_posix()
        ),
        "stem non media": lambda: job.result["stems"].__setitem__(
            "drums", (song_dir / "notes.txt").relative_to(settings.output_dir).as_posix()
        ),
        "vocal traversal": lambda: job.result.__setitem__(
            "replacedVocal", f"jobs/{job.job_id}/song_1/../../../secret.wav"
        ),
        "missing stem file": lambda: (settings.output_dir / job.result["stems"]["bass"]).unlink(),
        "malformed fullTrack": lambda: job.result.__setitem__("fullTrack", 123),
    }
    if mutation in cases:
        cases[mutation]()
    async with client(app) as http:
        url = (
            "/api/jobs/nope/mix"
            if mutation == "job"
            else mix_url(job, 1 if mutation == "other song" else 0)
        )
        response = await http.post(url)
    assert response.status_code == status, (mutation, response.text)
    assert not mixer.calls and "mixed" not in job.result


async def test_url_shaped_values_resolve_locally_without_any_fetch(
    tmp_path, install_stubs, monkeypatch
):
    """绝对 URL 只是路径提示：按 `/output/` 之后的部分定位本地文件，绝不发起请求。"""
    mixer = install_stubs()
    settings = make_settings(tmp_path)
    app = build_app(settings)
    job = seed_job(app, settings)
    expected = [settings.output_dir / job.result["replacedVocal"]] + [
        settings.output_dir / job.result["stems"][name] for name in ("drums", "bass", "other")
    ]
    base = "http://example.invalid/output/"
    job.result["fullTrack"] = base + job.result["fullTrack"]
    job.result["replacedVocal"] = base + job.result["replacedVocal"]
    job.result["stems"] = {name: base + url for name, url in job.result["stems"].items()}

    def forbidden(*_args, **_kwargs):
        raise AssertionError("the mix path must not perform remote requests")

    with monkeypatch.context() as patch:
        patch.setattr(httpx.AsyncClient, "get", forbidden)
        patch.setattr(httpx.AsyncClient, "stream", forbidden)
        async with client(app) as http:
            accepted = await run_job_mix(http, job)

    assert accepted.status_code == 202, accepted.text
    assert mixer.calls[0][0] == expected


async def test_unreadable_inputs_are_client_errors_not_crashes(
    tmp_path, install_stubs, monkeypatch
):
    """EACCES 时 Path.is_file() 抛 PermissionError：必须落到 404，而不是 500。"""
    mixer = install_stubs()
    settings = make_settings(tmp_path)
    app = build_app(settings)
    job = seed_job(app, settings)
    target = settings.output_dir / job.result["stems"]["bass"]
    real_is_file = Path.is_file

    def guarded(self):
        if self == target:
            raise PermissionError("private diagnostic")
        return real_is_file(self)

    monkeypatch.setattr(Path, "is_file", guarded)
    async with client(app) as http:
        assert (await http.post(mix_url(job))).status_code == 404
    assert not mixer.calls


async def test_unreadable_artifact_never_hides_the_whole_job(tmp_path, install_stubs, monkeypatch):
    """重启修复要"判不出来就保留引用"，不能把一个读不到的成品放大成整个任务消失。"""
    install_stubs()
    settings = make_settings(tmp_path)
    app = build_app(settings)
    job = seed_job(app, settings)
    async with client(app) as http:
        await run_job_mix(http, job)
    artifact = settings.output_dir / job.result["mixed"]["fullTrack"]
    real_is_file = Path.is_file

    def guarded(self):
        if self == artifact:
            raise PermissionError("private diagnostic")
        return real_is_file(self)

    monkeypatch.setattr(Path, "is_file", guarded)
    reloaded = load_jobs(settings.output_dir)
    assert job.job_id in reloaded
    # 加载时旧版成品迁成独立的替换后那首，引用照旧保留。
    assert reloaded[job.job_id].result["mixed"]["fullTrack"] == job.result["mixed"]["fullTrack"]


async def test_mix_code_path_stays_local_only(tmp_path, install_stubs):
    """护栏：引擎与路由都只碰本地文件，不得出现网络客户端。"""
    import inspect

    import app.services.voice as voice

    install_stubs()
    app = build_app(make_settings(tmp_path))
    endpoint = next(
        route.endpoint for route in app.routes if route.path == "/api/jobs/{job_id}/mix"
    )
    sources = {
        "engine": inspect.getsource(voice.mix_tracks) + inspect.getsource(voice._run_ffmpeg),
        "route": inspect.getsource(endpoint),
    }
    for label, source in sources.items():
        for forbidden in ("httpx", "urlopen", "urllib.request", "aiohttp", "requests."):
            assert forbidden not in source, (label, forbidden)


# ---------------------------------------------------------------- errors & cancellation


async def test_mix_failures_are_reported_through_job_state(tmp_path, install_stubs):
    """混音失败只进 job 状态：文案通用，内部细节只写日志。"""
    mixer = install_stubs()
    settings = make_settings(tmp_path)
    app = build_app(settings)
    job = seed_job(app, settings)
    expected = {
        RVCConversionError: "合轨失败，请检查音轨后重试。",
        GenerationError: "音频处理工具不可用，请检查 FFmpeg 配置。",
        RuntimeError: "合轨失败，请检查音轨后重试。",
    }
    async with client(app) as http:
        for error, message in expected.items():
            mixer.error = error("private diagnostic")
            assert (await http.post(mix_url(job))).status_code == 202
            await job.mix_task
            detail = (await http.get(f"/api/jobs/{job.job_id}")).json()
            assert (detail["mixStatus"], detail["mixError"]) == ("failed", message), error
            assert "private diagnostic" not in json.dumps(detail)


async def test_cancellation_discards_the_mix(tmp_path, install_stubs, monkeypatch):
    """两条取消路径都不得发布成品：PATCH 取消，以及波形阶段的取消信号。"""
    mixer = install_stubs(StubMixer(blocking=True))
    settings = make_settings(tmp_path)
    orchestrator = make_orchestrator(settings)
    app = create_app(settings, orchestrator)
    job = seed_job(app, settings)
    before = copy.deepcopy(job.result)
    async with client(app) as http:
        accepted = asyncio.create_task(http.post(mix_url(job)))
        try:
            await asyncio.wait_for(mixer.started.wait(), 2)
            patched = await http.patch(f"/api/jobs/{job.job_id}", json={"status": "cancelled"})
            assert patched.json()["mixStatus"] == "cancelled"
            assert patched.json()["status"] == "succeeded"
        finally:
            mixer.release.set()
            assert (await asyncio.wait_for(accepted, 2)).status_code == 202
            await job.mix_task
        assert (await http.get(f"/api/jobs/{job.job_id}")).json()["mixStatus"] == "cancelled"
    assert job.result == before and orchestrator.capacity.active == 0
    assert job.mix_cancel_requested is False

    # 波形阶段：取消信号只置位、不强停任务（replace 的收尾方式），成品同样不能发布。
    started = asyncio.Event()
    release = asyncio.Event()

    async def blocking_waveforms(_inputs, **_kwargs):
        started.set()
        await release.wait()
        return {"mix": MIX_WAVEFORM.copy()}

    with monkeypatch.context() as patch:
        patch.setattr("app.main.extract_waveforms", blocking_waveforms)
        async with client(app) as http:
            accepted = asyncio.create_task(http.post(mix_url(job)))
            await asyncio.wait_for(started.wait(), 2)
            job.mix_cancel_requested = True
            release.set()
            assert (await asyncio.wait_for(accepted, 2)).status_code == 202
            await job.mix_task
            after = (await http.get(f"/api/jobs/{job.job_id}")).json()
    assert after["mixStatus"] == "cancelled" and job.result == before
    assert not (settings.output_dir / "jobs" / job.job_id / "song_1" / "demo_rvc_mix.wav").exists()


async def test_cancel_during_preview_keeps_published_mix_succeeded(
    tmp_path, install_stubs, monkeypatch
):
    install_stubs()
    settings = make_settings(tmp_path)
    app = build_app(settings)
    job = seed_job(app, settings)
    encoding = asyncio.Event()

    async def blocked_preview(*_args):
        encoding.set()
        await asyncio.Event().wait()

    monkeypatch.setattr("app.main.make_playback_mp3", blocked_preview)
    async with client(app) as http:
        accepted = await http.post(mix_url(job))
        assert accepted.status_code == 202
        await asyncio.wait_for(encoding.wait(), 2)
        patched = (await http.patch(f"/api/jobs/{job.job_id}", json={"status": "cancelled"})).json()
        await job.mix_task
        detail = (await http.get(f"/api/jobs/{job.job_id}")).json()

    assert patched["mixStatus"] in {"running", "succeeded"}
    assert detail["mixStatus"] == "succeeded"
    assert detail["result"]["mixedTrack"].endswith(".wav")
    assert "mixedTrack" not in detail["result"].get("playback", {})
    assert (settings.output_dir / job.result["mixed"]["fullTrack"]).is_file()


async def test_preview_save_failure_keeps_published_mix_succeeded(
    tmp_path, install_stubs, monkeypatch
):
    install_stubs()
    settings = make_settings(tmp_path)
    app = build_app(settings)
    job = seed_job(app, settings)
    save = GenerationJob.save
    calls = 0

    def fail_second_save(current_job, root):
        nonlocal calls
        if current_job is not job:
            return save(current_job, root)
        calls += 1
        if calls == 2:
            raise OSError("preview metadata unavailable")
        save(current_job, root)

    async def preview(*_args):
        return "playtrack/preview.mp3"

    monkeypatch.setattr(GenerationJob, "save", fail_second_save)
    monkeypatch.setattr("app.main.make_playback_mp3", preview)
    async with client(app) as http:
        assert (await http.post(mix_url(job))).status_code == 202
        await job.mix_task
        detail = (await http.get(f"/api/jobs/{job.job_id}")).json()
    # 第三次落盘是把旧版成品迁成替换后那首。
    assert calls == 3
    assert detail["mixStatus"] == "succeeded"
    assert "fullTrack" not in job.result["mixed"].get("playback", {})
    assert (settings.output_dir / job.result["mixed"]["fullTrack"]).is_file()


# ------------------------------------------------------- replacement model / provenance


async def test_replacement_model_fingerprint_guard(tmp_path, install_stubs):
    """指纹不符且资产在场 → 409；资产不在（缺挂载实例）→ 放行，因为合轨只用 ffmpeg。"""
    mixer = install_stubs()
    settings = make_settings(tmp_path)
    app = build_app(settings)
    job = seed_job(app, settings)
    job.result["_replacedVocalModel"] = "v1:" + "0" * 64
    async with client(app) as http:
        blocked = await http.post(mix_url(job))
        # 替换侧同样认为这份人声失效。
        assert (await http.get(f"/api/jobs/{job.job_id}")).json()["replaceStatus"] is None
        settings.rvc_model_path.unlink()
        allowed = await run_job_mix(http, job)

    assert blocked.status_code == 409 and "模型" in blocked.json()["message"]
    assert allowed.status_code == 202 and job.result["mixed"]["fullTrack"]
    assert len(mixer.calls) == 1


async def test_assets_missing_does_not_report_a_stale_replacement(tmp_path, install_stubs):
    """缺资产时指纹无从验证，替换状态不该被判成失效（那会引导去跑必然失败的 /replace）。"""
    install_stubs()
    settings = make_settings(tmp_path)
    app = build_app(settings)
    job = seed_job(app, settings)
    job.result["_replacedVocalModel"] = "v1:" + "0" * 64
    settings.rvc_model_path.unlink()
    async with client(app) as http:
        lenient = (await http.get(f"/api/jobs/{job.job_id}")).json()
    settings.rvc_model_path.write_bytes(b"model again")
    app.state.rvc_model_fingerprint = "v1:" + "1" * 64
    async with client(app) as http:
        strict = (await http.get(f"/api/jobs/{job.job_id}")).json()
    assert lenient["replaceStatus"] == "succeeded"
    assert strict["replaceStatus"] is None


async def test_replacement_outcomes_against_the_exported_mix(tmp_path, install_stubs, monkeypatch):
    """替换成功 → 成品真正过期；替换失败/写盘失败 → 成品仍可播、落盘记录自洽。"""
    install_stubs()
    settings = make_settings(tmp_path)

    # 成功：替换后那首是独立的歌，原曲重新替换人声不会作废它，下一次提交才覆盖。
    app = build_app(settings, StubReplaceEngine())
    job = seed_job(app, settings)
    async with client(app) as http:
        await run_job_mix(http, job)
        job.result["_replacedVocalModel"] = "v1:" + "0" * 64  # 让 /replace 真的重跑
        await http.post(f"/api/jobs/{job.job_id}/replace?song=0")
        await job.replace_task
        detail = (await http.get(f"/api/jobs/{job.job_id}")).json()
    assert detail["replaceStatus"] == "succeeded"
    assert detail["result"]["mixedTrack"].endswith("/demo_rvc_mix.wav")
    assert detail["mixStatus"] == "succeeded"

    # 引擎失败：人声引用撤下（既有契约），但成品必须还能播。
    app = build_app(settings, FailingReplaceEngine())
    job = seed_job(app, settings, job_id="job-failed")
    async with client(app) as http:
        await run_job_mix(http, job)
        job.result["_replacedVocalModel"] = "v1:" + "0" * 64
        await http.post(f"/api/jobs/{job.job_id}/replace?song=0")
        await job.replace_task
        detail = (await http.get(f"/api/jobs/{job.job_id}")).json()
        audio = await http.get(detail["result"]["mixedTrack"])
    assert detail["replaceStatus"] == "failed"
    assert "replacedVocal" not in detail["result"]
    assert detail["mixStatus"] == "succeeded" and audio.content == b"RIFF-mixed-stub"

    # 写盘失败：撤下失效的人声引用，但旧音轨与旧成品仍可用。
    app = build_app(settings, StubReplaceEngine())
    job = seed_job(app, settings, job_id="job-savefail")
    previous_vocal = settings.output_dir / job.result["replacedVocal"]
    previous_audio = previous_vocal.read_bytes()
    async with client(app) as http:
        await run_job_mix(http, job)
        job.result["_replacedVocalModel"] = "v1:" + "0" * 64
        saved = GenerationJob.save

        def fail_save(self, output_dir):
            published = self is job and self.result.get("_replacedVocalModel") == (
                app.state.rvc_model_fingerprint
            )
            if published:
                raise OSError("private diagnostic")
            return saved(self, output_dir)

        monkeypatch.setattr(GenerationJob, "save", fail_save)
        await http.post(f"/api/jobs/{job.job_id}/replace?song=0")
        await job.replace_task
        assert (await http.get(f"/api/jobs/{job.job_id}")).json()["replaceStatus"] == "failed"
    reloaded = load_jobs(settings.output_dir)[job.job_id]
    assert "replacedVocal" not in reloaded.result
    assert previous_vocal.read_bytes() == previous_audio
    assert reloaded.result["mixed"]["fullTrack"] == job.result["mixed"]["fullTrack"]
    assert (
        settings.output_dir / reloaded.result["mixed"]["fullTrack"]
    ).read_bytes() == b"RIFF-mixed-stub"


# ------------------------------------------------------------- invalidation & restore


@pytest.mark.parametrize("deleted", ["stem", "replacement"])
async def test_deleting_an_original_input_keeps_the_replaced_song(tmp_path, install_stubs, deleted):
    """替换后那首是独立的歌：删掉原曲的分轨或替换人声、再撤回，都不影响它。"""
    install_stubs()
    settings = make_settings(tmp_path)
    app = build_app(settings)
    job = seed_job(app, settings)
    vocal = Path(job.result["replacedVocal"]).name
    async with client(app) as http:
        await run_job_mix(http, job)
        before = (await http.get(f"/api/jobs/{job.job_id}")).json()["result"]
        if deleted == "stem":
            removed = await http.request("DELETE", f"/api/jobs/{job.job_id}/stems/drums")
            while_deleted = (await http.get(f"/api/jobs/{job.job_id}")).json()
            undone = await http.put(f"/api/jobs/{job.job_id}/stems/drums")
        else:
            removed = await http.request(
                "DELETE", "/api/voice/result", data={"job_id": job.job_id, "filename": vocal}
            )
            while_deleted = (await http.get(f"/api/jobs/{job.job_id}")).json()
            undone = await http.request(
                "PUT", "/api/voice/result", data={"job_id": job.job_id, "filename": vocal}
            )
        restored = (await http.get(f"/api/jobs/{job.job_id}")).json()

    assert removed.status_code in {200, 204} and undone.status_code in {200, 204}
    for detail in (while_deleted, restored):
        assert detail["mixStatus"] == "succeeded"
        assert detail["result"]["mixed"] == before["mixed"]
        assert detail["result"]["mixedTrack"] == before["mixedTrack"]
        assert detail["result"]["waveforms"]["mix"] == MIX_WAVEFORM


async def test_voice_result_refuses_the_replaced_song(tmp_path, install_stubs):
    """这个入口只管派生音频：原曲母带与替换后那首的成品都是歌，一律拒绝。"""
    install_stubs()
    settings = make_settings(tmp_path)
    app = build_app(settings)
    job = seed_job(app, settings)
    async with client(app) as http:
        await run_job_mix(http, job)
        refused = [
            await http.request(
                "DELETE", "/api/voice/result", data={"job_id": job.job_id, "filename": filename}
            )
            for filename in (Path(job.result["fullTrack"]).name, "demo_rvc_mix.wav")
        ]
        detail = (await http.get(f"/api/jobs/{job.job_id}")).json()

    assert [response.status_code for response in refused] == [409, 409]
    assert (settings.output_dir / job.result["fullTrack"]).is_file()
    assert (settings.output_dir / job.result["mixed"]["fullTrack"]).is_file()
    assert detail["mixStatus"] == "succeeded"
