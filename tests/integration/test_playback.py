import array
import asyncio
import copy
import json
import math
import os
import shutil
import subprocess
import sys
import wave
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager, suppress
from datetime import datetime, timedelta, timezone
from pathlib import Path

import httpx
import pytest
from fastapi import FastAPI

from app.main import GenerationJob, _render_result_urls, create_app, load_jobs
from app.services.audio_files import make_playback_mp3
from app.services.job_files import capture_mix_artifact, drop_mix_artifact, restore_mix_artifact
from app.services.orchestrator import GenerationOrchestrator
from app.services.retention import cleanup_expired_jobs
from app.services.stems import STEM_NAMES
from tests.helpers import make_orchestrator, make_settings


def wav(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as output:
        output.setparams((1, 2, 8000, 0, "NONE", "not compressed"))
        output.writeframes(b"\0\0" * 8000)


async def test_preview_uses_constant_bitrate_for_firefox_seeking(tmp_path: Path) -> None:
    source = tmp_path / "guitar.wav"
    samples = array.array(
        "h",
        (
            int(8000 * math.sin(index * 440 * 2 * math.pi / 44100)) if index < 44100 else 0
            for index in range(88200)
        ),
    )
    with wave.open(str(source), "wb") as output:
        output.setparams((1, 2, 44100, 0, "NONE", "not compressed"))
        output.writeframes(samples.tobytes())
    preview = await make_playback_mp3(source, tmp_path)
    assert preview is not None
    probe = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "stream=bit_rate",
            "-of",
            "json",
            str(tmp_path / preview),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    assert json.loads(probe.stdout)["streams"][0]["bit_rate"] == "192000"


async def test_legacy_vbr_preview_is_replaced_without_reusing_its_cached_url(
    tmp_path: Path,
) -> None:
    settings = make_settings(tmp_path)
    source = settings.output_dir / "jobs/j/song_1/guitar.wav"
    wav(source)
    version = source.stat()
    old = (
        source.parent
        / "playtrack"
        / (f"{source.stem}.playback-{version.st_mtime_ns}-{version.st_size}.mp3")
    )
    old.parent.mkdir()
    old.write_bytes(b"ID3-legacy-vbr")
    result = {
        "fullTrack": source.relative_to(settings.output_dir).as_posix(),
        "stems": {},
        "playback": {"fullTrack": old.relative_to(settings.output_dir).as_posix()},
    }
    # An old job must fall back to its WAV until a seek-safe preview exists.
    rendered = _render_result_urls(result, "http://testserver", settings)
    assert not rendered["playback"]
    assert rendered["fullTrack"].endswith("guitar.wav")
    app = create_app(settings, make_orchestrator(settings))
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app), base_url="http://testserver"
    ) as client:
        legacy_url = f"/output/{old.relative_to(settings.output_dir)}"
        assert (await client.get(legacy_url)).status_code == 404
    fresh = await make_playback_mp3(source, settings.output_dir)
    assert fresh is not None and fresh != old.relative_to(settings.output_dir).as_posix()
    assert ".playback-cbr192-" in fresh
    rendered = _render_result_urls(result, "http://testserver", settings)
    assert rendered["playback"]["fullTrack"].endswith(fresh)


@pytest.mark.parametrize("suffix", [".wav", ".WAV", ".wAv"])
async def test_four_preview_urls_range_and_stale_versions(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, suffix: str
) -> None:
    settings = make_settings(tmp_path, song_retention_enabled=True, song_retention_dry_run=False)
    folder = settings.output_dir / "jobs/j/song_1"
    names = {
        "fullTrack": f"full[1]{suffix}",
        "vocal": f"vocal{suffix}",
        "replacedVocal": f"replaced{suffix}",
        "mixedTrack": f"mix{suffix}",
    }
    paths = {key: folder / name for key, name in names.items()}
    for path in paths.values():
        if suffix != ".wav":
            wav(path.with_suffix(".wav"))
        wav(path)
        if suffix != ".wav":
            old_mtime = path.stat().st_mtime_ns - 1_000_000
            os.utime(path.with_suffix(".wav"), ns=(old_mtime, old_mtime))
    previews = {
        key: await make_playback_mp3(path, settings.output_dir) for key, path in paths.items()
    }
    assert all(previews.values())
    assert all(Path(preview).parent.name == "playtrack" for preview in previews.values())
    assert all((settings.output_dir / preview).is_file() for preview in previews.values())
    result = {
        "fullTrack": paths["fullTrack"].relative_to(settings.output_dir).as_posix(),
        "stems": {"vocal": paths["vocal"].relative_to(settings.output_dir).as_posix()},
        "replacedVocal": paths["replacedVocal"].relative_to(settings.output_dir).as_posix(),
        "mixedTrack": paths["mixedTrack"].relative_to(settings.output_dir).as_posix(),
        "playback": {
            "fullTrack": previews["fullTrack"],
            "stems": {"vocal": previews["vocal"]},
            "replacedVocal": previews["replacedVocal"],
            "mixedTrack": previews["mixedTrack"],
        },
    }
    rendered = _render_result_urls(result, "http://testserver", settings)
    assert set(rendered["playback"]) == {"fullTrack", "stems", "replacedVocal", "mixedTrack"}
    assert rendered["fullTrack"].endswith(suffix)
    original_open = Path.open

    def reject_wav_read(path: Path, *args, **kwargs):
        if path.suffix.lower() == ".wav":
            raise AssertionError("result rendering must not read WAV bytes")
        return original_open(path, *args, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(Path, "open", reject_wav_read)
        assert (
            _render_result_urls(result, "http://testserver", settings)["playback"]
            == rendered["playback"]
        )
    job = GenerationJob(job_id="j", prompt="rock", status="succeeded", stage="completed")
    job.save(settings.output_dir)
    app = create_app(settings, make_orchestrator(settings))
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app), base_url="http://testserver"
    ) as client:
        pairs = [(rendered[key], rendered["playback"][key]) for key in names if key != "vocal"]
        pairs.append((rendered["stems"]["vocal"], rendered["playback"]["stems"]["vocal"]))
        for source_url, preview_url in pairs:
            assert (await client.get(source_url)).status_code == 200
            mp3 = await client.get(preview_url)
            assert mp3.status_code == 200
            assert mp3.headers["content-type"].startswith("audio/mpeg")
        response = await client.get(
            rendered["playback"]["fullTrack"], headers={"Range": "bytes=0-2"}
        )
    assert response.status_code == 206
    assert response.headers["content-type"].startswith("audio/mpeg")
    assert response.headers["content-length"] == "3"
    assert response.headers["content-range"].startswith("bytes 0-2/")
    cache_control = response.headers["cache-control"]
    assert cache_control.startswith("public, max-age=") and cache_control.endswith(", immutable")
    max_age = int(cache_control.split("max-age=")[1].split(",")[0])
    assert 3 * 86400 - 5 <= max_age <= 3 * 86400
    assert len(response.content) == 3

    async def unexpected_encode(*_args, **_kwargs):
        raise AssertionError("an existing MP3 must not be encoded again")

    with monkeypatch.context() as patch:
        patch.setattr(asyncio, "create_subprocess_exec", unexpected_encode)
        assert (
            await make_playback_mp3(paths["fullTrack"], settings.output_dir)
            == previews["fullTrack"]
        )

    previous_mtime = paths["replacedVocal"].stat().st_mtime_ns
    replacement = paths["replacedVocal"].with_suffix(".new")
    wav(replacement)
    replacement.write_bytes(replacement.read_bytes()[:-2] + b"\1\0")
    os.utime(replacement, ns=(previous_mtime + 1_000_000, previous_mtime + 1_000_000))
    replacement.replace(paths["replacedVocal"])
    assert (
        "replacedVocal"
        not in _render_result_urls(result, "http://testserver", settings)["playback"]
    )
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app), base_url="http://testserver"
    ) as client:
        assert (await client.get(rendered["playback"]["replacedVocal"])).status_code == 404
    fresh = await make_playback_mp3(paths["replacedVocal"], settings.output_dir)
    assert fresh != previews["replacedVocal"]
    assert not (settings.output_dir / previews["replacedVocal"]).exists()
    saved = capture_mix_artifact(result)
    assert saved["mixPlayback"] == previews["mixedTrack"]
    drop_mix_artifact(result)
    assert "mixedTrack" not in result["playback"]
    assert restore_mix_artifact(result, saved)
    assert result["playback"]["mixedTrack"] == previews["mixedTrack"]


async def test_generation_and_split_publish_previews(tmp_path: Path) -> None:
    settings = make_settings(tmp_path, mock_full_song_path=tmp_path / "source.wav")
    orchestrator = make_orchestrator(settings)
    wav(settings.mock_full_song_path)
    app = create_app(settings, orchestrator)
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app), base_url="http://testserver"
    ) as client:
        started = await client.post("/api/jobs", json={"prompt": "test"})
        job_id = started.json()["jobId"]
        await app.state.jobs[job_id].task
        generated = (await client.get(f"/api/jobs/{job_id}")).json()["result"]
        assert generated["fullTrack"].endswith(".wav")
        assert generated["playback"]["fullTrack"].endswith(".mp3")
        assert (await client.post(f"/api/jobs/{job_id}/split")).status_code == 202
        await app.state.jobs[job_id].split_task
        split = (await client.get(f"/api/jobs/{job_id}")).json()["result"]
        assert set(split["playback"]["stems"]) == set(split["stems"])
        assert all(url.endswith(".mp3") for url in split["playback"]["stems"].values())
        assert all(url.endswith(".wav") for url in split["stems"].values())
        preview_url = next(iter(split["playback"]["stems"].values()))
        preview_response = await client.get(preview_url, headers={"Range": "bytes=0-2"})
        assert preview_response.status_code == 206
        assert preview_response.headers["content-type"].startswith("audio/mpeg")
        assert (
            await client.delete(
                f"/api/jobs/{job_id}/stems/vocal",
                headers={"X-Audio-Revision": "1"},
            )
        ).status_code == 204
        deleted = (await client.get(f"/api/jobs/{job_id}")).json()["result"]
        assert "vocal" not in deleted["playback"]["stems"]
        assert (await client.get(split["stems"]["vocal"])).status_code == 404
        assert (await client.get(split["playback"]["stems"]["vocal"])).status_code == 404
        restored = (
            await client.put(
                f"/api/jobs/{job_id}/stems/vocal",
                headers={"X-Audio-Revision": "1"},
            )
        ).json()["result"]
        assert restored["playback"]["stems"]["vocal"] == split["playback"]["stems"]["vocal"]
        assert (await client.get(restored["stems"]["vocal"])).status_code == 200
        assert (await client.get(restored["playback"]["stems"]["vocal"])).status_code == 200


async def test_failed_stem_preview_does_not_fail_split(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    settings = make_settings(tmp_path, mock_full_song_path=tmp_path / "source.wav")
    orchestrator = make_orchestrator(settings)
    wav(settings.mock_full_song_path)
    app = create_app(settings, orchestrator)
    encoding_started = asyncio.Event()
    release_encoder = asyncio.Event()
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app), base_url="http://testserver"
    ) as client:
        started = await client.post("/api/jobs", json={"prompt": "test"})
        job_id = started.json()["jobId"]
        await app.state.jobs[job_id].task

        async def fail_preview(*_args: object) -> None:
            # 线上编码失败就是 make_playback_mp3 返回 None（它自己吞掉异常），这里照样复现。
            encoding_started.set()
            await release_encoder.wait()
            return None

        monkeypatch.setattr("app.main.make_playback_mp3", fail_preview)
        assert (await client.post(f"/api/jobs/{job_id}/split")).status_code == 202
        await encoding_started.wait()
        waiting = (await client.get(f"/api/jobs/{job_id}")).json()
        assert waiting["splitStatus"] == "running"
        assert orchestrator.capacity.active == 0
        release_encoder.set()
        await app.state.jobs[job_id].split_task
        completed = (await client.get(f"/api/jobs/{job_id}")).json()
        assert completed["splitStatus"] == "succeeded"
        assert all(url.endswith(".wav") for url in completed["result"]["stems"].values())
        assert "stems" not in completed["result"]["playback"]


async def test_partial_stem_preview_failure_keeps_split_successful(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    settings = make_settings(tmp_path, mock_full_song_path=tmp_path / "source.wav")
    orchestrator = make_orchestrator(settings)
    wav(settings.mock_full_song_path)
    app = create_app(settings, orchestrator)
    encode = make_playback_mp3

    async def flaky_preview(source: Path, output_dir: Path) -> str | None:
        if source.stem.endswith("_vocal"):
            return None
        if source.stem.endswith("_drums"):
            raise RuntimeError("encoding failed")
        return await encode(source, output_dir)

    monkeypatch.setattr("app.main.make_playback_mp3", flaky_preview)
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app), base_url="http://testserver"
    ) as client:
        started = await client.post("/api/jobs", json={"prompt": "test"})
        job_id = started.json()["jobId"]
        await app.state.jobs[job_id].task
        assert (await client.post(f"/api/jobs/{job_id}/split")).status_code == 202
        await app.state.jobs[job_id].split_task
        completed = (await client.get(f"/api/jobs/{job_id}")).json()

    assert completed["splitStatus"] == "succeeded"
    assert completed["stage"] == "completed"
    assert orchestrator.capacity.active == 0
    stems = completed["result"]["stems"]
    previews = completed["result"]["playback"]["stems"]
    assert set(stems) == set(STEM_NAMES)
    assert all(url.endswith(".wav") for url in stems.values())
    assert set(previews) == set(stems) - {"vocal", "drums"}
    assert all(url.endswith(".mp3") for url in previews.values())


@asynccontextmanager
async def blocked_preview_split(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> AsyncIterator[tuple[FastAPI, httpx.AsyncClient, str, GenerationOrchestrator]]:
    """把分轨卡在 preview 阶段：所有编码任务挂起，由调用方决定此时发什么请求。"""
    settings = make_settings(tmp_path, mock_full_song_path=tmp_path / "source.wav")
    orchestrator = make_orchestrator(settings)
    wav(settings.mock_full_song_path)
    app = create_app(settings, orchestrator)
    encoding_started = asyncio.Event()

    async def blocked_preview(*_args: object) -> str | None:
        encoding_started.set()
        await asyncio.Event().wait()
        return None

    monkeypatch.setattr("app.main.make_playback_mp3", blocked_preview)
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app), base_url="http://testserver"
    ) as client:
        started = await client.post("/api/jobs", json={"prompt": "test"})
        job_id = started.json()["jobId"]
        await app.state.jobs[job_id].task
        assert (await client.post(f"/api/jobs/{job_id}/split")).status_code == 202
        await encoding_started.wait()
        try:
            yield app, client, job_id, orchestrator
        finally:
            split_task = app.state.jobs[job_id].split_task
            if split_task is not None and not split_task.done():
                split_task.cancel()
                with suppress(asyncio.CancelledError):
                    await split_task


async def test_cancel_during_preview_only_stops_encoders(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    async with blocked_preview_split(tmp_path, monkeypatch) as (app, client, job_id, orchestrator):
        encoding = (await client.get(f"/api/jobs/{job_id}")).json()
        assert encoding["splitStatus"] == "running"
        assert encoding["operationStage"] == "preview"
        # 容量在编码前就让给了下一个 BS-RoFormer 任务（分支 reduce-audio-stress 的既定取舍），
        # 编码并发由 app.services.audio_files 的信号量单独限制。
        assert orchestrator.capacity.active == 0
        cancelled = (await client.patch(f"/api/jobs/{job_id}", json={"status": "cancelled"})).json()
        # preview 阶段的取消不写终态：PATCH 的响应与随后的最终状态必须一致。
        assert cancelled["splitStatus"] == "running"
        assert cancelled["operationStage"] == "preview"
        await app.state.jobs[job_id].split_task
        completed = (await client.get(f"/api/jobs/{job_id}")).json()

    assert completed["splitStatus"] == "succeeded"
    assert completed["stage"] == "completed"
    assert all(url.endswith(".wav") for url in completed["result"]["stems"].values())
    assert "stems" not in completed["result"]["playback"]


async def test_stem_delete_and_restore_rejected_while_preview_encodes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    async with blocked_preview_split(tmp_path, monkeypatch) as (app, client, job_id, _):
        rejected = await client.delete(
            f"/api/jobs/{job_id}/stems/vocal",
            headers={"X-Audio-Revision": "1"},
        )
        assert rejected.status_code == 409
        assert rejected.json()["message"] == "该歌曲正在拆轨，请稍后重试。"
        assert (
            await client.put(
                f"/api/jobs/{job_id}/stems/vocal",
                headers={"X-Audio-Revision": "1"},
            )
        ).status_code == 409
        await client.patch(f"/api/jobs/{job_id}", json={"status": "cancelled"})
        await app.state.jobs[job_id].split_task
        assert (
            await client.delete(
                f"/api/jobs/{job_id}/stems/vocal",
                headers={"X-Audio-Revision": "1"},
            )
        ).status_code == 204
        restored = (
            await client.put(
                f"/api/jobs/{job_id}/stems/vocal",
                headers={"X-Audio-Revision": "1"},
            )
        ).json()

    assert set(restored["result"]["stems"]) == set(STEM_NAMES)
    # 试听编码被取消过，恢复出来的分轨只有 WAV，没有可回填的 preview。
    assert "stems" not in restored["result"]["playback"]


async def test_mp3_stem_transients_stay_aligned_after_seeks(tmp_path: Path) -> None:
    if not shutil.which("ffmpeg"):
        pytest.skip("ffmpeg unavailable")
    rate = 48_000
    previews = []
    for index in range(4):
        source = tmp_path / f"stem{index}.wav"
        samples = array.array("h", [0]) * (rate * 4)
        for second in (0.25, 1.0, 2.0, 3.5):
            start = int(second * rate)
            for offset in range(96):
                samples[start + offset] = int(20_000 * math.exp(-offset / 24))
        with wave.open(str(source), "wb") as audio:
            audio.setparams((1, 2, rate, 0, "NONE", "not compressed"))
            audio.writeframes(samples.tobytes())
        preview = await make_playback_mp3(source, tmp_path)
        assert preview
        previews.append(tmp_path / preview)

    for seek in (0, 0.8, 1.8, 3.3):
        onsets = []
        for preview in previews:
            command = ["ffmpeg", "-v", "error"]
            if seek:
                command += ["-ss", str(seek)]
            command += ["-i", str(preview), "-t", "0.55", "-f", "f32le", "-ac", "1", "-"]
            # 16 次解码都在子进程里，但仍然会卡住事件循环，交给线程池跑。
            decoded = await asyncio.to_thread(
                subprocess.run, command, check=True, capture_output=True
            )
            samples = array.array("f")
            samples.frombytes(decoded.stdout)
            onsets.append(next(i for i, sample in enumerate(samples) if abs(sample) > 0.15))
        # 5 ms：LAME 的 encoder delay/priming 与不同 ffmpeg 构建的取整差异都在这之内，
        # 而真正的错位（整帧/整段填充，几十毫秒）依然会被抓住。
        assert max(onsets) - min(onsets) <= rate // 200


async def test_failed_encoding_leaves_wav_and_no_preview(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "full.wav"
    wav(source)
    original = source.read_bytes()

    def unavailable() -> None:
        raise RuntimeError("ffmpeg unavailable")

    monkeypatch.setattr("app.services.stems.prepare_ffmpeg_environment", unavailable)
    assert await make_playback_mp3(source, tmp_path) is None
    assert source.read_bytes() == original
    assert not list(tmp_path.rglob("*.mp3"))


async def test_encoder_timeout_kills_child_and_removes_temp(tmp_path: Path, monkeypatch) -> None:
    source = tmp_path / "full.wav"
    wav(source)
    original = source.read_bytes()
    spawned = []
    create = asyncio.create_subprocess_exec

    async def sleeping_encoder(*_args, **kwargs):
        process = await create(sys.executable, "-c", "import time; time.sleep(10)", **kwargs)
        spawned.append(process)
        return process

    monkeypatch.setattr("app.services.audio_files.PLAYBACK_ENCODE_TIMEOUT_SECONDS", 0.01)
    monkeypatch.setattr("app.services.audio_files.asyncio.create_subprocess_exec", sleeping_encoder)
    assert await make_playback_mp3(source, tmp_path) is None
    assert spawned[0].returncode is not None
    assert source.read_bytes() == original
    assert not list((tmp_path / "playtrack").iterdir())


async def test_cancelled_encoder_kills_child_and_removes_temp(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """preview 阶段的取消就是取消编码器：子进程和临时 MP3 都不许留下。"""
    source = tmp_path / "full.wav"
    wav(source)
    original = source.read_bytes()
    spawned = []
    process_started = asyncio.Event()
    create = asyncio.create_subprocess_exec

    async def sleeping_encoder(*_args, **kwargs):
        process = await create(sys.executable, "-c", "import time; time.sleep(10)", **kwargs)
        spawned.append(process)
        process_started.set()
        return process

    monkeypatch.setattr("app.services.audio_files.asyncio.create_subprocess_exec", sleeping_encoder)
    encoding = asyncio.create_task(make_playback_mp3(source, tmp_path))
    await process_started.wait()
    assert list((tmp_path / "playtrack").iterdir())
    encoding.cancel()
    with suppress(asyncio.CancelledError):
        await encoding
    assert spawned[0].returncode is not None
    assert source.read_bytes() == original
    assert not list((tmp_path / "playtrack").iterdir())


async def test_replacement_waveform_and_delete_restore_preview(tmp_path: Path) -> None:
    class CopyVoice:
        loaded = True

        async def convert(self, source: Path, target: Path, **_kwargs: object) -> None:
            shutil.copy2(source, target)

    settings = make_settings(tmp_path)
    app = create_app(settings, make_orchestrator(settings), CopyVoice())
    folder = settings.output_dir / "jobs/j/song_1"
    full, vocal = folder / "full.wav", folder / "vocal.wav"
    wav(full)
    wav(vocal)
    result = {
        "success": True,
        "jobId": "j",
        "prompt": "test",
        "durationMinutes": "auto",
        "structuredPrompt": "test",
        "lyrics": "test",
        "count": 1,
        "alternatives": [],
        "fullTrack": full.relative_to(settings.output_dir).as_posix(),
        "stems": {"vocal": vocal.relative_to(settings.output_dir).as_posix()},
        "stemUrls": [vocal.relative_to(settings.output_dir).as_posix()],
        "waveforms": {},
        "splitEnabled": True,
        "debug": {},
    }
    job = GenerationJob(job_id="j", prompt="test", status="succeeded", result=result)
    app.state.jobs["j"] = job
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app), base_url="http://testserver"
    ) as client:
        accepted = await client.post("/api/jobs/j/replace")
        assert accepted.status_code == 202
        await job.replace_task
        completed = (await client.get("/api/jobs/j")).json()["result"]
        assert len(completed["waveforms"]["replaced"]) == 640
        assert completed["playback"]["replacedVocal"].endswith(".mp3")
        original_preview = completed["playback"]["replacedVocal"]
        filename = completed["replacedVocal"]
        payload = {"filename": filename, "job_id": "j", "song": "0"}
        assert (
            await client.request("DELETE", "/api/voice/result", data=payload)
        ).status_code == 200
        deleted = (await client.get("/api/jobs/j")).json()["result"]
        assert "replacedVocal" not in deleted
        assert "replacedVocal" not in deleted["playback"]
        assert "replaced" not in deleted["waveforms"]
        assert (await client.get(filename)).status_code == 404
        assert (await client.get(original_preview)).status_code == 404
        # Undo metadata and both formats survive a backend restart.
        app.state.jobs = load_jobs(settings.output_dir)
        assert (await client.request("PUT", "/api/voice/result", data=payload)).status_code == 200
        restored = (await client.get("/api/jobs/j")).json()["result"]
        assert restored["playback"]["replacedVocal"] == original_preview
        assert len(restored["waveforms"]["replaced"]) == 640
        assert (await client.get(restored["replacedVocal"])).status_code == 200
        assert (await client.get(original_preview)).status_code == 200


@pytest.mark.parametrize("legacy_mp3", [False, True])
async def test_download_formats_consistent_across_responses_restart_and_expiry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, legacy_mp3: bool
) -> None:
    settings = make_settings(tmp_path, song_retention_enabled=True, song_retention_dry_run=False)
    app = create_app(settings, make_orchestrator(settings))
    songs = []
    for song in (1, 2):
        folder = settings.output_dir / f"jobs/download/song_{song}"
        paths = {key: folder / f"{key}.wav" for key in ("fullTrack", "mixedTrack", "replacedVocal")}
        paths["vocal"] = folder / "vocal.wav"
        for key, path in paths.items():
            wav(path)
            preview = await make_playback_mp3(path, settings.output_dir)
            assert preview
            if key in {"fullTrack", "vocal"} and legacy_mp3:
                paths[key] = path.with_suffix(".mp3")
                shutil.copy2(settings.output_dir / preview, paths[key])
                path.unlink()
        relative = {
            key: path.relative_to(settings.output_dir).as_posix() for key, path in paths.items()
        }
        songs.append(
            {
                "songNumber": song,
                **{key: relative[key] for key in ("fullTrack", "mixedTrack", "replacedVocal")},
                "_replacedVocalModel": app.state.rvc_model_fingerprint,
                "stems": {"vocal": relative["vocal"]},
                "stemUrls": [relative["vocal"]],
                "waveforms": {},
                "splitEnabled": True,
                "debug": {},
                # A missing or stale metadata URL must reuse the current file on disk.
                "playback": {"fullTrack": "jobs/download/song_1/playtrack/stale.mp3"},
            }
        )
    job = GenerationJob(
        job_id="download",
        prompt="test",
        status="succeeded",
        stage="completed",
        result={
            **songs[0],
            "success": True,
            "jobId": "download",
            "prompt": "test",
            "durationMinutes": "auto",
            "structuredPrompt": "test",
            "lyrics": "test",
            "count": 2,
            "alternatives": songs[1:],
        },
    )
    job.save(settings.output_dir)
    app.state.jobs = load_jobs(settings.output_dir)
    # 加载时旧版 mixedTrack 迁成独立的替换后那首；之后读取不得再改动结果。
    stored_result = copy.deepcopy(app.state.jobs["download"].result)
    assert all("mixed" in output for output in [stored_result, *stored_result["alternatives"]])

    async def unexpected_encode(*_args, **_kwargs):
        raise AssertionError("reading results must not encode audio")

    monkeypatch.setattr(asyncio, "create_subprocess_exec", unexpected_encode)
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app), base_url="http://testserver"
    ) as client:
        detail = (await client.get("/api/jobs/download")).json()["result"]
        history = (await client.get("/api/jobs")).json()["jobs"][0]["result"]
        sse = await client.get("/api/jobs/download/events")
        terminal = json.loads(
            next(line[6:] for line in sse.text.splitlines() if line.startswith("data: "))
        )
        cached = await client.post("/api/jobs/download/replace?song=1")
        assert cached.status_code == 200
        for payload in (history, terminal["result"], cached.json()["result"]):
            for expected, actual in zip(
                [detail, *detail["alternatives"]], [payload, *payload["alternatives"]], strict=True
            ):
                for key in (
                    "fullTrack",
                    "mixedTrack",
                    "replacedVocal",
                    "stems",
                    "stemUrls",
                    "playback",
                ):
                    assert actual[key] == expected[key]
        assert app.state.jobs["download"].result == stored_result
        urls = []
        for output in [detail, *detail["alternatives"]]:
            assert f"song_{output['songNumber']}/" in output["fullTrack"]
            for key in ("fullTrack", "mixedTrack", "replacedVocal"):
                source, mp3 = output[key], output["playback"][key]
                if key == "fullTrack" and legacy_mp3:
                    assert source == mp3
                else:
                    assert source.endswith(".wav")
                assert mp3.endswith(".mp3")
                urls.extend((source, mp3))
            urls.extend((output["stems"]["vocal"], output["playback"]["stems"]["vocal"]))
            if legacy_mp3:
                assert output["stems"]["vocal"] == output["playback"]["stems"]["vocal"]
            else:
                assert output["stems"]["vocal"].endswith(".wav")
        keys = ("fullTrack", "mixedTrack", "replacedVocal")
        assert {detail["playback"][key] for key in keys}.isdisjoint(
            detail["alternatives"][0]["playback"][key] for key in keys
        )
        for url in urls:
            response = await client.get(url + "?download=1")
            assert response.status_code == 200
            if url.endswith(".wav"):
                assert response.content[:4] == b"RIFF" and response.content[8:12] == b"WAVE"
                assert response.headers["content-type"].startswith("audio/wav")
            else:
                assert response.content[:3] == b"ID3" or response.content[0] == 0xFF
                assert response.headers["content-type"].startswith("audio/mpeg")
        app.state.jobs["download"].created_at = (
            datetime.now(timezone.utc) - timedelta(days=4)
        ).isoformat()
        app.state.jobs["download"].save(settings.output_dir)
        for url in urls:
            assert (await client.get(url)).status_code == 404
        assert (await client.get("/api/jobs")).json()["jobs"] == []
        assert (await client.get("/api/jobs/download")).status_code == 404
        assert (await client.get("/api/jobs/download/events")).status_code == 404
        await cleanup_expired_jobs(app, reason="test")
        assert not (settings.output_dir / "jobs/download").exists()
