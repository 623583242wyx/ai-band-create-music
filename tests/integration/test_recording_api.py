from __future__ import annotations

import asyncio
import copy
import subprocess

import pytest

from app.main import create_app
from app.services.audio_files import output_path_from_url
from tests.helpers import client, make_orchestrator, make_settings
from tests.integration.test_mix_api import seed_job
from tests.integration.test_reference_api import sine_wav


def recording_app(tmp_path, **updates):
    settings = make_settings(tmp_path, recording_max_upload_mb=1, **updates)
    app = create_app(settings, make_orchestrator(settings))
    return app, settings, seed_job(app, settings, count=2)


async def test_upload_preview_delete_restore_commit_and_restart(tmp_path):
    app, settings, job = recording_app(tmp_path)
    other_before = copy.deepcopy(job.result)
    audio = sine_wav(tmp_path / "recording.wav", 1)
    sine_wav(settings.output_dir / job.result["alternatives"][0]["fullTrack"], 1)
    async with client(app) as http:
        uploaded = await http.post(
            f"/api/jobs/{job.job_id}/recordings?song=1",
            data={"name": "  我的和声  "},
            files={"file": ("../../vocal.wav", audio, "application/octet-stream")},
        )
        assert uploaded.status_code == 201, uploaded.text
        track = uploaded.json()
        stem_id = track["id"]
        assert stem_id.startswith("recording_") and track["name"] == "我的和声"
        assert track["audioRevision"] == 1
        assert track["durationSeconds"] == pytest.approx(1, abs=0.01)
        assert len(track["waveform"]) == 640
        preview = await http.get(track["playbackUrl"], headers={"Range": "bytes=0-31"})
        assert preview.status_code == 206 and len(preview.content) == 32
        full_audio = await http.get(track["audioUrl"])
        assert full_audio.content[:4] == b"RIFF"
        output = (await http.get(f"/api/jobs/{job.job_id}")).json()["result"]["alternatives"][0]
        assert output["stems"][stem_id] == track["audioUrl"]
        assert output["stemNames"] == {stem_id: "我的和声"}
        assert output["waveforms"][stem_id] == track["waveform"]
        assert output["playback"]["stems"][stem_id] == track["playbackUrl"]
        assert {k: v for k, v in job.result.items() if k != "alternatives"} == {
            k: v for k, v in other_before.items() if k != "alternatives"
        }
        deletion = await http.delete(
            f"/api/jobs/{job.job_id}/stems/{stem_id}?song=1",
            headers={"X-Audio-Revision": "1"},
        )
        assert deletion.status_code == 204
        assert stem_id not in job.result["alternatives"][0]["stemNames"]
        restored = await http.put(
            f"/api/jobs/{job.job_id}/stems/{stem_id}?song=1",
            headers={"X-Audio-Revision": "1"},
        )
        assert restored.status_code == 200
        assert restored.json()["result"]["alternatives"][0]["stemNames"][stem_id] == "我的和声"
        mixed = await http.post(
            f"/api/jobs/{job.job_id}/mix?song=1",
            json={
                "commit": True,
                "audioRevision": 1,
                "editor": "tracks",
                "tracks": [{"id": stem_id, "source": "stem", "stemId": stem_id, "gainDb": -6}],
            },
        )
        assert mixed.status_code == 202, mixed.text
        await job.mix_task
        assert job.mix_status == "succeeded", job.mix_error
        output = job.result["alternatives"][0]
        assert set(output["stems"]) == {stem_id}
        assert output["stemNames"] == {stem_id: "我的和声"}
        assert output["audioRevision"] == 2
    relaunched = create_app(settings, make_orchestrator(settings))
    async with client(relaunched) as http:
        output = (await http.get(f"/api/jobs/{job.job_id}")).json()["result"]["alternatives"][0]
        assert output["stemNames"][stem_id] == "我的和声"
        assert output["audioRevision"] == 2
    assert not list(settings.output_dir.rglob(".recording-*"))
    assert app.state.orchestrator.capacity.active == 0


@pytest.mark.parametrize("format,codec", [("webm", "libopus"), ("mp4", "aac")])
async def test_browser_recording_formats_are_decoded(tmp_path, format, codec):
    app, _, job = recording_app(tmp_path)
    sine_wav(tmp_path / "input.wav", 0.5)
    encoded = tmp_path / f"browser.{format}"
    subprocess.run(
        ["ffmpeg", "-v", "error", "-i", str(tmp_path / "input.wav"), "-c:a", codec, str(encoded)],
        check=True,
    )
    async with client(app) as http:
        response = await http.post(
            f"/api/jobs/{job.job_id}/recordings",
            data={"name": "浏览器录音"},
            files={"file": ("blob", encoded.read_bytes(), "application/octet-stream")},
        )
    assert response.status_code == 201, response.text


@pytest.mark.parametrize(
    "name,content,status",
    [
        (" ", b"audio", 400),
        ("名字\x00", b"audio", 400),
        ("名字\u202e", b"audio", 400),
        ("名\u200b字", b"audio", 400),
        ("录音", b"", 400),
        ("录音", b"not audio", 400),
        ("录音", b"a" * (1024 * 1024 + 1), 413),
        ("x" * 101, b"audio", 422),
    ],
)
async def test_invalid_upload_does_not_change_project(tmp_path, name, content, status):
    app, settings, job = recording_app(tmp_path)
    previous = copy.deepcopy(job.result)
    async with client(app) as http:
        response = await http.post(
            f"/api/jobs/{job.job_id}/recordings",
            data={"name": name},
            files={"file": ("recording.wav", content, "audio/wav")},
        )
    assert response.status_code == status, response.text
    assert job.result == previous
    assert not list(settings.output_dir.rglob("*recording*"))
    assert app.state.orchestrator.capacity.active == 0


async def test_duration_limit_rejects_instead_of_silently_trimming(tmp_path):
    app, settings, job = recording_app(tmp_path, max_duration_minutes=1, default_duration_minutes=1)
    audio = sine_wav(tmp_path / "long.wav", 61)
    async with client(app) as http:
        response = await http.post(
            f"/api/jobs/{job.job_id}/recordings",
            data={"name": "过长录音"},
            files={"file": ("blob", audio)},
        )
    assert response.status_code == 400 and "60 秒" in response.json()["message"]
    assert not list(settings.output_dir.rglob("*recording*"))


@pytest.mark.parametrize("chunked", [False, True])
async def test_request_body_limit_also_applies_before_multipart_parsing(tmp_path, chunked):
    app, _, job = recording_app(tmp_path)
    payload = (
        b'--oversized\r\nContent-Disposition: form-data; name="file"; filename="big.wav"\r\n'
        b"Content-Type: audio/wav\r\n\r\n" + b"a" * (2 * 1024 * 1024 + 1) + b"\r\n--oversized--\r\n"
    )

    async def chunks():
        for offset in range(0, len(payload), 64 * 1024):
            yield payload[offset : offset + 64 * 1024]

    async with client(app) as http:
        response = await http.post(
            f"/api/jobs/{job.job_id}/recordings",
            content=chunks() if chunked else payload,
            headers={"Content-Type": "multipart/form-data; boundary=oversized"},
        )
    assert response.status_code == 413


async def test_upload_rejects_playlists(tmp_path):
    app, _, job = recording_app(tmp_path)
    async with client(app) as http:
        response = await http.post(
            f"/api/jobs/{job.job_id}/recordings",
            data={"name": "录音"},
            files={
                "file": (
                    "audio.wav",
                    b"#EXTM3U\n#EXT-X-TARGETDURATION:1\n"
                    b"#EXTINF:1,\nfile:///etc/passwd\n#EXT-X-ENDLIST\n",
                )
            },
        )
    assert response.status_code == 400


@pytest.mark.parametrize(
    "case,status",
    [
        ("missing", 404),
        ("song", 404),
        ("mixed", 404),
        ("pending", 409),
        ("unsplit", 409),
        ("stale", 409),
        ("busy", 409),
        ("capacity", 429),
    ],
)
async def test_upload_respects_project_and_operation_state(tmp_path, case, status):
    app, _, job = recording_app(tmp_path)
    url = f"/api/jobs/{job.job_id}/recordings"
    if case == "missing":
        url = "/api/jobs/absent/recordings"
    elif case == "song":
        url += "?song=2"
    elif case == "mixed":
        url += "?variant=mixed"
    elif case == "pending":
        job.status = "pending"
    elif case == "unsplit":
        job.result["splitEnabled"] = False
    elif case == "stale":
        job.result["audioRevision"] = 1
    elif case == "busy":
        job.split_status = "running"
    elif case == "capacity":
        await app.state.orchestrator.capacity.acquire()
    previous = copy.deepcopy(job.result)
    async with client(app) as http:
        response = await http.post(url, data={"name": "录音"}, files={"file": ("blob", b"audio")})
    assert response.status_code == status, response.text
    assert job.result == previous


async def test_upload_rechecks_revision_after_decoding(tmp_path, monkeypatch):
    app, settings, job = recording_app(tmp_path)
    started, release = asyncio.Event(), asyncio.Event()

    async def blocked_decode(source, target, max_seconds):
        started.set()
        await release.wait()
        target.write_bytes(sine_wav(tmp_path / "decoded.wav", 0.1))
        return 0.1

    monkeypatch.setattr("app.main.prepare_recording_audio", blocked_decode)
    async with client(app) as http:
        upload = asyncio.create_task(
            http.post(
                f"/api/jobs/{job.job_id}/recordings",
                data={"name": "录音"},
                files={"file": ("blob", b"input")},
            )
        )
        await started.wait()
        job.result["audioRevision"] = 1
        release.set()
        response = await upload
    assert response.status_code == 409
    assert "stemNames" not in job.result
    assert not list(settings.output_dir.rglob("*recording*"))


async def test_failed_metadata_save_rolls_back_files_and_project(tmp_path, monkeypatch):
    app, settings, job = recording_app(tmp_path)
    previous = copy.deepcopy(job.result)

    def fail_save(self, directory):
        raise OSError("disk full")

    monkeypatch.setattr(type(job), "save", fail_save)
    audio = sine_wav(tmp_path / "recording.wav", 0.1)
    async with client(app) as http:
        response = await http.post(
            f"/api/jobs/{job.job_id}/recordings",
            data={"name": "录音"},
            files={"file": ("blob", audio)},
        )
    assert response.status_code == 500
    assert job.result == previous
    assert not list(settings.output_dir.rglob("*recording*"))


async def test_upload_to_mixed_project_is_independent(tmp_path):
    app, settings, job = recording_app(tmp_path)
    original = copy.deepcopy(job.result)
    job.result["mixed"] = copy.deepcopy(job.result["alternatives"][0])
    audio = sine_wav(tmp_path / "input.wav", 0.1)
    async with client(app) as http:
        response = await http.post(
            f"/api/jobs/{job.job_id}/recordings?variant=mixed",
            data={"name": "第二版录音"},
            files={"file": ("blob", audio)},
        )
    assert response.status_code == 201, response.text
    track = response.json()
    assert job.result["mixed"]["stemNames"][track["id"]] == "第二版录音"
    assert {k: v for k, v in job.result.items() if k != "mixed"} == original
    assert output_path_from_url(track["audioUrl"], settings.output_dir).name.startswith("mixed_")
