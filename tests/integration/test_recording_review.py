from __future__ import annotations

import asyncio

import pytest

from app.core.errors import GenerationError
from app.main import create_app
from tests.helpers import client, make_orchestrator
from tests.integration.test_recording_api import recording_app
from tests.integration.test_recording_position import samples
from tests.integration.test_reference_api import sine_wav


async def upload(http, tmp_path, settings, job):
    sine_wav(settings.output_dir / job.result["fullTrack"], 4)
    sine_wav(settings.output_dir / job.result["stems"]["vocal"], 4)
    job.result.pop("replacedVocal", None)
    response = await http.post(
        f"/api/jobs/{job.job_id}/recordings",
        data={"name": "review"},
        files={"file": ("take.wav", sine_wav(tmp_path / "take.wav", 0.5), "audio/wav")},
    )
    assert response.status_code == 201, response.text
    return response.json()


async def commit(http, job, tracks, editor="tracks"):
    response = await http.post(
        f"/api/jobs/{job.job_id}/mix",
        json={
            "commit": True,
            "audioRevision": job.result.get("audioRevision", 0),
            "editor": editor,
            "tracks": tracks,
        },
    )
    assert response.status_code == 202, response.text
    await job.mix_task


async def move(http, job, stem, position):
    return await http.patch(
        f"/api/jobs/{job.job_id}/stems/{stem}",
        json={"position": position},
        headers={"X-Audio-Revision": str(job.result.get("audioRevision", 0))},
    )


@pytest.mark.parametrize("completed", [False, True])
@pytest.mark.parametrize("cancelled", [False, True])
async def test_failed_or_cancelled_mix_does_not_change_committed_recordings(
    tmp_path, monkeypatch, completed, cancelled
):
    app, settings, job = recording_app(tmp_path)
    async with client(app) as http:
        take = await upload(http, tmp_path, settings, job)
        stem = take["id"]
        recording = {"id": stem, "source": "stem", "stemId": stem, "gainDb": 0}
        if completed:
            await commit(
                http,
                job,
                [recording, {"id": "vocal", "source": "stem", "stemId": "vocal", "gainDb": 0}],
            )
            assert job.mix_status == "succeeded"

        async def fail(*args, **kwargs):
            if cancelled:
                raise asyncio.CancelledError
            raise GenerationError("test failure")

        monkeypatch.setattr("app.main.mix_tracks", fail)
        await commit(
            http,
            job,
            [{"id": "vocal", "source": "stem", "stemId": "vocal", "gainDb": 0}]
            if completed
            else [recording],
        )
        assert job.mix_status == ("cancelled" if cancelled else "failed")
        response = await move(http, job, stem, 0.25)
        assert response.status_code == (409 if completed else 200), response.text


async def test_replacement_commit_locks_target_only_and_preserves_raw_pcm(tmp_path):
    app, settings, job = recording_app(tmp_path)
    async with client(app) as http:
        take = await upload(http, tmp_path, settings, job)
        stem = take["id"]
        original = samples((await http.get(take["audioUrl"])).content)[2]
        recording = {"id": stem, "source": "stem", "stemId": stem, "gainDb": -6}
        await commit(http, job, [recording])
        assert job.mix_status == "succeeded"
        job.result["replacedVocal"] = job.result["fullTrack"]
        shifted = await move(http, job, stem, 0.5)
        assert shifted.status_code == 200, shifted.text
        rate, size, moved = samples(
            (await http.get(shifted.json()["result"]["stems"][stem])).content
        )
        assert moved[round(0.5 * rate) * size :] == original
        await commit(http, job, [recording], editor="replace")
        assert job.mix_status == "succeeded"
        # Original keeps its earlier committed state; target has its own committed ids.
        assert stem in job.result["committedRecordingIds"]
        assert stem in job.result["mixed"]["committedRecordingIds"]


async def test_uncommitted_original_remains_editable_after_replacement_commit(tmp_path):
    app, settings, job = recording_app(tmp_path)
    async with client(app) as http:
        take = await upload(http, tmp_path, settings, job)
        stem = take["id"]
        job.result["replacedVocal"] = job.result["stems"][stem]
        await commit(
            http,
            job,
            [{"id": stem, "source": "stem", "stemId": stem, "gainDb": 0}],
            editor="replace",
        )
        assert job.mix_status == "succeeded"
        assert (await move(http, job, stem, 0.25)).status_code == 200
        assert stem in job.result["mixed"]["committedRecordingIds"]


async def test_moves_collect_obsolete_audio_but_keep_immutable_source_and_projection_light(
    tmp_path,
):
    app, settings, job = recording_app(tmp_path)
    async with client(app) as http:
        take = await upload(http, tmp_path, settings, job)
        stem = take["id"]
        for position in [0.25, 0.5, 1, 1.5, 0]:
            response = await move(http, job, stem, position)
            assert response.status_code == 200, response.text
        project = job.result
        directory = (settings.output_dir / project["stems"][stem]).parent
        assert len(list(directory.glob(f"{stem}*.wav"))) == 1
        assert len(list((directory / "playtrack").glob(f"{stem}*.mp3"))) == 1
        assert (settings.output_dir / project["recordingClips"][stem]["source"]).is_file()
        rows = (await http.get("/api/jobs")).json()
        row = next(row for row in rows["jobs"] if row["jobId"] == job.job_id)
        assert "waveform" not in row["result"]["recordingClips"][stem]
        assert len(project["recordingClips"][stem]["waveform"]) == 640
    restarted = create_app(settings, make_orchestrator(settings))
    async with client(restarted) as http:
        detail = (await http.get(f"/api/jobs/{job.job_id}")).json()["result"]
        assert (await http.get(detail["stems"][stem])).status_code == 200


async def test_missing_clip_waveform_recovers_from_immutable_audio(tmp_path, monkeypatch):
    app, settings, job = recording_app(tmp_path)
    from app.services.waveforms import extract_waveforms

    async def no_waveform(*args, **kwargs):
        return {}

    monkeypatch.setattr("app.main.extract_waveforms", no_waveform)
    async with client(app) as http:
        take = await upload(http, tmp_path, settings, job)
        assert take["waveform"] == []
        monkeypatch.setattr("app.main.extract_waveforms", extract_waveforms)
        response = await move(http, job, take["id"], 0.5)
        assert response.status_code == 200
        assert len(response.json()["result"]["recordingClips"][take["id"]]["waveform"]) == 640
        job.result["recordingClips"][take["id"]]["waveform"] = []

        async def raw_waveform_unavailable(paths, **kwargs):
            if any(path.parent.name == "recording_sources" for path in paths.values()):
                return {}
            return await extract_waveforms(paths, **kwargs)

        monkeypatch.setattr("app.main.extract_waveforms", raw_waveform_unavailable)
        response = await move(http, job, take["id"], 0.75)
        assert response.status_code == 200
        assert response.json()["result"]["recordingClips"][take["id"]]["waveform"] == []
        monkeypatch.setattr("app.main.extract_waveforms", extract_waveforms)
        response = await move(http, job, take["id"], 1)
        assert len(response.json()["result"]["recordingClips"][take["id"]]["waveform"]) == 640


async def test_old_rendered_recording_without_raw_source_is_not_silently_recropped(tmp_path):
    app, settings, job = recording_app(tmp_path)
    async with client(app) as http:
        take = await upload(http, tmp_path, settings, job)
        stem = take["id"]
        await commit(http, job, [{"id": stem, "source": "stem", "stemId": stem, "gainDb": -6}])
        job.result.pop("committedRecordingIds")
        job.result.pop("recordingClips")
        job.result["replacedVocal"] = job.result["fullTrack"]
        snapshot = job.result["stems"][stem]
        response = await move(http, job, stem, 0.5)
        assert response.status_code == 409
        assert job.result["stems"][stem] == snapshot


async def test_collecting_old_versions_preserves_other_candidate_references(tmp_path):
    app, settings, job = recording_app(tmp_path)
    async with client(app) as http:
        take = await upload(http, tmp_path, settings, job)
        stem = take["id"]
        other = job.result["alternatives"][0]
        old_audio = job.result["stems"][stem]
        old_preview = job.result["playback"]["stems"][stem]
        other["stems"]["recording_shared"] = old_audio
        other.setdefault("playback", {}).setdefault("stems", {})["recording_shared"] = old_preview
        assert (await move(http, job, stem, 0.5)).status_code == 200
        assert (settings.output_dir / old_audio).is_file()
        assert (settings.output_dir / old_preview).is_file()
