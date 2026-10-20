from __future__ import annotations

import copy
import io
import subprocess
import wave

import pytest

from app.main import create_app
from app.services.recordings import move_recording_audio
from tests.helpers import client, make_orchestrator
from tests.integration.test_recording_api import recording_app
from tests.integration.test_reference_api import sine_wav


def samples(data):
    with wave.open(io.BytesIO(data), "rb") as audio:
        return (
            audio.getframerate(),
            audio.getnchannels() * audio.getsampwidth(),
            audio.readframes(audio.getnframes()),
        )


async def test_alignment_preserves_pcm_preview_restart_undo_and_committed_mix(tmp_path):
    app, settings, job = recording_app(tmp_path)
    master = settings.output_dir / job.result["fullTrack"]
    sine_wav(master, 4)
    other_song = copy.deepcopy(job.result["alternatives"])
    async with client(app) as http:
        uploaded = await http.post(
            f"/api/jobs/{job.job_id}/recordings",
            data={"name": "和声"},
            files={"file": ("take.wav", sine_wav(tmp_path / "take.wav", 0.5), "audio/wav")},
        )
        assert uploaded.status_code == 201, uploaded.text
        take = uploaded.json()
        stem = take["id"]
        original = await http.get(take["audioUrl"])
        rate, size, pcm = samples(original.content)
        position = f"/api/jobs/{job.job_id}/stems/{stem}"
        for revision, start in [(1, 1.25), (2, 0.25), (3, 0)]:
            response = await http.patch(
                position, json={"position": start}, headers={"X-Audio-Revision": str(revision)}
            )
            assert response.status_code == 200, response.text
            output = response.json()["result"]
            assert output["audioRevision"] == revision + 1
            assert output["recordingClips"][stem] == {
                "position": start,
                "duration": 0.5,
                "waveform": take["waveform"],
                "source": job.result["recordingClips"][stem]["source"],
            }
            shifted = await http.get(output["stems"][stem])
            _, _, data = samples(shifted.content)
            offset = round(start * rate) * size
            assert data[:offset] == bytes(offset)
            assert data[offset:] == pcm, (
                "repeated moves preserve exact samples without accumulating padding"
            )
            preview = await http.get(output["playback"]["stems"][stem])
            assert preview.status_code == 200 and len(preview.content) > 0
        assert job.result["alternatives"] == other_song
        assert (
            await http.patch(position, json={"position": 1}, headers={"X-Audio-Revision": "1"})
        ).status_code == 409
        assert (
            await http.patch(position, json={"position": 3.75}, headers={"X-Audio-Revision": "4"})
        ).status_code == 400
        assert job.result["audioRevision"] == 4
        assert (await http.delete(position, headers={"X-Audio-Revision": "4"})).status_code == 204
        assert stem not in job.result.get("recordingClips", {})
        assert (await http.put(position, headers={"X-Audio-Revision": "4"})).status_code == 200
        assert job.result["recordingClips"][stem]["duration"] == 0.5
        # The committed mix reads the same physically aligned WAV as the preview.
        mixed = await http.post(
            f"/api/jobs/{job.job_id}/mix",
            json={
                "commit": True,
                "audioRevision": 4,
                "editor": "tracks",
                "tracks": [{"id": stem, "source": "stem", "stemId": stem, "gainDb": 0}],
            },
        )
        assert mixed.status_code == 202, mixed.text
        await job.mix_task
        assert job.mix_status == "succeeded", job.mix_error
        assert job.result["recordingClips"][stem]["duration"] == 0.5
        moved_again = await http.patch(
            position,
            json={"position": 1.5},
            headers={"X-Audio-Revision": str(job.result["audioRevision"])},
        )
        assert moved_again.status_code == 409, moved_again.text
        assert job.result["recordingClips"][stem]["position"] == 0
        # A new pending vocal replacement opens a new arranging stage.
        job.result["replacedVocal"] = job.result["stems"][stem]
        unlocked = await http.patch(
            position,
            json={"position": 1.5},
            headers={"X-Audio-Revision": str(job.result["audioRevision"])},
        )
        assert unlocked.status_code == 200, unlocked.text
        job.result.pop("replacedVocal")
        job.save(settings.output_dir)
    relaunched = create_app(settings, make_orchestrator(settings))
    async with client(relaunched) as http:
        output = (await http.get(f"/api/jobs/{job.job_id}")).json()["result"]
        assert output["recordingClips"][stem] == {
            "position": 1.5,
            "duration": 0.5,
            "waveform": take["waveform"],
            "source": job.result["recordingClips"][stem]["source"],
        }
    assert not list(settings.output_dir.rglob(".recording-*"))
    assert app.state.orchestrator.capacity.active == 0


async def test_preview_failure_does_not_publish_a_half_moved_recording(tmp_path, monkeypatch):
    app, settings, job = recording_app(tmp_path)
    sine_wav(settings.output_dir / job.result["fullTrack"], 4)
    async with client(app) as http:
        uploaded = await http.post(
            f"/api/jobs/{job.job_id}/recordings",
            data={"name": "录音"},
            files={"file": ("take.wav", sine_wav(tmp_path / "take.wav", 0.5), "audio/wav")},
        )
        stem = uploaded.json()["id"]
        previous = copy.deepcopy(job.result)

        async def fail_preview(*args):
            return None

        monkeypatch.setattr("app.main.make_playback_mp3", fail_preview)
        response = await http.patch(
            f"/api/jobs/{job.job_id}/stems/{stem}",
            json={"position": 1},
            headers={"X-Audio-Revision": "1"},
        )
        assert response.status_code == 500
        assert job.result == previous
    assert app.state.orchestrator.capacity.active == 0
    assert not list(settings.output_dir.rglob(".recording-*"))


def test_moving_a_padded_track_has_exact_bounds_and_rejects_invalid_pcm(tmp_path):
    source = tmp_path / "input.wav"
    sine_wav(source, 0.5)
    output = tmp_path / "output.wav"
    clip = move_recording_audio(source, output, 0.5, None, 1, 360)
    assert clip == {"position": 0.5, "duration": 0.5}
    with pytest.raises(ValueError):
        move_recording_audio(source, output, 0.5001, None, 1, 360)
    with pytest.raises(ValueError):
        move_recording_audio(source, output, -1, None, 1, 360)


@pytest.mark.parametrize("song,variant", [(1, "original"), (0, "mixed")])
async def test_alignment_changes_only_the_selected_candidate_and_variant(tmp_path, song, variant):
    app, settings, job = recording_app(tmp_path)
    selected = job.result if song == 0 else job.result["alternatives"][0]
    if variant == "mixed":
        selected["mixed"] = copy.deepcopy(
            {key: value for key, value in selected.items() if key != "alternatives"}
        )
        project = selected["mixed"]
    else:
        project = selected
    sine_wav(settings.output_dir / project["fullTrack"], 4)
    before = copy.deepcopy(job.result)
    query = f"song={song}&variant={variant}"
    async with client(app) as http:
        uploaded = await http.post(
            f"/api/jobs/{job.job_id}/recordings?{query}",
            data={"name": "录音"},
            files={"file": ("take.wav", sine_wav(tmp_path / "take.wav", 0.5), "audio/wav")},
        )
        assert uploaded.status_code == 201, uploaded.text
        stem = uploaded.json()["id"]
        response = await http.patch(
            f"/api/jobs/{job.job_id}/stems/{stem}?{query}",
            json={"position": 1.5},
            headers={"X-Audio-Revision": "1"},
        )
        assert response.status_code == 200, response.text
        assert project["recordingClips"][stem] == {
            "position": 1.5,
            "duration": 0.5,
            "waveform": uploaded.json()["waveform"],
            "source": project["recordingClips"][stem]["source"],
        }
    if song == 1:
        assert {key: value for key, value in job.result.items() if key != "alternatives"} == {
            key: value for key, value in before.items() if key != "alternatives"
        }
    else:
        assert {key: value for key, value in job.result.items() if key != "mixed"} == {
            key: value for key, value in before.items() if key != "mixed"
        }


async def test_mp3_master_uses_actual_duration_instead_of_an_estimated_hint(tmp_path):
    app, settings, job = recording_app(tmp_path)
    master = settings.output_dir / job.result["fullTrack"]
    sine_wav(master, 1)
    mp3 = master.with_suffix(".mp3")
    subprocess.run(["ffmpeg", "-v", "error", "-i", str(master), str(mp3)], check=True)
    job.result["fullTrack"] = mp3.relative_to(settings.output_dir).as_posix()
    job.result["durationSeconds"] = 999
    async with client(app) as http:
        uploaded = await http.post(
            f"/api/jobs/{job.job_id}/recordings",
            data={"name": "录音"},
            files={"file": ("take.wav", sine_wav(tmp_path / "take.wav", 0.5), "audio/wav")},
        )
        stem = uploaded.json()["id"]
        target = f"/api/jobs/{job.job_id}/stems/{stem}"
        overflow = await http.patch(target, json={"position": 1}, headers={"X-Audio-Revision": "1"})
        assert overflow.status_code == 400, overflow.text
        valid = await http.patch(target, json={"position": 0.25}, headers={"X-Audio-Revision": "1"})
        assert valid.status_code == 200, valid.text


def test_legacy_recording_keeps_existing_clip_waveform_across_later_moves(tmp_path):
    source = tmp_path / "take.wav"
    padded = tmp_path / "padded.wav"
    moved = tmp_path / "moved.wav"
    sine_wav(source, 0.5)
    legacy = move_recording_audio(source, padded, 0.5, None, 4, 360)
    bins = [0, 0, 0, 0, 0.1, 0.4, 1, 0.2]
    migrated = move_recording_audio(padded, moved, 1.5, legacy, 4, 360, bins)
    assert migrated["waveform"] == bins[4:]
    restored = move_recording_audio(moved, padded, 0, migrated, 4, 360, [0, 0, 1])
    assert restored["waveform"] == bins[4:]
