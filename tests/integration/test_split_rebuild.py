from __future__ import annotations

import asyncio
import copy
from pathlib import Path
from unittest.mock import AsyncMock
from uuid import uuid4

import pytest

from app.core.errors import GenerationError
from app.main import create_app
from app.services.stems import BS_ROFORMER_MODEL, STEM_NAMES, SplitResult, stem_output_files
from tests.helpers import client, make_orchestrator, make_settings
from tests.integration.test_mix_api import seed_job
from tests.integration.test_reference_api import sine_wav


def rebuild_app(tmp_path, monkeypatch):
    settings = make_settings(tmp_path)
    orchestrator = make_orchestrator(settings)
    app = create_app(settings, orchestrator)
    job = seed_job(app, settings, count=2)

    async def split(input_path, output_dir):
        assert input_path.name in {"full_song.wav", "mixed_master.wav"}
        files = stem_output_files(Path(f"bs_rebuilt_{uuid4().hex[:12]}.wav"))
        for filename in files.values():
            sine_wav(output_dir / filename, 0.1)
        return SplitResult("six stems generated", "", 12, files)

    separator = AsyncMock(side_effect=split)
    monkeypatch.setattr(orchestrator.stem_separator, "split", separator)
    monkeypatch.setattr(
        "app.main.extract_waveforms",
        AsyncMock(return_value={"full": [0.25], **{name: [0.5] for name in STEM_NAMES}}),
    )
    monkeypatch.setattr("app.main.make_playback_mp3", AsyncMock(return_value=None))
    return app, settings, job, separator


async def test_force_rebuild_replaces_old_draft_and_keeps_other_candidate(tmp_path, monkeypatch):
    app, settings, job, separator = rebuild_app(tmp_path, monkeypatch)
    original = {
        key: copy.deepcopy(value) for key, value in job.result.items() if key != "alternatives"
    }
    target = job.result["alternatives"][0]
    old_stems = dict(target["stems"])
    target["stemNames"] = {"vocal": "旧车道名称"}
    old_master = target["fullTrack"]
    async with client(app) as http:
        cached = await http.post(f"/api/jobs/{job.job_id}/split?song=1")
        assert cached.status_code == 200
        separator.assert_not_awaited()
        accepted = await http.post(
            f"/api/jobs/{job.job_id}/split?song=1&force=true",
            headers={"X-Audio-Revision": "0"},
        )
        assert accepted.status_code == 202
        await job.split_task
        detail = (await http.get(f"/api/jobs/{job.job_id}")).json()
        rebuilt = detail["result"]["alternatives"][0]
        assert detail["splitStatus"] == "succeeded"
        assert rebuilt["audioRevision"] == 1
        assert set(rebuilt["stems"]) == set(STEM_NAMES)
        assert "stemNames" not in rebuilt and "replacedVocal" not in rebuilt
        assert rebuilt["debug"]["splitterEngine"] == "bs_roformer"
        assert rebuilt["debug"]["splitterModel"] == BS_ROFORMER_MODEL
        assert target["fullTrack"] == old_master
        assert not set(target["stems"].values()) & set(old_stems.values())
        assert all((settings.output_dir / value).exists() for value in old_stems.values())
        assert {
            key: value for key, value in job.result.items() if key != "alternatives"
        } == original
        stale = await http.post(
            f"/api/jobs/{job.job_id}/split?song=1&force=true",
            headers={"X-Audio-Revision": "0"},
        )
        assert stale.status_code == 409
        assert separator.await_count == 1
    relaunched = create_app(settings, make_orchestrator(settings))
    async with client(relaunched) as http:
        stored = (await http.get(f"/api/jobs/{job.job_id}")).json()["result"]["alternatives"][0]
    assert stored["debug"]["splitterEngine"] == "bs_roformer" and stored["audioRevision"] == 1


async def test_mixed_variant_splits_only_the_selected_project(tmp_path, monkeypatch):
    app, settings, job, separator = rebuild_app(tmp_path, monkeypatch)
    master = settings.output_dir / "jobs" / job.job_id / "song_1" / "mixed_master.wav"
    sine_wav(master, 0.1)
    job.result["mixed"] = {
        "fullTrack": master.relative_to(settings.output_dir).as_posix(),
        "stems": {},
        "stemUrls": [],
        "waveforms": {},
        "splitEnabled": False,
        "debug": {},
    }
    original = {key: copy.deepcopy(value) for key, value in job.result.items() if key != "mixed"}
    async with client(app) as http:
        response = await http.post(f"/api/jobs/{job.job_id}/split?variant=mixed")
        assert response.status_code == 202
        await job.split_task
        detail = (await http.get(f"/api/jobs/{job.job_id}")).json()
    assert detail["splitStatus"] == "succeeded" and job.split_variant == "mixed"
    assert {key: value for key, value in job.result.items() if key != "mixed"} == original
    assert set(detail["result"]["mixed"]["stems"]) == set(STEM_NAMES)
    assert detail["result"]["mixed"]["audioRevision"] == 1
    assert separator.await_args.args[0] == master


@pytest.mark.parametrize("failure", ["model", "save", "cancel"])
async def test_failed_rebuild_preserves_project_and_cleans_new_outputs(
    tmp_path, monkeypatch, failure
):
    app, settings, job, separator = rebuild_app(tmp_path, monkeypatch)
    previous = copy.deepcopy(job.result)
    old_bytes = {
        value: (settings.output_dir / value).read_bytes() for value in job.result["stems"].values()
    }
    entered = asyncio.Event()
    if failure == "model":
        separator.side_effect = GenerationError("model failed")
    elif failure == "save":

        def fail_save(self, directory):
            raise OSError("disk full")

        monkeypatch.setattr(type(job), "save", fail_save)
    else:

        async def blocked_waveforms(*args, **kwargs):
            entered.set()
            await asyncio.Event().wait()

        monkeypatch.setattr("app.main.extract_waveforms", blocked_waveforms)
    async with client(app) as http:
        response = await http.post(
            f"/api/jobs/{job.job_id}/split?force=true", headers={"X-Audio-Revision": "0"}
        )
        assert response.status_code == 202
        if failure == "cancel":
            await asyncio.wait_for(entered.wait(), 2)
            assert (
                await http.patch(f"/api/jobs/{job.job_id}", json={"status": "cancelled"})
            ).status_code == 200
        await job.split_task
    assert job.split_status == ("cancelled" if failure == "cancel" else "failed")
    assert job.result == previous
    assert all(
        (settings.output_dir / name).read_bytes() == data for name, data in old_bytes.items()
    )
    assert not list(settings.output_dir.rglob("bs_rebuilt_*.wav"))
    assert app.state.orchestrator.capacity.active == 0


@pytest.mark.parametrize("revision,status", [("bad", 400), ("-1", 400), ("0", 409)])
async def test_force_rebuild_validates_revision_before_model_work(
    tmp_path, monkeypatch, revision, status
):
    app, _, job, separator = rebuild_app(tmp_path, monkeypatch)
    job.result["audioRevision"] = 1
    async with client(app) as http:
        response = await http.post(
            f"/api/jobs/{job.job_id}/split?force=true",
            headers={"X-Audio-Revision": revision},
        )
    assert response.status_code == status
    separator.assert_not_awaited()
