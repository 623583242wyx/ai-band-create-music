from __future__ import annotations

import asyncio

import pytest

from app.main import _referenced_output_paths
from tests.helpers import client, make_settings
from tests.integration.test_reference_api import sine_wav
from tests.integration.test_split_rebuild import rebuild_app


async def upload(http, job, name, audio, revision):
    return await http.post(
        f"/api/jobs/{job.job_id}/recordings?song=1",
        data={"name": name},
        files={"file": ("a.wav", audio, "audio/wav")},
        headers={"X-Audio-Revision": str(revision)},
    )


async def force(http, job, revision):
    r = await http.post(
        f"/api/jobs/{job.job_id}/split?song=1&force=true",
        headers={"X-Audio-Revision": str(revision)},
    )
    assert r.status_code == 202, r.text
    await job.split_task


async def test_real_upload_survives_two_rebuilds_and_old_stems_go(tmp_path, monkeypatch):
    app, settings, job, _ = rebuild_app(tmp_path, monkeypatch)
    target = job.result["alternatives"][0]
    sine_wav(settings.output_dir / target["fullTrack"], 1)
    audio = sine_wav(tmp_path / "rec.wav", 1)
    async with client(app) as http:
        up = await upload(http, job, "和声", audio, 0)
        assert up.status_code == 201, up.text
        rid = up.json()["id"]
        rec_url = target["stems"][rid]
        rec_preview = target.get("playback", {}).get("stems", {}).get(rid)
        await force(http, job, 1)
        first = dict(target["stems"])
        await force(http, job, 2)
        assert target["stems"][rid] == rec_url
        assert target["stemNames"] == {rid: "和声"}
        assert target.get("playback", {}).get("stems", {}).get(rid) == rec_preview
        assert (settings.output_dir / rec_url).exists()
        assert rec_preview is None or (settings.output_dir / rec_preview).exists()
        for name, url in first.items():
            if name != rid:
                assert not (settings.output_dir / url).exists(), name
        detail = (await http.get(f"/api/jobs/{job.job_id}")).json()["result"]["alternatives"][0]
        assert detail["stemNames"][rid] == "和声" and rid in detail["stems"]


async def test_file_shared_with_mixed_variant_is_not_deleted(tmp_path, monkeypatch):
    app, settings, job, _ = rebuild_app(tmp_path, monkeypatch)
    target = job.result["alternatives"][0]
    shared = target["stems"]["drums"]
    target["mixed"] = {
        "fullTrack": target["fullTrack"],
        "stems": {"drums": shared},
        "stemUrls": [shared],
        "waveforms": {},
        "splitEnabled": True,
        "debug": {},
    }
    async with client(app) as http:
        await force(http, job, 0)
    assert job.split_status == "succeeded"
    assert (settings.output_dir / shared).exists()
    assert not (settings.output_dir / target["stems"]["drums"]).samefile(
        settings.output_dir / shared
    )


async def test_deleted_recording_then_rebuild(tmp_path, monkeypatch):
    app, settings, job, _ = rebuild_app(tmp_path, monkeypatch)
    target = job.result["alternatives"][0]
    sine_wav(settings.output_dir / target["fullTrack"], 1)
    async with client(app) as http:
        up = await upload(http, job, "x", sine_wav(tmp_path / "r.wav", 1), 0)
        rid = up.json()["id"]
        d = await http.delete(
            f"/api/jobs/{job.job_id}/stems/{rid}?song=1", headers={"X-Audio-Revision": "1"}
        )
        assert d.status_code == 204
        await force(http, job, 1)
        assert job.split_status == "succeeded"
        assert rid not in target["stems"]
        r = await http.put(
            f"/api/jobs/{job.job_id}/stems/{rid}?song=1", headers={"X-Audio-Revision": "2"}
        )
        assert r.status_code in {404, 409}, r.text


async def test_upload_rejected_while_split_running(tmp_path, monkeypatch):
    app, settings, job, separator = rebuild_app(tmp_path, monkeypatch)
    target = job.result["alternatives"][0]
    gate = asyncio.Event()
    orig = separator.side_effect

    async def slow(*a):
        await gate.wait()
        return await orig(*a)

    separator.side_effect = slow
    async with client(app) as http:
        r = await http.post(
            f"/api/jobs/{job.job_id}/split?song=1&force=true", headers={"X-Audio-Revision": "0"}
        )
        assert r.status_code == 202
        up = await upload(http, job, "x", sine_wav(tmp_path / "r.wav", 1), 0)
        assert up.status_code == 409
        gate.set()
        await job.split_task
    assert not any(k.startswith("recording_") for k in target["stems"])


@pytest.mark.parametrize(
    "name,ok",
    [
        ("👨‍👩‍👧 和声", True),
        ("می‌خواهم", True),
        ("❤️ take", True),
        ("名‮字", False),
        ("​", False),
        ("a﻿b", False),
        ("a⁦b", False),
        ("a\tb", False),
    ],
)
async def test_name_edge_cases(tmp_path, monkeypatch, name, ok):
    app, settings, job, _ = rebuild_app(tmp_path, monkeypatch)
    sine_wav(settings.output_dir / job.result["alternatives"][0]["fullTrack"], 1)
    async with client(app) as http:
        up = await upload(http, job, name, sine_wav(tmp_path / "r.wav", 1), 0)
    assert (up.status_code == 201) is ok, up.text


def test_reference_walker_ignores_foreign_and_escaping_paths(tmp_path):
    settings = make_settings(tmp_path)
    found = _referenced_output_paths(
        {
            "a": "https://cdn.example.com/x.wav",
            "b": "../../etc/passwd.wav",
            "c": [0.1, None, {"d": "jobs/j/song_1/v.wav"}],
            "e": "http://host/output/jobs/j/song_1/w.mp3",
            "f": "notes.txt",
        },
        settings,
    )
    root = settings.output_dir.resolve()
    assert found == {root / "jobs/j/song_1/v.wav", root / "jobs/j/song_1/w.mp3"}


@pytest.mark.parametrize("cancel_preview", [False, True])
async def test_recording_preview_survives_rebuild(tmp_path, monkeypatch, cancel_preview):
    from unittest.mock import AsyncMock

    from app.services import audio_files

    app, settings, job, _ = rebuild_app(tmp_path, monkeypatch)
    target = job.result["alternatives"][0]
    sine_wav(settings.output_dir / target["fullTrack"], 1)
    monkeypatch.setattr("app.main.make_playback_mp3", audio_files.make_playback_mp3)
    async with client(app) as http:
        up = await upload(http, job, "和声", sine_wav(tmp_path / "r.wav", 1), 0)
        rid = up.json()["id"]
        preview = target["playback"]["stems"][rid]
        if cancel_preview:
            entered = asyncio.Event()

            async def hang(*a, **k):
                entered.set()
                await asyncio.Event().wait()

            monkeypatch.setattr("app.main.make_playback_mp3", hang)
            r = await http.post(
                f"/api/jobs/{job.job_id}/split?song=1&force=true",
                headers={"X-Audio-Revision": "1"},
            )
            assert r.status_code == 202
            await asyncio.wait_for(entered.wait(), 2)
            p = await http.patch(f"/api/jobs/{job.job_id}", json={"status": "cancelled"})
            assert p.status_code == 200
            await job.split_task
        else:
            monkeypatch.setattr(
                "app.main.make_playback_mp3", AsyncMock(side_effect=lambda s, o: f"{s.stem}.mp3")
            )
            await force(http, job, 1)
    assert job.split_status == "succeeded"
    stems_preview = target["playback"]["stems"]
    assert stems_preview[rid] == preview
    assert (settings.output_dir / preview).exists()
    if not cancel_preview:
        assert set(stems_preview) == set(target["stems"])
