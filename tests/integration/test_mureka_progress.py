from datetime import datetime
from unittest.mock import AsyncMock

import httpx
import pytest

from app.core.errors import GenerationError
from app.main import create_app, load_jobs
from tests.helpers import make_orchestrator, make_settings


@pytest.mark.parametrize("endpoint", ["/api/jobs", "/api/generate"])
@pytest.mark.parametrize("outcome", ["succeeded", "failed"])
async def test_mureka_progress_has_per_song_clock_and_provider_messages(
    tmp_path, monkeypatch, endpoint, outcome
):
    settings = make_settings(tmp_path, music_provider="mureka_music")
    orchestrator = make_orchestrator(settings)
    original_generate = orchestrator.music_provider.generate
    monkeypatch.setattr("app.services.orchestrator.make_playback_mp3", AsyncMock(return_value=None))
    monkeypatch.setattr(
        "app.services.orchestrator.extract_waveforms", AsyncMock(return_value={"full": [0.5]})
    )
    clocks = []

    async def generate(*args, variation=0, progress=None, job_id=None, **kwargs):
        started_at = app.state.jobs[job_id].response()["songStates"][variation][
            "generationStartedAt"
        ]
        for stage in ["waiting_music", "generating_music"]:
            await progress(stage, None, None)
            row = app.state.jobs[job_id].response()
            song = row["songStates"][variation]
            assert song["generationStartedAt"] == started_at
            assert song["progress"] is None
            assert song["receivedAudioSeconds"] is None
            text = (
                "等待 Mureka 返回进度"
                if stage == "waiting_music"
                else "Mureka 正在生成，总时长待确定"
            )
            assert song["message"] == row["message"] == f"第 {variation + 1}/2 首：{text}"
            if variation == 0:
                assert row["songStates"][1].get("generationStartedAt") is None
            else:
                assert row["songStates"][0]["status"] == "succeeded"
        clocks.append(datetime.fromisoformat(song["generationStartedAt"]))
        if variation == 1 and outcome == "failed":
            raise GenerationError("Mureka failed")
        return await original_generate(*args, variation=variation, **kwargs)

    orchestrator.music_provider.generate = generate
    app = create_app(settings, orchestrator)
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app), base_url="http://testserver"
    ) as client:
        response = await client.post(endpoint, json={"prompt": "rock", "count": 2})
        assert response.status_code == (202 if endpoint == "/api/jobs" else 200)
        job_id = response.json()["jobId"]
        if endpoint == "/api/jobs":
            await app.state.jobs[job_id].task
        row = (await client.get(f"/api/jobs/{job_id}")).json()
        assert len(clocks) == 2
        assert clocks[1] > clocks[0]
        assert [song["status"] for song in row["songStates"]] == ["succeeded", outcome]
        restored = load_jobs(settings.output_dir)[job_id].response()
        assert restored["songStates"] == row["songStates"]
