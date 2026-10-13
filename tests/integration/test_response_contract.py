"""固化可用歌曲的 result / alternatives / count 与独立歌曲状态的响应契约。

生成中的任务也可携带已完成歌曲；count 始终是可用歌曲数，requestedCount 是请求数。
前端按 songNumber 与 songStates 区分尚未完成、失败和已经可以播放的歌曲。
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from starlette.requests import Request

from app.main import GenerationJob, _job_response
from tests.helpers import make_settings

REPO_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = REPO_ROOT / "output"


def fake_request(app) -> Request:
    return Request(
        {
            "type": "http",
            "app": app,
            "headers": [],
            "scheme": "http",
            "server": ("testserver", 80),
            "path": "/api/jobs/active",
            "root_path": "",
            "query_string": b"",
            "method": "GET",
        }
    )


def _real_records() -> list[tuple[str, dict]]:
    records = []
    for path in sorted((OUTPUT_DIR / "jobs").glob("*/job.json")):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if isinstance(data.get("result"), dict):
            records.append((path.parent.name, data))
    return records


REAL_RECORDS = _real_records()
REAL_TWO_SONG = [item for item in REAL_RECORDS if item[1]["result"].get("count") == 2]


def _song(full_track: str) -> dict:
    """按 orchestrator 实际写出的字段结构造一首歌的输出。"""
    return {
        "fullTrack": full_track,
        "stems": {},
        "stemUrls": [],
        "waveforms": {"full": [0.25, 0.5]},
        "splitEnabled": False,
        "durationSeconds": 30.0,
        "debug": {"music": {}},
    }


def _job(job_id: str, *, status: str, count: int | None, stage: str = "completed") -> GenerationJob:
    result = None
    if count is not None:
        songs = [_song(f"jobs/{job_id}/song_{index + 1}/full_song.wav") for index in range(count)]
        result = {
            "success": True,
            "jobId": job_id,
            "prompt": "rock",
            "durationMinutes": "auto",
            "structuredPrompt": "[Genre: Rock]",
            "lyrics": "[Verse]",
            "count": count,
            "requestedCount": count,
            "provider": "minimax_music",
            "createdAt": "2026-01-01T00:00:00+00:00",
            "alternatives": songs[1:],
            **songs[0],
        }
    return GenerationJob(
        job_id=job_id,
        prompt="rock",
        status=status,
        stage=stage,
        progress=100 if status == "succeeded" else None,
        result=result,
    )


def test_real_two_song_jobs_keep_result_and_alternatives_consistent():
    """真实落盘的 count=2 任务：result 一定在，alternatives 长度恒为 count-1。"""
    if not REAL_TWO_SONG:
        pytest.skip("output/ 里没有真实的 count=2 任务")
    for job_id, data in REAL_TWO_SONG:
        result = data["result"]
        assert isinstance(result.get("alternatives"), list), job_id
        assert result["count"] == 2, job_id
        assert result["count"] == 1 + len(result["alternatives"]), job_id
        assert "count" not in result["alternatives"][0], job_id


def test_every_persisted_result_declares_alternatives_and_matching_count():
    """全部落盘任务：有 result 就一定有 alternatives 列表，且 count 与长度自洽。"""
    if not REAL_RECORDS:
        pytest.skip("output/ 里没有带 result 的任务")
    for job_id, data in REAL_RECORDS:
        result = data["result"]
        assert isinstance(result.get("alternatives"), list), job_id
        assert result["count"] == 1 + len(result["alternatives"]), job_id


def test_running_job_can_publish_first_song_while_second_is_running(tmp_path):
    settings = make_settings(tmp_path)
    job = _job("job_running", status="running", count=1, stage="generating_music")
    job.result.update(requestedCount=2, songNumber=1)
    job.current_song = 2
    job.song_states = [
        {"songNumber": 1, "status": "succeeded", "stage": "song_completed", "progress": 100},
        {"songNumber": 2, "status": "running", "stage": "generating_music", "progress": None},
    ]
    payload = _job_response(job, fake_request(object()), settings)
    assert payload["status"] == "running"
    assert payload["currentSong"] == 2
    assert payload["result"]["requestedCount"] == 2
    assert payload["result"]["count"] == 1
    assert payload["result"]["alternatives"] == []
    assert payload["result"]["waveforms"] == {"full": [0.25, 0.5]}
    assert payload["songStates"] == job.song_states


def test_terminal_frame_always_carries_every_song(tmp_path):
    """终态帧：result.count == 1 + len(alternatives)，两首都带得满。"""
    settings = make_settings(tmp_path)
    for count in (1, 2):
        job = _job(f"job_done_{count}", status="succeeded", count=count)
        payload = _job_response(job, fake_request(object()), settings, include_waveforms=False)
        result = payload["result"]
        assert payload["status"] == "succeeded"
        assert result["count"] == count
        assert result["count"] == 1 + len(result["alternatives"])
        assert isinstance(result["fullTrack"], str)
        for alternative in result["alternatives"]:
            assert isinstance(alternative["fullTrack"], str)


def test_successful_job_can_never_be_missing_alternatives(tmp_path):
    """中途发布和最终返回的 result 都包含 alternatives 键。"""
    settings = make_settings(tmp_path)
    job = _job("job_two", status="succeeded", count=2)
    assert "alternatives" in job.result

    payload = _job_response(job, fake_request(object()), settings, include_waveforms=False)
    assert "alternatives" in payload["result"]
    assert len(payload["result"]["alternatives"]) == 1

    # 记录退化形状：即使 alternatives 被清空，count 仍然说明有两首候选。
    # 也就是说 count 比 alternatives.length 更稳，供契约核对时对照。
    job.result["alternatives"] = []
    degraded = _job_response(job, fake_request(object()), settings, include_waveforms=False)
    assert degraded["result"]["count"] == 2
    assert len(degraded["result"]["alternatives"]) == 0


def test_all_song_variants_put_guitar_and_piano_above_other(tmp_path):
    settings = make_settings(tmp_path)
    job = _job("ordered", status="succeeded", count=2)
    original_order = ["vocal", "drums", "bass", "other", "guitar", "piano", "recording_1"]
    job.result["mixed"] = _song("jobs/ordered/mixed.wav")
    for song in (job.result, job.result["alternatives"][0], job.result["mixed"]):
        song["stems"] = {name: f"jobs/ordered/{name}.wav" for name in original_order}
        song["splitEnabled"] = True

    rendered = _job_response(job, fake_request(object()), settings)["result"]
    for song in (rendered, rendered["alternatives"][0], rendered["mixed"]):
        assert list(song["stems"]) == [
            "vocal",
            "drums",
            "bass",
            "guitar",
            "piano",
            "other",
            "recording_1",
        ]
        assert song["stemUrls"] == list(song["stems"].values())
    assert list(job.result["stems"]) == original_order
