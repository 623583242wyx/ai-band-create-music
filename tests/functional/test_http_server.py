from __future__ import annotations

import shutil
import socket
import threading
import time
from pathlib import Path

import httpx
import uvicorn

from app.main import create_app
from tests.helpers import make_orchestrator, make_settings


def test_complete_generation_over_real_http(tmp_path: Path) -> None:
    settings = make_settings(tmp_path)
    app = create_app(settings, make_orchestrator(settings))

    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind(("127.0.0.1", 0))
    sock.listen(128)
    port = sock.getsockname()[1]
    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="error"))
    thread = threading.Thread(target=server.run, kwargs={"sockets": [sock]}, daemon=True)
    thread.start()
    deadline = time.monotonic() + 5
    while not server.started and time.monotonic() < deadline:
        time.sleep(0.01)

    try:
        assert server.started
        with httpx.Client(
            base_url=f"http://127.0.0.1:{port}", timeout=5, trust_env=False
        ) as client:
            health = client.get("/api/health")
            generated = client.post(
                "/api/generate",
                json={"prompt": "cinematic rock with Mandarin vocal", "durationMinutes": 2},
            )
            body = generated.json()
            split = client.post(f"/api/jobs/{body['jobId']}/split?song=0")
            deadline = time.monotonic() + 5
            while time.monotonic() < deadline:
                completed = client.get(f"/api/jobs/{body['jobId']}").json()
                if completed["splitStatus"] not in {"pending", "running"}:
                    break
                time.sleep(0.01)
            assert completed["status"] == "succeeded"
            assert completed["splitStatus"] == "succeeded"
            downloads = {
                name: client.get(url) for name, url in completed["result"]["stems"].items()
            }
        assert health.status_code == 200
        assert generated.status_code == 200
        assert split.status_code == 202
        assert body["durationMinutes"] == 2
        assert body["structuredPrompt"].startswith("[Genre: Test]")
        assert sorted(downloads) == ["bass", "drums", "guitar", "other", "piano", "vocal"]
        assert all(response.status_code == 200 for response in downloads.values())
        assert all(
            response.content[:4] == b"RIFF" and response.content[8:12] == b"WAVE"
            for response in downloads.values()
        )
    finally:
        server.should_exit = True
        thread.join(timeout=5)
        sock.close()


def _tone(path: Path, frequency: float, *, seconds: float = 4.0) -> str:
    """一段真实可解码的双声道音频；长度要够 loudnorm 测得出响度。"""
    import math
    import struct
    import wave

    frames = bytearray()
    for index in range(int(44_100 * seconds)):
        value = int(9000 * math.sin(2 * math.pi * frequency * index / 44_100))
        frames += struct.pack("<hh", value, value)
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(2)
        handle.setsampwidth(2)
        handle.setframerate(44_100)
        handle.writeframes(bytes(frames))
    return path


def _live_server(app):
    """在真实 uvicorn 上跑一个 app，返回 (base_url, 关闭函数)。"""
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind(("127.0.0.1", 0))
    sock.listen(128)
    port = sock.getsockname()[1]
    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="error"))
    thread = threading.Thread(target=server.run, kwargs={"sockets": [sock]}, daemon=True)
    thread.start()
    deadline = time.monotonic() + 5
    while not server.started and time.monotonic() < deadline:
        time.sleep(0.01)

    def stop() -> None:
        server.should_exit = True
        thread.join(timeout=5)

    return f"http://127.0.0.1:{port}", stop, server.started


def test_mix_over_real_http_with_real_ffmpeg(tmp_path: Path) -> None:
    """任务级合轨走真实 Uvicorn + 真实 FFmpeg，并核对成品格式与下载。"""
    import json
    import subprocess

    import pytest

    if not shutil.which("ffmpeg") or not shutil.which("ffprobe"):
        pytest.skip("real FFmpeg tools are unavailable")

    from app.main import GenerationJob

    settings = make_settings(tmp_path)
    app = create_app(settings, make_orchestrator(settings))
    job_id = "job-live-mix"
    directory = settings.output_dir / "jobs" / job_id / "song_1"
    relative = {}
    for key, (name, frequency) in {
        "fullTrack": ("full_song.wav", 440.0),
        "replacedVocal": ("demo_rvc_vocal.wav", 220.0),
        "vocal": ("demo_vocal.wav", 220.0),
        "drums": ("demo_drums.wav", 330.0),
        "bass": ("demo_bass.wav", 110.0),
        "other": ("demo_other.wav", 550.0),
    }.items():
        relative[key] = (
            _tone(directory / name, frequency).relative_to(settings.output_dir).as_posix()
        )
    stems = {name: relative[name] for name in ("vocal", "drums", "bass", "other")}
    app.state.jobs[job_id] = GenerationJob(
        job_id=job_id,
        prompt="rock",
        status="succeeded",
        stage="completed",
        progress=100,
        result={
            "success": True,
            "jobId": job_id,
            "prompt": "rock",
            "durationMinutes": "auto",
            "structuredPrompt": "[Genre: Rock]",
            "lyrics": "[Verse]",
            "count": 1,
            "alternatives": [],
            "fullTrack": relative["fullTrack"],
            "replacedVocal": relative["replacedVocal"],
            "_replacedVocalModel": app.state.rvc_model_fingerprint,
            "stems": stems,
            "stemUrls": list(stems.values()),
            "waveforms": {"full": [0.2] * 640},
            "splitEnabled": True,
            "debug": {},
        },
    )
    base_url, stop, started = _live_server(app)
    try:
        assert started
        with httpx.Client(base_url=base_url, timeout=60, trust_env=False) as client:
            accepted = client.post(f"/api/jobs/{job_id}/mix?song=0")
            deadline = time.monotonic() + 60
            while time.monotonic() < deadline:
                detail = client.get(f"/api/jobs/{job_id}").json()
                if detail.get("mixStatus") not in {"pending", "running"}:
                    break
                time.sleep(0.05)
            audio = client.get(detail["result"]["mixedTrack"])
            while time.monotonic() < deadline and not detail["result"].get("playback", {}).get(
                "mixedTrack"
            ):
                time.sleep(0.05)
                detail = client.get(f"/api/jobs/{job_id}").json()
            preview = client.get(
                detail["result"]["playback"]["mixedTrack"], headers={"Range": "bytes=0-2"}
            )
            history = client.get("/api/jobs").json()["jobs"][0]

        assert accepted.status_code == 202, accepted.text
        assert detail["mixStatus"] == "succeeded"
        assert detail["result"]["mixedTrack"].endswith("/demo_rvc_mix.wav")
        assert history["result"]["mixedTrack"] == detail["result"]["mixedTrack"]
        assert len(detail["result"]["waveforms"]["mix"]) == 640
        assert audio.status_code == 200 and len(audio.content) > 1000
        assert preview.status_code == 206 and preview.headers["content-type"].startswith(
            "audio/mpeg"
        )
        assert preview.headers["content-length"] == "3"
        # 成品与原始音频同格式（真实 ffprobe 读盘）。
        probe = json.loads(
            subprocess.run(
                [
                    "ffprobe",
                    "-v",
                    "error",
                    "-select_streams",
                    "a:0",
                    "-show_entries",
                    "stream=sample_rate,channels",
                    "-of",
                    "json",
                    str(settings.output_dir / app.state.jobs[job_id].result["mixed"]["fullTrack"]),
                ],
                capture_output=True,
                check=True,
                text=True,
            ).stdout
        )["streams"][0]
        assert (int(probe["sample_rate"]), int(probe["channels"])) == (44_100, 2)
    finally:
        stop()
