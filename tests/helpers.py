from __future__ import annotations

import asyncio
import wave
from pathlib import Path

import httpx
from starlette.requests import Request

from app.core.config import Settings
from app.infrastructure.events import NullEventPublisher
from app.infrastructure.queue import InlineTaskDispatcher
from app.main import GenerationJob
from app.services.orchestrator import GenerationOrchestrator
from app.services.prompt import PreparedPrompt
from app.services.providers import MusicResult
from app.services.stems import SplitResult, stem_output_files


class StubPromptExpander:
    async def prepare(
        self,
        user_prompt: str,
        duration_minutes: float | None = None,
        *,
        job_id: str | None = None,
        title: str | None = None,
    ) -> PreparedPrompt:
        duration_seconds = duration_minutes * 60 if duration_minutes is not None else None
        structured_prompt = f"[Genre: Test], [Source: {user_prompt}]"
        if user_prompt.startswith("[歌词与创作内容]"):
            lyrics = user_prompt.split("\n\n[风格要求]", 1)[0].split("\n", 1)[1]
            return PreparedPrompt(
                structured_prompt, f"[Verse]\n{lyrics}", duration_seconds, title or "测试歌名"
            )
        return PreparedPrompt(
            structured_prompt, "[Verse]\n自动生成的测试歌词", duration_seconds, title or "测试歌名"
        )


class BlockingPromptExpander:
    def __init__(self) -> None:
        self.started = asyncio.Event()
        self.release = asyncio.Event()

    async def prepare(
        self,
        user_prompt: str,
        duration_minutes: float | None = None,
        *,
        job_id: str | None = None,
        title: str | None = None,
    ) -> PreparedPrompt:
        self.started.set()
        await self.release.wait()
        return PreparedPrompt(
            "[Genre: Test]",
            "[Verse]\n自动生成的测试歌词",
            duration_minutes * 60 if duration_minutes is not None else None,
            title or "测试歌名",
        )


class StubMusicProvider:
    def __init__(self, source: Path, name: str = "stub") -> None:
        self.source = source
        self.name = name
        self.user_prompt = ""
        self.variations: list[int] = []
        self.requested_durations: list[int | None] = []

    async def generate(
        self,
        structured_prompt: str,
        duration_seconds: int | None,
        user_prompt: str,
        *,
        variation: int = 0,
        progress=None,
        job_id=None,
    ) -> MusicResult:
        self.user_prompt = user_prompt
        self.variations.append(variation)
        self.requested_durations.append(duration_seconds)
        return MusicResult(
            self.source, {"provider": self.name, "durationSeconds": duration_seconds or 150}
        )


class StubStemSeparator:
    async def split(self, input_path: Path, output_dir: Path) -> SplitResult:
        output_dir.mkdir(parents=True, exist_ok=True)
        output_files = stem_output_files(input_path)
        for file_name in output_files.values():
            with wave.open(str(output_dir / file_name), "wb") as audio:
                audio.setparams((1, 2, 8_000, 0, "NONE", "not compressed"))
                audio.writeframes(b"\x00\x00" * 800)
        return SplitResult("stub split", "", 5, output_files)


def make_settings(tmp_path: Path, **updates: object) -> Settings:
    rvc_model = tmp_path / "rvc-model.pth"
    rvc_model.write_bytes(b"test model")
    values: dict[str, object] = {
        "output_dir": tmp_path / "output",
        "mock_full_song_path": tmp_path / "source.mp3",
        "public_base_url": "",
        "rvc_model_path": rvc_model,
        "rvc_index_path": None,
        "rvc_base_model_dir": None,
        "bs_roformer_models_dir": tmp_path / "model-cache",
    }
    values.update(updates)
    return Settings(_env_file=None, **values)


def make_orchestrator(
    settings: Settings,
    *,
    prompt_expander=None,
) -> GenerationOrchestrator:
    source = settings.mock_full_song_path
    source.parent.mkdir(parents=True, exist_ok=True)
    source.write_bytes(b"ID3-full-audio")
    return GenerationOrchestrator(
        settings=settings,
        prompt_expander=prompt_expander or StubPromptExpander(),
        music_provider=StubMusicProvider(source),
        stem_separator=StubStemSeparator(),
        task_dispatcher=InlineTaskDispatcher(),
        events=NullEventPublisher(),
    )


def client(app):
    return httpx.AsyncClient(transport=httpx.ASGITransport(app), base_url="http://testserver")


def events_request(app, job_id: str) -> Request:
    return Request(
        {
            "type": "http",
            "app": app,
            "headers": [],
            "scheme": "http",
            "server": ("testserver", 80),
            "path": f"/api/jobs/{job_id}/events",
            "root_path": "",
            "query_string": b"",
            "method": "GET",
        }
    )


def seed_job(app, settings, job_id: str = "job-replace") -> tuple[GenerationJob, Path]:
    song_dir = settings.output_dir / "jobs" / job_id / "song_1"
    song_dir.mkdir(parents=True, exist_ok=True)
    full_track = song_dir / "full_song.wav"
    vocal = song_dir / "demo_vocal.mp3"
    full_track.write_bytes(b"RIFF-full")
    vocal.write_bytes(b"ID3-vocal")
    job = GenerationJob(
        job_id=job_id,
        prompt="rock",
        structured_prompt="[Genre: Rock]",
        lyrics="[Verse]\ntest",
        status="succeeded",
        stage="completed",
        progress=100,
        message="音乐生成完成",
        result={
            "success": True,
            "jobId": job_id,
            "prompt": "rock",
            "durationMinutes": "auto",
            "structuredPrompt": "[Genre: Rock]",
            "lyrics": "[Verse]\ntest",
            "count": 1,
            "alternatives": [],
            "fullTrack": full_track.relative_to(settings.output_dir).as_posix(),
            "stems": {"vocal": vocal.relative_to(settings.output_dir).as_posix()},
            "stemUrls": [vocal.relative_to(settings.output_dir).as_posix()],
            "waveforms": {"vocal": [0.5]},
            "splitEnabled": True,
            "debug": {},
        },
    )
    app.state.jobs[job_id] = job
    job.save(settings.output_dir)
    return job, vocal


class BlockingVoiceEngine:
    loaded = True

    def __init__(self) -> None:
        self.started = asyncio.Event()
        self.release = asyncio.Event()
        self.calls = 0
        self.params = None

    async def convert(self, input_path: Path, output_path: Path, **_params) -> None:
        self.calls += 1
        self.params = _params
        assert input_path.name == "demo_vocal.mp3"
        self.started.set()
        await self.release.wait()
        output_path.write_bytes(b"RIFF-replaced")


class ReadyVoiceEngine:
    """转换在 POST 返回后立刻完成，用来模拟"客户端连上 SSE 时替换已经收尾"。"""

    loaded = True

    async def convert(self, _input_path: Path, output_path: Path, **_params) -> None:
        output_path.write_bytes(b"RIFF-replaced")
