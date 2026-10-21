from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Any, Protocol
from uuid import uuid4

from app.core.config import Settings
from app.core.errors import CapacityExceededError, GenerationError, ProviderGlobalError
from app.infrastructure.events import EventPublisher, GenerationEvent
from app.infrastructure.queue import TaskDispatcher
from app.schemas import SongGenerationState
from app.services.audio_files import ensure_file_under_root, make_playback_mp3
from app.services.job_files import job_song_dir, update_job_diagnostics
from app.services.prompt import PromptExpander, split_generation_prompt
from app.services.providers import AudioReference, MusicProvider
from app.services.stems import StemSeparator
from app.services.waveforms import extract_waveforms

logger = logging.getLogger(__name__)


class ProgressCallback(Protocol):
    async def __call__(
        self,
        stage: str,
        progress: int | None,
        message: str,
        *,
        structured_prompt: str | None = None,
        lyrics: str | None = None,
        step: int | None = None,
        total_steps: int | None = None,
        title: str | None = None,
        received_audio_seconds: float | None = None,
        expected_audio_seconds: float | None = None,
        current_song: int | None = None,
        song_states: list[dict[str, Any]] | None = None,
        result: dict[str, Any] | None = None,
    ) -> None: ...


class GenerationCapacity:
    def __init__(self, maximum: int) -> None:
        self.maximum = maximum
        self._active = 0
        self._lock = asyncio.Lock()

    @property
    def active(self) -> int:
        return self._active

    async def acquire(self) -> None:
        async with self._lock:
            if self._active >= self.maximum:
                raise CapacityExceededError("当前生成任务过多，请稍后重试。")
            self._active += 1

    async def release(self) -> None:
        async with self._lock:
            self._active = max(0, self._active - 1)

    @asynccontextmanager
    async def slot(self) -> AsyncIterator[None]:
        await self.acquire()
        try:
            yield
        finally:
            await self.release()


class GenerationOrchestrator:
    def __init__(
        self,
        settings: Settings,
        prompt_expander: PromptExpander,
        music_provider: MusicProvider,
        stem_separator: StemSeparator,
        task_dispatcher: TaskDispatcher,
        events: EventPublisher,
        music_providers: dict[str, MusicProvider] | None = None,
    ) -> None:
        self.settings = settings
        self.prompt_expander = prompt_expander
        self.music_provider = music_provider
        self.music_providers = music_providers or {}
        self.stem_separator = stem_separator
        self.task_dispatcher = task_dispatcher
        self.events = events
        self.capacity = GenerationCapacity(settings.max_concurrent_generations)

    async def generate(
        self,
        user_prompt: str,
        duration_minutes: float | None,
        request_id: str,
        *,
        job_id: str | None = None,
        progress: ProgressCallback | None = None,
        capacity_reserved: bool = False,
        provider: str | None = None,
        count: int = 1,
        title: str | None = None,
        reference: AudioReference | None = None,
        direct_prompt: bool = False,
    ) -> dict[str, Any]:
        job_id = job_id or f"job_{int(time.time() * 1000)}_{uuid4().hex[:8]}"
        selected_provider = provider or self.settings.music_provider
        # MiniMax and Mureka choose the song length themselves; requested durations are ignored.
        provider_sets_duration = selected_provider in {"minimax_music", "mureka_music"}
        effective_duration_minutes = None if provider_sets_duration else duration_minutes
        effective_count = (
            count
            if selected_provider in {"minimax_music", "elevenlabs_music", "mureka_music"}
            else 1
        )
        current_song = None
        latest_result = None
        song_states = [
            SongGenerationState(
                songNumber=number, status="pending", stage="pending", message="等待生成"
            ).model_dump()
            for number in range(1, effective_count + 1)
        ]

        async def report(
            stage: str,
            value: int | None,
            message: str,
            *,
            structured_prompt: str | None = None,
            lyrics: str | None = None,
            step: int | None = None,
            total_steps: int | None = None,
            title: str | None = None,
            received_audio_seconds: float | None = None,
            expected_audio_seconds: float | None = None,
            song_progress: int | None = None,
        ) -> None:
            if current_song is not None:
                song_state = song_states[current_song - 1]
                song_state.update(
                    stage=stage,
                    message=message,
                    step=step,
                    totalSteps=total_steps,
                    receivedAudioSeconds=received_audio_seconds,
                    expectedAudioSeconds=expected_audio_seconds,
                )
                if song_progress is not None:
                    song_state["progress"] = song_progress
            if progress is not None:
                await progress(
                    stage,
                    value,
                    message,
                    structured_prompt=structured_prompt,
                    lyrics=lyrics,
                    step=step,
                    total_steps=total_steps,
                    title=title,
                    received_audio_seconds=received_audio_seconds,
                    expected_audio_seconds=expected_audio_seconds,
                    current_song=current_song,
                    song_states=[state.copy() for state in song_states],
                    result=latest_result,
                )

        async def execute() -> dict[str, Any]:
            nonlocal current_song, latest_result
            update_job_diagnostics(
                self.settings.output_dir,
                job_id,
                requestId=request_id,
                prompt=user_prompt,
                requestedDurationMinutes=(
                    duration_minutes if duration_minutes is not None else "auto"
                ),
                provider=selected_provider,
                requestedCount=count,
                effectiveCount=effective_count,
                audioReferenceId=reference.song_id if reference else None,
                structuredPrompt=None,
                lyrics=None,
            )
            await self.events.publish(GenerationEvent("generation.started", job_id, request_id))
            if direct_prompt:
                # Create 收起页只提交用户确认的风格，不再自动扩写或生成歌词。
                style = structured_prompt = user_prompt
                lyrics = ""
                song_title = title
                duration_seconds = (
                    round(effective_duration_minutes * 60)
                    if effective_duration_minutes is not None
                    else None
                )
            else:
                await report("expanding_prompt", None, "正在处理歌词与音乐风格")
                _, style = split_generation_prompt(user_prompt)
                prepared = await self.prompt_expander.prepare(
                    user_prompt,
                    effective_duration_minutes,
                    job_id=job_id,
                    title=title,
                )
                song_title = title or prepared.title
                structured_prompt = prepared.structured_prompt
                lyrics = prepared.lyrics
                duration_seconds = prepared.duration_seconds
            if duration_seconds is None and selected_provider not in {
                "minimax_music",
                "elevenlabs_music",
                "mureka_music",
            }:
                duration_seconds = self.settings.default_duration_minutes * 60
            provider_prompt = user_prompt if direct_prompt else f"[歌词与创作内容]\n{lyrics}"
            if style and not direct_prompt:
                provider_prompt += f"\n\n[风格要求]\n{style}"
            duration_diagnostics: dict[str, object] = {
                "durationSource": (
                    "provider_override"
                    if provider_sets_duration and duration_minutes is not None
                    else "provider"
                    if selected_provider in {"minimax_music", "elevenlabs_music", "mureka_music"}
                    and duration_seconds is None
                    else "user"
                    if duration_minutes is not None
                    else "default"
                )
            }
            if duration_seconds is not None:
                duration_diagnostics.update(
                    effectiveDurationSeconds=duration_seconds,
                    effectiveDurationMinutes=duration_seconds / 60,
                )
            update_job_diagnostics(
                self.settings.output_dir,
                job_id,
                style=style,
                structuredPrompt=structured_prompt,
                lyrics=lyrics,
                providerPrompt=provider_prompt,
                **duration_diagnostics,
            )
            outputs: list[dict[str, Any]] = []
            task_progress = (
                0
                if selected_provider == "elevenlabs_music" and duration_seconds is not None
                else None
            )
            music_provider = self.music_providers.get(selected_provider, self.music_provider)
            response: dict[str, Any] = {
                "success": True,
                "jobId": job_id,
                "prompt": user_prompt,
                "title": song_title,
                "durationMinutes": duration_minutes if duration_minutes is not None else "auto",
                "structuredPrompt": structured_prompt,
                "lyrics": lyrics,
                "requestedCount": count,
                "provider": selected_provider,
                "directPrompt": direct_prompt,
            }
            if duration_seconds is not None:
                response["requestedDurationSeconds"] = duration_seconds
            failures = []
            # Each song is published after its own audio and waveform are ready.
            for index in range(effective_count):
                song_number = current_song = index + 1
                song_state = song_states[index]
                song_state.update(
                    status="running", progress=0 if task_progress is not None else None
                )
                if selected_provider == "mureka_music":
                    song_state["generationStartedAt"] = datetime.now(timezone.utc).isoformat()
                await report(
                    "waiting_music" if selected_provider == "mureka_music" else "generating_music",
                    task_progress,
                    (
                        f"第 {song_number}/{effective_count} 首：等待 Mureka 返回进度"
                        if selected_provider == "mureka_music"
                        else f"音乐模型正在生成第 {song_number}/{effective_count} 首"
                    ),
                    structured_prompt=structured_prompt,
                    lyrics=lyrics,
                    title=song_title,
                )

                async def provider_progress(
                    stage: str,
                    step: int | None,
                    total_steps: int | None,
                    song_number: int = song_number,
                    *,
                    received_audio_seconds: float | None = None,
                    expected_audio_seconds: float | None = None,
                ) -> None:
                    nonlocal task_progress
                    song_progress = None
                    message = f"音乐模型正在生成第 {song_number}/{effective_count} 首"
                    if selected_provider == "elevenlabs_music":
                        message = f"第 {song_number}/{effective_count} 首：等待 ElevenLabs 返回音频"
                    if selected_provider == "mureka_music":
                        detail = (
                            "等待 Mureka 返回进度"
                            if stage == "waiting_music"
                            else "Mureka 正在生成，总时长待确定"
                        )
                        message = f"第 {song_number}/{effective_count} 首：{detail}"
                    if stage == "preparing_reference":
                        detail = "正在准备参考音频"
                        if selected_provider == "elevenlabs_music":
                            detail += "；首次上传及版权检测按服务商规则计费，已上传则复用"
                        message = f"第 {song_number}/{effective_count} 首：{detail}"
                    if received_audio_seconds is not None:
                        message = (
                            f"第 {song_number}/{effective_count} 首："
                            f"已接收 {received_audio_seconds:g} 秒音频"
                        )
                        if expected_audio_seconds is not None:
                            message += f"，目标 {expected_audio_seconds:g} 秒"
                            ratio = min(1, received_audio_seconds / expected_audio_seconds)
                            song_progress = min(99, int(ratio * 100))
                            task_progress = min(
                                99, int((song_number - 1 + ratio) / effective_count * 100)
                            )
                        else:
                            message += "，总时长待确定"
                    await report(
                        stage,
                        task_progress,
                        message,
                        step=step,
                        total_steps=total_steps,
                        received_audio_seconds=received_audio_seconds,
                        expected_audio_seconds=expected_audio_seconds,
                        song_progress=song_progress,
                    )

                try:
                    provider_options: dict[str, Any] = (
                        {"reference": reference} if reference is not None else {}
                    )
                    if direct_prompt and selected_provider == "elevenlabs_music":
                        provider_options["direct_prompt"] = True
                    music_result = await music_provider.generate(
                        structured_prompt,
                        duration_seconds,
                        provider_prompt,
                        variation=index,
                        progress=provider_progress,
                        job_id=job_id,
                        **provider_options,
                    )
                    if task_progress is not None:
                        task_progress = min(99, int(song_number / effective_count * 100))
                    if music_result.title and not song_title:
                        song_title = music_result.title
                    if music_result.lyrics and not lyrics:
                        lyrics = music_result.lyrics
                    response["title"] = song_title
                    response["lyrics"] = lyrics
                    if music_result.title or music_result.lyrics:
                        # job.json 与 prompts.json 都留下服务商返回的歌名/歌词。
                        update_job_diagnostics(
                            self.settings.output_dir,
                            job_id,
                            title=song_title,
                            lyrics=lyrics,
                        )
                    await report(
                        "saving_audio",
                        task_progress,
                        f"正在保存第 {song_number} 首",
                        title=song_title,
                        lyrics=lyrics,
                    )
                    song_dir = job_song_dir(self.settings.output_dir, job_id, index)
                    full_path = await ensure_file_under_root(
                        music_result.audio_path,
                        song_dir,
                        f"full_song_{job_id}_{song_number}{music_result.audio_path.suffix}",
                    )
                    full_relative = full_path.relative_to(self.settings.output_dir)
                    full_playback = await make_playback_mp3(full_path, self.settings.output_dir)

                    await report("waveform", task_progress, f"正在提取第 {song_number} 首真实波形")
                    output = {
                        "songNumber": song_number,
                        "fullTrack": full_relative.as_posix(),
                        **({"playback": {"fullTrack": full_playback}} if full_playback else {}),
                        "stems": {},
                        "stemUrls": [],
                        "waveforms": await extract_waveforms({"full": full_path}),
                        "splitEnabled": False,
                        "durationSeconds": music_result.debug.get("durationSeconds"),
                        **({"lyrics": music_result.lyrics} if music_result.lyrics else {}),
                        "debug": {"music": music_result.debug},
                    }

                except Exception as exc:
                    logger.exception(
                        "song generation failed job_id=%s song=%s", job_id, song_number
                    )
                    song_state.update(status="failed", progress=None, error=str(exc))
                    failures.append(f"第 {song_number} 首生成失败：{exc}")
                    if task_progress is not None:
                        task_progress = min(99, int(song_number / effective_count * 100))
                    if latest_result is not None:
                        latest_result = {**latest_result, "warning": "；".join(failures)}
                    await report("song_failed", task_progress, failures[-1])
                    if isinstance(exc, ProviderGlobalError) or effective_count == 1:
                        raise
                    continue
                outputs.append(output)
                song_state.update(status="succeeded", progress=100)
                latest_result = {
                    **response,
                    **outputs[0],
                    "count": len(outputs),
                    "alternatives": outputs[1:],
                    "warning": "；".join(failures) or None,
                }
                await report(
                    "song_completed", task_progress, f"第 {song_number} 首已生成，可播放和下载"
                )

            current_song = None
            if not outputs:
                raise GenerationError("；".join(failures))
            await report("finalizing", task_progress, "正在校验并整理输出文件")
            await self.events.publish(GenerationEvent("generation.succeeded", job_id, request_id))
            logger.info("generation succeeded job_id=%s request_id=%s", job_id, request_id)
            return latest_result

        async def dispatch() -> dict[str, Any]:
            try:
                return await self.task_dispatcher.submit(job_id, execute)
            except Exception:
                await self.events.publish(GenerationEvent("generation.failed", job_id, request_id))
                raise

        if capacity_reserved:
            return await dispatch()
        async with self.capacity.slot():
            return await dispatch()
