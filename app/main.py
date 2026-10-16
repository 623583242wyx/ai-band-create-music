from __future__ import annotations

import asyncio
import copy
import errno
import hashlib
import json
import logging
import os
import re
import shutil
import stat
import tempfile
import unicodedata
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager, suppress
from dataclasses import dataclass, field
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Annotated, Any, Literal
from uuid import uuid4

import httpx
import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response, StreamingResponse
from pydantic import ValidationError
from starlette.background import BackgroundTask
from starlette.datastructures import Headers
from starlette.types import ASGIApp, Receive, Scope, Send

from app.core.config import Settings
from app.core.errors import CapacityExceededError, GenerationError, public_error_reason
from app.infrastructure.cache import NullCache
from app.infrastructure.events import NullEventPublisher
from app.infrastructure.logging import configure_logging
from app.infrastructure.queue import InlineTaskDispatcher
from app.schemas import (
    CreateGenerationJobResponse,
    ErrorResponse,
    GenerateRequest,
    GenerateResponse,
    GenerationJobResponse,
    RecordingUploadResponse,
    TrackMixRequest,
    UpdateGenerationJobRequest,
)
from app.services.audio_files import (
    build_public_audio_url,
    detect_audio_content_type,
    existing_playback_mp3,
    make_playback_mp3,
    output_path_from_url,
    playback_matches,
    require_readable_file,
)
from app.services.job_files import (
    capture_mix_artifact,
    drop_editor_state,
    invalidate_mix_artifact,
    job_song_dir,
    normalize_mixed_song,
    read_job_diagnostics,
    require_audio_revision,
    reset_mix_state,
    restore_mix_artifact,
    song_project,
    stored_path_exists,
)
from app.services.orchestrator import GenerationOrchestrator
from app.services.prompt import OpenAICompatiblePromptExpander, split_generation_prompt
from app.services.providers import (
    ELEVENLABS_REFERENCE_MODELS,
    AudioReference,
    create_music_provider,
)
from app.services.recordings import prepare_recording_audio
from app.services.references import (
    REFERENCE_DIR,
    REFERENCE_MAX_SECONDS,
    ReferenceInputError,
    clean_reference_name,
    find_reference,
    load_references,
    new_reference_entry,
    prepare_reference_audio,
    reference_audio_name,
    reference_lock,
    references_dir,
    save_references,
)
from app.services.retention import (
    expires_at,
    is_expired,
    is_job_active,
    retained_job,
    retention_enforced,
    retention_lifecycle,
    retention_status,
)
from app.services.stems import (
    BS_ROFORMER_MODEL,
    STEM_NAMES,
    BSRoFormerStemSeparator,
    is_vocal_stem,
)
from app.services.voice import (
    RVCConversionError,
    RVCEngine,
    install_voice_api,
    mix_busy,
    mix_output_filename,
    mix_tracks,
    replacement_filename,
)
from app.services.waveforms import extract_waveforms

logger = logging.getLogger(__name__)
JOB_EVENT_QUEUE_SIZE = 16
WAVEFORM_PROGRESS = 90
SPLIT_COMPLETE_PROGRESS = 100
# 保留合轨现有进度契约。
# 免得运行期的 progress 恒为 null（波形阶段再跳到 WAVEFORM_PROGRESS）。
MIX_PROGRESS = 76
MIX_PREVIEW_PROGRESS = 95
MIX_INPUT_SUFFIXES = frozenset({".wav", ".mp3", ".flac", ".ogg", ".m4a", ".aac", ".webm"})
REPLACE_COMPLETE_PROGRESS = 100
# 替换已经进入终态（超时/取消），但 RVC 推理线程还没退出、因而仍占着一个生成并发额度的数量。
# 额度按设计等到线程真正退出才释放，这里只把它暴露出来，让 /api/health 与日志能发现卡住的替换。
_replacement_wind_down_active = 0


class RequestSizeLimitMiddleware:
    def __init__(
        self, app: ASGIApp, default_limit: int, path_limits: dict[str, int] | None = None
    ) -> None:
        self.app = app
        self.default_limit = default_limit
        self.path_limits = path_limits or {}

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        path = scope["path"]
        limit = self.path_limits.get(path, self.default_limit)
        if re.fullmatch(r"/api/jobs/[^/]+/recordings", path):
            limit = self.path_limits.get("/api/jobs/*/recordings", self.default_limit)
        content_length = Headers(scope=scope).get("content-length")
        if content_length is not None:
            try:
                declared_size = int(content_length)
                if declared_size < 0:
                    raise ValueError
            except ValueError:
                await JSONResponse(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    content={"success": False, "message": "Content-Length 无效。"},
                )(scope, receive, send)
                return
            if declared_size > limit:
                await JSONResponse(
                    status_code=status.HTTP_413_CONTENT_TOO_LARGE,
                    content={"success": False, "message": "请求体过大。"},
                )(scope, receive, send)
                return

        received_size = 0
        too_large = False

        async def limited_receive():
            nonlocal received_size, too_large
            if too_large:
                return {"type": "http.disconnect"}
            message = await receive()
            if message["type"] == "http.request":
                received_size += len(message.get("body", b""))
                if received_size > limit:
                    too_large = True
                    return {"type": "http.disconnect"}
            return message

        async def limited_send(message):
            if not too_large:
                await send(message)

        try:
            await self.app(scope, limited_receive, limited_send)
        except Exception:
            if not too_large:
                raise
        if too_large:
            await JSONResponse(
                status_code=status.HTTP_413_CONTENT_TOO_LARGE,
                content={"success": False, "message": "请求体过大。"},
            )(scope, receive, send)


@dataclass(slots=True)
class GenerationJob:
    job_id: str
    prompt: str
    title: str | None = None
    structured_prompt: str | None = None
    lyrics: str | None = None
    status: str = "pending"
    stage: str = "pending"
    progress: int | None = None
    step: int | None = None
    total_steps: int | None = None
    received_audio_seconds: float | None = None
    expected_audio_seconds: float | None = None
    current_song: int | None = None
    song_states: list[dict[str, Any]] = field(default_factory=list)
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    message: str = "任务已创建"
    # Persisted response compatibility for jobs created by older releases.
    warning: str | None = None
    result: dict[str, Any] | None = None
    error: str | None = None
    task: asyncio.Task[None] | None = None
    deleted_stems: dict[str, dict[str, Any]] = field(default_factory=dict)
    deleted_replaced_vocals: dict[str, dict[str, str]] = field(default_factory=dict)
    # Runtime operation metadata only; never changes the persisted generation state.
    last_operation: str | None = field(default=None, repr=False)
    split_task: asyncio.Task[None] | None = field(default=None, repr=False)
    split_song: int | None = field(default=None, repr=False)
    split_variant: str | None = field(default=None, repr=False)
    split_status: str | None = field(default=None, repr=False)
    split_stage: str | None = field(default=None, repr=False)
    split_progress: int | None = field(default=None, repr=False)
    split_message: str | None = field(default=None, repr=False)
    split_error: str | None = field(default=None, repr=False)
    replace_task: asyncio.Task[None] | None = field(default=None, repr=False)
    replace_song: int | None = field(default=None, repr=False)
    replace_status: str | None = field(default=None, repr=False)
    replace_stage: str | None = field(default=None, repr=False)
    replace_progress: int | None = field(default=None, repr=False)
    replace_message: str | None = field(default=None, repr=False)
    replace_error: str | None = field(default=None, repr=False)
    replace_cancel_requested: bool = field(default=False, repr=False)
    # 判定缓存失效时被摘掉的成品引用与车道，替换失败/取消时还原。
    replace_previous: dict[str, Any] | None = field(default=None, repr=False)
    mix_task: asyncio.Task | None = field(default=None, repr=False)
    mix_song: int | None = field(default=None, repr=False)
    mix_status: str | None = field(default=None, repr=False)
    mix_stage: str | None = field(default=None, repr=False)
    mix_progress: int | None = field(default=None, repr=False)
    mix_message: str | None = field(default=None, repr=False)
    mix_error: str | None = field(default=None, repr=False)
    mix_cancel_requested: bool = field(default=False, repr=False)

    def response(self) -> dict[str, Any]:
        return {
            "jobId": self.job_id,
            "createdAt": self.created_at,
            "title": self.title,
            "step": self.step,
            "totalSteps": self.total_steps,
            "receivedAudioSeconds": self.received_audio_seconds,
            "expectedAudioSeconds": self.expected_audio_seconds,
            "currentSong": self.current_song,
            "songStates": self.song_states,
            "prompt": self.prompt,
            "structuredPrompt": self.structured_prompt,
            "lyrics": self.lyrics,
            "status": self.status,
            "stage": self.stage,
            "progress": self.progress,
            "message": self.message,
            "warning": self.warning,
            "result": self.result,
            "error": self.error,
        }

    def finish_generation(self) -> None:
        self.current_song = None
        for song in self.song_states:
            if song["status"] in {"pending", "running"}:
                song.update(
                    status=self.status,
                    stage=self.stage,
                    message=self.error or self.message,
                    progress=None,
                    step=None,
                    totalSteps=None,
                    receivedAudioSeconds=None,
                    expectedAudioSeconds=None,
                    error=self.error or self.message,
                )

    def save(self, output_dir: Path) -> None:
        target = output_dir / "jobs" / self.job_id / "job.json"
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_name(f"job.{uuid4().hex}.tmp")
        try:
            with temporary.open("w", encoding="utf-8") as stream:
                payload = {
                    **self.response(),
                    "deletedStems": self.deleted_stems,
                }
                if self.deleted_replaced_vocals:
                    payload["deletedReplacedVocals"] = self.deleted_replaced_vocals
                diagnostics = read_job_diagnostics(output_dir, self.job_id)
                if diagnostics:
                    payload["diagnostics"] = diagnostics
                json.dump(payload, stream, ensure_ascii=False)
                stream.flush()
                os.fsync(stream.fileno())
            temporary.replace(target)
        finally:
            temporary.unlink(missing_ok=True)


def load_jobs(
    output_dir: Path, metadata_errors: dict[str, str] | None = None
) -> dict[str, GenerationJob]:
    jobs = {}
    for target in (output_dir / "jobs").glob("*/job.json"):
        safe_message = None
        try:
            data = json.loads(target.read_text(encoding="utf-8"))
            discarded_state = False
            for output in [
                data.get("result"),
                *((data.get("result") or {}).get("alternatives") or []),
            ]:
                if isinstance(output, dict):
                    discarded_state = drop_editor_state(output) or discarded_state
                    discarded_state = normalize_mixed_song(output) or discarded_state
            if discarded_state:
                logger.info("discarding obsolete editor settings job_id=%s", target.parent.name)
            validated = GenerationJobResponse.model_validate(data)
            if validated.jobId != target.parent.name:
                safe_message = "jobId does not match directory"
                raise ValueError(safe_message)
            job = GenerationJob(
                job_id=validated.jobId,
                prompt=validated.prompt,
                title=validated.title,
                created_at=validated.createdAt,
                structured_prompt=validated.structuredPrompt,
                lyrics=validated.lyrics,
                status=validated.status,
                stage=validated.stage,
                progress=validated.progress,
                step=validated.step,
                total_steps=validated.totalSteps,
                received_audio_seconds=validated.receivedAudioSeconds,
                expected_audio_seconds=validated.expectedAudioSeconds,
                current_song=validated.currentSong,
                song_states=[song.model_dump() for song in validated.songStates],
                message=validated.message,
                warning=validated.warning,
                error=validated.error,
                result=data.get("result"),
                deleted_stems=data.get("deletedStems", {}),
                deleted_replaced_vocals=data.get("deletedReplacedVocals", {}),
            )
            repaired = discarded_state
            for output in [job.result, *((job.result or {}).get("alternatives") or [])]:
                mixed = song_project(output, "mixed")
                if mixed is None or not isinstance(mixed.get("fullTrack"), str):
                    continue
                try:
                    artifact = output_path_from_url(mixed["fullTrack"], output_dir)
                    missing = not artifact.is_file()
                except ValueError:
                    missing = True
                except OSError as exc:
                    # 权限/IO 问题不等于文件不存在（`Path.is_file()` 对 EACCES 会抛
                    # PermissionError）。判断不了就保留引用——绝不能让一个读不到的成品
                    # 把整个任务从 API 里抹掉。
                    logger.warning(
                        "cannot verify mix reference job_id=%s path=%s error=%s",
                        job.job_id,
                        mixed["fullTrack"],
                        exc,
                    )
                    continue
                if missing:
                    # 元数据先落盘、文件后替换（或文件被外部清掉）时会留下这种引用；
                    # 对外报"有成品"却 404 比直接当作没有成品更糟。
                    logger.warning(
                        "dropping mix reference without a file job_id=%s path=%s",
                        job.job_id,
                        mixed["fullTrack"],
                    )
                    output.pop("mixed")
                    repaired = True
            if job.status in {"pending", "running"}:
                job.status = job.stage = "failed"
                job.progress = job.step = job.total_steps = None
                job.received_audio_seconds = job.expected_audio_seconds = None
                job.error = job.message = "服务器重启，生成任务已中断。"
                job.finish_generation()
                repaired = True
            if repaired:
                job.save(output_dir)
            jobs[job.job_id] = job
        except (OSError, ValueError, TypeError) as exc:
            if metadata_errors is not None:
                if isinstance(exc, ValidationError):
                    reason = "; ".join(
                        f"{'.'.join(map(str, error['loc']))}: {error['type']}"
                        for error in exc.errors(include_input=False, include_context=False)[:3]
                    )
                else:
                    reason = public_error_reason(exc, safe_message=safe_message)
                metadata_errors[target.parent.name] = reason[:160]
            logger.exception("Unable to load job metadata: %s", target)
    for leftover in sorted((output_dir / "jobs").glob("*/song_*/.mix-*")):
        # 进程被硬杀时，混音用的临时目录（内含体积不小的 premix.wav）会留在歌曲目录下；
        # 启动时没有任何合轨在跑，直接清掉，避免长期堆积。
        logger.warning("removing stale mix workdir left by an interrupted mix: %s", leftover)
        shutil.rmtree(leftover, ignore_errors=True)
    return jobs


def _song_result(job: GenerationJob, song: int) -> dict[str, Any] | None:
    if job.result is None or song < 0:
        return None
    if song == 0:
        return job.result
    alternatives = job.result.get("alternatives")
    if not isinstance(alternatives, list) or song > len(alternatives):
        return None
    result = alternatives[song - 1]
    return result if isinstance(result, dict) else None


def _playback_dict(output: dict[str, Any]) -> dict[str, Any]:
    """可写的 `playback` 字典。

    `playback` 在响应模型里允许是 null，而 job.json 里的 result 不经过模型再赋值就到
    这里，所以缺失和 null 都得当作"还没有"处理，不能直接 `.pop()`/取下标。
    """
    playback = output.get("playback")
    if not isinstance(playback, dict):
        playback = {}
        output["playback"] = playback
    return playback


def build_orchestrator(
    settings: Settings,
    client: httpx.AsyncClient,
    direct_client: httpx.AsyncClient | None = None,
) -> GenerationOrchestrator:
    # Instantiate the reserved local cache so the wiring point stays explicit. The
    # current pipeline intentionally performs no cache reads or writes.
    NullCache()
    music_providers = {
        name: create_music_provider(settings, client, direct_client, name)
        for name in ("minimax_music", "elevenlabs_music", "mureka_music")
    }
    return GenerationOrchestrator(
        settings=settings,
        prompt_expander=OpenAICompatiblePromptExpander(settings, client),
        music_provider=music_providers.get(settings.music_provider)
        or create_music_provider(settings, client, direct_client),
        stem_separator=BSRoFormerStemSeparator(settings),
        task_dispatcher=InlineTaskDispatcher(),
        events=NullEventPublisher(),
        music_providers=music_providers,
    )


_SUPPORTED_PROXY_SCHEMES = frozenset({"http", "https", "socks5", "socks5h"})


def _sanitize_proxy_environment() -> None:
    """Make proxy environment variables consumable by httpx.

    httpx only accepts ``http``/``https``/``socks5``/``socks5h`` proxy schemes and
    raises ``ValueError`` at client construction time for anything else. Real
    machines often export proxy variables with schemes httpx rejects, e.g. the
    curl-style ``socks://host:port`` alias (urllib can surface it from either the
    uppercase or lowercase ``*_proxy`` variable, whichever wins iteration order)
    or ``socks4://``. Rewrite ``socks://`` to ``socks5://`` and drop variables
    whose scheme httpx cannot use, so the application starts successfully whether
    or not a proxy is configured.
    """
    for name in list(os.environ):
        if not name.lower().endswith("_proxy"):
            continue
        value = os.environ[name]
        if not value:
            continue
        lowered = value.lower()
        if lowered.startswith("socks://"):
            os.environ[name] = "socks5://" + value[len("socks://") :]
        elif "://" in lowered:
            scheme = lowered.split("://", 1)[0]
            if scheme not in _SUPPORTED_PROXY_SCHEMES:
                del os.environ[name]


def create_app(
    settings: Settings | None = None,
    orchestrator: GenerationOrchestrator | None = None,
    voice_engine: RVCEngine | None = None,
) -> FastAPI:
    application_settings = settings or Settings()
    application_settings.output_dir.mkdir(parents=True, exist_ok=True)

    @asynccontextmanager
    async def lifespan(application: FastAPI):
        async with retention_lifecycle(application):
            if orchestrator is not None:
                application.state.orchestrator = orchestrator
                yield
                return
            # Tolerate proxy environments httpx would otherwise reject at client
            # construction (e.g. `socks://...` exported by common proxy clients).
            _sanitize_proxy_environment()
            async with (
                httpx.AsyncClient() as client,
                httpx.AsyncClient(trust_env=False) as direct_client,
            ):
                application.state.orchestrator = build_orchestrator(
                    application_settings, client, direct_client
                )
                yield

    application = FastAPI(
        title="AI Band Music Generation API",
        version="1.0.0",
        docs_url="/docs",
        redoc_url=None,
        lifespan=lifespan,
    )
    application.state.settings = application_settings
    application.state.job_metadata_errors = {}
    application.state.jobs = load_jobs(
        application_settings.output_dir, application.state.job_metadata_errors
    )
    application.state.job_subscribers = {}
    # Startup runs before this app's event loop serves requests. Hash the large RVC assets
    # once here so request/SSE paths only read the cached value.
    application.state.rvc_model_fingerprint = _rvc_model_fingerprint(application_settings)
    install_voice_api(application, application_settings, voice_engine)
    if orchestrator is not None:
        application.state.orchestrator = orchestrator

    cors_origins = application_settings.cors_origin_list
    application.add_middleware(
        CORSMiddleware,
        allow_origins=cors_origins,
        allow_credentials="*" not in cors_origins,
        allow_methods=["*"],
        allow_headers=["*"],
        expose_headers=[
            "Content-Disposition",
            "X-Mix-Output",
            "X-RVC-Model",
            "X-RVC-Output",
        ],
    )
    application.add_middleware(
        RequestSizeLimitMiddleware,
        default_limit=application_settings.request_max_bytes,
        # Multipart framing adds a little on top of the audio file itself.
        path_limits={
            "/api/references": application_settings.audio_reference_max_upload_bytes
            + 1024 * 1024,
            "/api/jobs/*/recordings": (application_settings.recording_max_upload_mb + 1)
            * 1024
            * 1024,
        },
    )

    @application.get("/api/health")
    async def health(request: Request) -> dict[str, object]:
        active_orchestrator = _orchestrator(request)
        base_url = _public_base_url(request, application_settings)
        return {
            "ok": True,
            "service": "AI Band System Backend",
            "outputUrl": f"{base_url}/output",
            "maxConcurrentGenerations": active_orchestrator.capacity.maximum,
            "activeGenerationCount": active_orchestrator.capacity.active,
            "duration": {
                "defaultMinutes": application_settings.default_duration_minutes,
                "minMinutes": application_settings.min_duration_minutes,
                "maxMinutes": application_settings.max_duration_minutes,
            },
            "llm": {
                "model": application_settings.llm_model,
                "timeoutSeconds": application_settings.llm_timeout_seconds,
                "maxTokens": application_settings.llm_max_tokens,
                "disableThinking": application_settings.llm_disable_thinking,
            },
            "elevenLabs": {
                "modelId": application_settings.elevenlabs_music_model_id,
                "outputFormat": application_settings.elevenlabs_music_output_format,
                "clearChineseVocalMode": application_settings.elevenlabs_clear_chinese_vocal_mode,
                "audioReference": {
                    "supported": application_settings.elevenlabs_music_model_id
                    in ELEVENLABS_REFERENCE_MODELS
                    and not application_settings.elevenlabs_force_instrumental,
                    "maxSeconds": REFERENCE_MAX_SECONDS,
                    "maxUploadMb": application_settings.audio_reference_max_upload_mb,
                    "conditionStrength": (
                        application_settings.elevenlabs_reference_condition_strength
                    ),
                },
            },
            "splitting": {
                "enabled": True,
                "mode": "on_demand",
                "engine": "bs_roformer",
                "model": BS_ROFORMER_MODEL,
                "device": application_settings.bs_roformer_device,
                "stems": list(STEM_NAMES),
                "timeoutSeconds": application_settings.bs_roformer_timeout_seconds,
            },
            "mixing": {
                "timeoutSeconds": application_settings.rvc_mix_timeout_seconds,
                # 模型指纹守卫只在资产确实在场时才生效；缺挂载的实例会放行合轨（见 README），
                # 这里显式暴露，运维不必去翻日志里那条 warning。
                "modelGuardEnforced": _rvc_assets_present(application_settings),
            },
            "replacement": {
                "conversionTimeoutSeconds": application_settings.rvc_conversion_timeout_seconds,
                # 已进入终态但 RVC 线程仍未退出的替换数量；这些替换按设计继续占着生成并发额度。
                "workersHoldingCapacityAfterTerminal": _replacement_wind_down_active,
            },
            "infrastructure": {
                "taskBackend": application_settings.task_backend,
                "cacheBackend": application_settings.cache_backend,
                "eventBackend": application_settings.event_backend,
            },
            "retention": retention_status(request.app),
        }

    @application.post(
        "/api/generate",
        response_model=GenerateResponse,
        responses={
            400: {"model": ErrorResponse},
            413: {"model": ErrorResponse},
            429: {"model": ErrorResponse},
            500: {"model": ErrorResponse},
        },
    )
    async def generate(payload: GenerateRequest, request: Request):
        try:
            prompt, duration = _generation_parameters(payload, application_settings)
            reference = _audio_reference(payload, application_settings)
        except ValueError as exc:
            return JSONResponse(
                status_code=status.HTTP_400_BAD_REQUEST,
                content={"success": False, "message": str(exc)},
            )
        request_id = request.headers.get("X-Request-ID") or uuid4().hex
        job = GenerationJob(job_id=f"job_{uuid4().hex}", prompt=prompt, title=payload.title)
        job.save(application_settings.output_dir)
        request.app.state.jobs[job.job_id] = job

        async def report(
            stage,
            progress,
            message,
            *,
            structured_prompt=None,
            lyrics=None,
            step=None,
            total_steps=None,
            title=None,
            received_audio_seconds=None,
            expected_audio_seconds=None,
            current_song=None,
            song_states=None,
            result=None,
        ):
            job.status = "running"
            job.stage, job.progress, job.message = stage, progress, message
            job.step, job.total_steps = step, total_steps
            job.received_audio_seconds = received_audio_seconds
            job.expected_audio_seconds = expected_audio_seconds
            job.current_song = current_song
            if song_states is not None:
                job.song_states = copy.deepcopy(song_states)
            if result is not None:
                job.result = copy.deepcopy(result)
                job.result["createdAt"] = job.created_at
                job.warning = result.get("warning")
            if structured_prompt is not None:
                job.structured_prompt = structured_prompt
            if lyrics is not None:
                job.lyrics = lyrics
            if title is not None:
                job.title = title
            job.save(application_settings.output_dir)

            _publish_job(job, request)

        try:
            result = await _orchestrator(request).generate(
                prompt,
                duration,
                request_id,
                job_id=job.job_id,
                progress=report,
                provider=payload.provider,
                count=payload.count,
                title=job.title,
                reference=reference,
            )
            job.result = result
            job.status, job.stage, job.progress = "succeeded", "completed", 100
            job.message = result.get("warning") or "音乐生成完成"
            job.warning = result.get("warning")
            result["createdAt"] = job.created_at
            job.save(application_settings.output_dir)
            request.app.state.jobs[job.job_id] = job
            return _render_result_urls(
                result, _public_base_url(request, application_settings), application_settings
            )
        except asyncio.CancelledError:
            job.status = job.stage = "cancelled"
            job.progress = None
            job.message = "任务已取消"
            raise
        except HTTPException as exc:
            job.error = str(exc)
            raise
        except CapacityExceededError as exc:
            job.error = str(exc)
            return JSONResponse(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                content={"success": False, "message": str(exc)},
            )
        except GenerationError as exc:
            job.error = str(exc)
            logger.exception("generation failed request_id=%s", request_id)
            return JSONResponse(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                content={"success": False, "message": str(exc)},
            )
        except Exception as exc:
            job.error = str(exc)
            logger.exception("unexpected generation failure request_id=%s", request_id)
            return JSONResponse(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                content={"success": False, "message": "生成失败，请查看后端日志。"},
            )
        finally:
            if job.status in {"pending", "running"}:
                job.status = job.stage = "failed"
                job.progress = None
                job.error = job.error or "生成任务已中断。"
                job.message = job.error
            job.step = job.total_steps = None
            job.received_audio_seconds = job.expected_audio_seconds = None
            job.finish_generation()
            job.save(application_settings.output_dir)
            _publish_job(job, request)

    @application.post(
        "/api/jobs",
        response_model=CreateGenerationJobResponse,
        status_code=status.HTTP_202_ACCEPTED,
        responses={400: {"model": ErrorResponse}, 429: {"model": ErrorResponse}},
    )
    async def create_generation_job(payload: GenerateRequest, request: Request):
        try:
            prompt, duration = _generation_parameters(payload, application_settings)
            reference = _audio_reference(payload, application_settings)
        except ValueError as exc:
            return JSONResponse(
                status_code=status.HTTP_400_BAD_REQUEST,
                content={"success": False, "message": str(exc)},
            )

        active_orchestrator = _orchestrator(request)
        try:
            await active_orchestrator.capacity.acquire()
        except CapacityExceededError:
            return JSONResponse(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                content={"success": False, "message": "当前生成任务过多，请稍后重试。"},
            )

        jobs: dict[str, GenerationJob] = request.app.state.jobs
        job_id = f"job_{int(asyncio.get_running_loop().time() * 1000)}_{uuid4().hex[:8]}"
        job = GenerationJob(
            job_id=job_id,
            prompt=prompt,
            title=payload.title,
        )

        def publish() -> None:
            _publish_job(job, request)

        def save_and_publish() -> None:
            job.save(application_settings.output_dir)
            publish()

        try:
            job.save(application_settings.output_dir)
        except Exception:
            await active_orchestrator.capacity.release()
            raise
        jobs[job_id] = job
        request_id = request.headers.get("X-Request-ID") or uuid4().hex

        async def report(
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
        ) -> None:
            job.status = "running"
            job.stage = stage
            job.progress = progress
            job.step = step
            job.total_steps = total_steps
            job.received_audio_seconds = received_audio_seconds
            job.expected_audio_seconds = expected_audio_seconds
            job.current_song = current_song
            if song_states is not None:
                job.song_states = copy.deepcopy(song_states)
            if result is not None:
                job.result = copy.deepcopy(result)
                job.result["createdAt"] = job.created_at
                job.warning = result.get("warning")
            job.message = message
            if structured_prompt is not None:
                job.structured_prompt = structured_prompt
            if lyrics is not None:
                job.lyrics = lyrics
            if title is not None:
                job.title = title
            save_and_publish()

        async def execute() -> None:
            try:
                job.status = "running"
                job.message = "服务器正在生成音乐"
                save_and_publish()
                job.result = await active_orchestrator.generate(
                    prompt,
                    duration,
                    request_id,
                    job_id=job_id,
                    progress=report,
                    capacity_reserved=True,
                    provider=payload.provider,
                    count=payload.count,
                    title=job.title,
                    reference=reference,
                )
                job.status = "succeeded"
                job.result["createdAt"] = job.created_at
                job.stage = "completed"
                job.progress = 100
                job.message = job.result.get("warning") or "音乐生成完成"
                job.warning = job.result.get("warning")
            except asyncio.CancelledError:
                job.status = "cancelled"
                job.stage = "cancelled"
                job.message = "任务已取消"
            except Exception as exc:
                logger.exception("background generation failed job_id=%s", job_id)
                job.status = "failed"
                job.stage = "failed"
                job.message = "音乐生成失败"
                job.error = str(exc)
            finally:
                job.step = job.total_steps = None
                job.received_audio_seconds = job.expected_audio_seconds = None
                job.finish_generation()
                if job.status != "succeeded":
                    job.progress = None
                await active_orchestrator.capacity.release()
                save_and_publish()

        job.task = asyncio.create_task(execute(), name=job_id)
        return {"jobId": job_id, "status": job.status}

    def _reference_response(entry: dict[str, Any], request: Request) -> dict[str, Any]:
        audio = entry.get("audio")
        return {
            "id": entry["id"],
            "provider": entry.get("provider"),
            "name": entry.get("name") or "参考音频",
            "durationSeconds": entry["durationSeconds"],
            "createdAt": entry.get("createdAt"),
            "audioUrl": build_public_audio_url(
                _public_base_url(request, application_settings), f"{REFERENCE_DIR}/{audio}"
            )
            if isinstance(audio, str) and audio
            else None,
        }

    @application.get("/api/references")
    async def list_audio_references(
        request: Request,
        provider: Literal["elevenlabs_music", "mureka_music"] = "elevenlabs_music",
    ):
        return {
            "provider": provider,
            "references": [
                _reference_response(entry, request)
                for entry in load_references(application_settings.output_dir)
                if entry.get("provider") == provider
            ],
            "maxSeconds": REFERENCE_MAX_SECONDS,
            "maxUploadMb": application_settings.audio_reference_max_upload_mb,
        }

    @application.post(
        "/api/references",
        status_code=status.HTTP_201_CREATED,
        responses={
            400: {"model": ErrorResponse},
            413: {"model": ErrorResponse},
            500: {"model": ErrorResponse},
        },
    )
    async def upload_audio_reference(
        request: Request,
        file: Annotated[UploadFile, File()],
        provider: Literal["elevenlabs_music", "mureka_music"] = "elevenlabs_music",
    ):
        def error(code: int, message: str) -> JSONResponse:
            return JSONResponse(status_code=code, content={"success": False, "message": message})

        name = clean_reference_name(file.filename)
        directory = references_dir(application_settings.output_dir)
        directory.mkdir(parents=True, exist_ok=True)
        limit = application_settings.audio_reference_max_upload_bytes
        descriptor, raw_name = tempfile.mkstemp(dir=directory, prefix=".upload.")
        raw_path = Path(raw_name)
        prepared = directory / reference_audio_name()
        registered = False
        try:
            written = 0
            with os.fdopen(descriptor, "wb") as raw:
                while chunk := await file.read(1024 * 1024):
                    written += len(chunk)
                    if written > limit:
                        return error(
                            413,
                            "参考音频不能超过 "
                            f"{application_settings.audio_reference_max_upload_mb} MB。",
                        )
                    raw.write(chunk)
            if written == 0:
                return error(400, "上传的文件为空。")
            try:
                duration = await prepare_reference_audio(raw_path, prepared)
            except ReferenceInputError as exc:
                return error(400, str(exc))
            except GenerationError as exc:
                logger.error("audio reference preprocessing unavailable: %s", exc)
                return error(500, str(exc))
            # The random local name, rather than the client filename, is the reference id.
            entry = new_reference_entry(prepared.stem, name, duration, prepared.name)
            entry["provider"] = provider
            entry["localOnly"] = True
            entry["sha256"] = hashlib.sha256(prepared.read_bytes()).hexdigest()
            async with reference_lock(application_settings.output_dir):
                existing = load_references(application_settings.output_dir)
                duplicate = None
                for old in existing:
                    if old.get("provider") != provider:
                        continue
                    audio = directory / Path(str(old.get("audio") or "")).name
                    if not audio.is_file():
                        continue
                    digest = old.get("sha256") or hashlib.sha256(audio.read_bytes()).hexdigest()
                    if digest == entry["sha256"]:
                        duplicate = old
                        break
                if duplicate is not None:
                    entry = duplicate
                    if "sha256" not in entry:
                        entry["sha256"] = digest
                        save_references(application_settings.output_dir, existing)
                else:
                    save_references(application_settings.output_dir, [entry, *existing])
                    registered = True
            return JSONResponse(
                status_code=status.HTTP_201_CREATED if registered else status.HTTP_200_OK,
                content=_reference_response(entry, request),
            )
        finally:
            await file.close()
            raw_path.unlink(missing_ok=True)
            if not registered:
                prepared.unlink(missing_ok=True)

    @application.delete(
        "/api/references/{reference_id}",
        status_code=status.HTTP_204_NO_CONTENT,
        responses={404: {"model": ErrorResponse}},
    )
    async def delete_audio_reference(
        reference_id: str,
        request: Request,
        provider: Literal["elevenlabs_music", "mureka_music"] = "elevenlabs_music",
    ):
        async with reference_lock(application_settings.output_dir):
            existing = load_references(application_settings.output_dir)
            entry = next(
                (
                    item
                    for item in existing
                    if item["id"] == reference_id and item.get("provider") == provider
                ),
                None,
            )
            if entry is None:
                return JSONResponse(
                    status_code=status.HTTP_404_NOT_FOUND,
                    content={"success": False, "message": "参考音频不存在。"},
                )
            save_references(
                application_settings.output_dir,
                [item for item in existing if item["id"] != reference_id],
            )
        audio = Path(str(entry.get("audio") or "")).name
        if audio:
            (references_dir(application_settings.output_dir) / audio).unlink(missing_ok=True)
        return Response(status_code=status.HTTP_204_NO_CONTENT)

    @application.get("/api/jobs")
    async def generation_history(request: Request):
        now = datetime.now(timezone.utc)
        return {
            "jobs": [
                _job_response(
                    job,
                    request,
                    application_settings,
                    include_operations=False,
                    include_waveforms=False,
                )
                for job in sorted(
                    request.app.state.jobs.values(), key=lambda job: job.created_at, reverse=True
                )
                if not retention_enforced(application_settings)
                or not is_expired(job, application_settings.song_retention_days, now)
                or is_job_active(job)
            ]
        }

    @application.get("/api/jobs/{job_id}/events")
    async def generation_job_events(job_id: str, request: Request):
        job = retained_job(request, job_id, allow_active=True)
        if job is None:
            return JSONResponse(
                status_code=status.HTTP_404_NOT_FOUND,
                content={"success": False, "message": "生成任务不存在。"},
            )

        fingerprint = request.app.state.rvc_model_fingerprint

        async def events() -> AsyncIterator[str]:
            # 操作阶段保存快照，避免快速阶段被后续状态覆盖；生成进度仍按现有方式合并。
            queue: asyncio.Queue[GenerationJob | None] = asyncio.Queue(maxsize=JOB_EVENT_QUEUE_SIZE)
            subscribers = request.app.state.job_subscribers.setdefault(job_id, set())
            subscribers.add(queue)

            def frame(snapshot: GenerationJob) -> tuple[bool, str]:
                payload = _job_response(
                    snapshot,
                    request,
                    application_settings,
                    include_waveforms=(
                        (_operation_status(snapshot) or snapshot.status)
                        not in {"pending", "running"}
                        or snapshot.stage == "song_completed"
                    ),
                    fingerprint=fingerprint,
                )
                return (payload["operationStatus"] or payload["status"]) not in {
                    "pending",
                    "running",
                }, json.dumps(payload, ensure_ascii=False)

            snapshot = job

            try:
                while True:
                    if (
                        retention_enforced(application_settings)
                        and not is_job_active(job)
                        and is_expired(job, application_settings.song_retention_days)
                    ):
                        body = json.dumps(
                            {
                                "success": False,
                                "status": "expired",
                                "jobId": job_id,
                                "message": "歌曲已过期。",
                            },
                            ensure_ascii=False,
                        )
                        yield f"event: done\ndata: {body}\n\n"
                        return
                    # 失败/取消优先于缓冲的阶段快照，RVC 收尾不延迟终态通知。
                    if _operation_status(job) in {"failed", "cancelled"}:
                        snapshot = job
                    final, body = frame(snapshot)
                    if final:
                        yield f"event: done\ndata: {body}\n\n"
                        return
                    yield f"data: {body}\n\n"
                    try:
                        snapshot = await asyncio.wait_for(queue.get(), timeout=15) or job
                    except asyncio.TimeoutError:
                        yield ": keep-alive\n\n"
            finally:
                subscribers.discard(queue)
                if not subscribers:
                    request.app.state.job_subscribers.pop(job_id, None)

        return StreamingResponse(
            events(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    @application.get(
        "/api/jobs/{job_id}",
        response_model=GenerationJobResponse,
        responses={404: {"model": ErrorResponse}},
    )
    async def generation_job(job_id: str, request: Request):
        job = retained_job(request, job_id, allow_active=True)
        if job is None:
            return JSONResponse(
                status_code=status.HTTP_404_NOT_FOUND,
                content={"success": False, "message": "生成任务不存在。"},
            )
        return await _async_job_response(job, request, application_settings)

    @application.post(
        "/api/jobs/{job_id}/split",
        response_model=GenerationJobResponse,
        status_code=status.HTTP_202_ACCEPTED,
        responses={
            200: {"model": GenerationJobResponse},
            404: {"model": ErrorResponse},
            409: {"model": ErrorResponse},
            429: {"model": ErrorResponse},
        },
    )
    async def split_generation_job(
        job_id: str,
        request: Request,
        response: Response,
        song: int = 0,
        force: bool = False,
        variant: Literal["original", "mixed"] = "original",
    ):
        job = retained_job(request, job_id)
        if job is None:
            return JSONResponse(
                status_code=status.HTTP_404_NOT_FOUND,
                content={"success": False, "message": "生成任务不存在。"},
            )
        if job.status != "succeeded":
            return JSONResponse(
                status_code=status.HTTP_409_CONFLICT,
                content={"success": False, "message": "生成任务尚未完成，无法拆轨。"},
            )
        song_output = _song_result(job, song)
        result = song_project(song_output, variant)
        if result is None or not isinstance(result.get("fullTrack"), str):
            return JSONResponse(
                status_code=status.HTTP_409_CONFLICT,
                content={"success": False, "message": "该歌曲没有可供拆轨的完整音频。"},
            )
        # Checked before the cached-stems response: a mix reads the same stem files that
        # deleting or re-splitting would replace, so "nothing to do" must not be the reply.
        if mix_busy(job):
            return JSONResponse(
                status_code=status.HTTP_409_CONFLICT,
                content={"success": False, "message": "该任务正在合轨，请稍后重试。"},
            )
        if job.replace_task is not None and not job.replace_task.done():
            return JSONResponse(
                status_code=status.HTTP_409_CONFLICT,
                content={
                    "success": False,
                    "message": f"第 {(job.replace_song or 0) + 1} 首歌曲正在替换人声。",
                },
            )
        if job.split_status in {"pending", "running"}:
            if job.split_song == song and (job.split_variant or "original") == variant:
                return await _async_job_response(job, request, application_settings)
            return JSONResponse(
                status_code=status.HTTP_409_CONFLICT,
                content={
                    "success": False,
                    "message": f"第 {(job.split_song or 0) + 1} 首歌曲正在拆轨。",
                },
            )
        stems = result.get("stems")
        if force:
            try:
                require_audio_revision(result, request.headers.get("X-Audio-Revision"))
            except HTTPException as exc:
                return JSONResponse(
                    status_code=exc.status_code,
                    content={"success": False, "message": exc.detail},
                )
        if not force and (result.get("splitEnabled") or (isinstance(stems, dict) and stems)):
            job.last_operation = "split"
            job.split_song = song
            job.split_variant = variant
            job.split_error = None
            _mark_split_succeeded(job)
            response.status_code = status.HTTP_200_OK
            return await _async_job_response(job, request, application_settings)
        try:
            full_path = _output_path_from_url(result["fullTrack"], application_settings)
            if not full_path.is_file():
                raise OSError(f"missing full track: {full_path}")
        except (OSError, ValueError):
            return JSONResponse(
                status_code=status.HTTP_409_CONFLICT,
                content={"success": False, "message": "完整音频文件不存在，无法拆轨。"},
            )

        active_orchestrator = _orchestrator(request)
        job.split_song = song
        job.split_variant = variant
        job.split_status = "pending"
        job.split_stage = "starting_split"
        job.split_progress = 0
        job.split_message = "正在读取音频"
        job.split_error = None
        try:
            await active_orchestrator.capacity.acquire()
        except CapacityExceededError:
            job.split_song = None
            job.split_variant = None
            job.split_status = None
            job.split_stage = None
            job.split_progress = None
            job.split_message = None
            return JSONResponse(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                content={"success": False, "message": "当前拆轨任务过多，请稍后重试。"},
            )
        except asyncio.CancelledError:
            job.split_song = None
            job.split_variant = None
            job.split_status = None
            job.split_stage = None
            job.split_progress = None
            job.split_message = None
            raise

        job.last_operation = "split"
        previous_result = copy.deepcopy(result)
        previous_deleted_stems = copy.deepcopy(job.deleted_stems)
        previous_deleted_vocals = copy.deepcopy(job.deleted_replaced_vocals)

        def publish() -> None:
            _publish_job(job, request, snapshot=True)

        async def execute() -> None:
            wav_ready = False
            capacity_released = False
            generated_paths: list[Path] = []
            try:
                job.split_status = "running"
                publish()
                output_dir = job_song_dir(
                    application_settings.output_dir,
                    job_id,
                    (song_output.get("songNumber") or song + 1) - 1,
                )
                relative_output = output_dir.relative_to(application_settings.output_dir)
                await asyncio.to_thread(_read_operation_input, full_path)
                job.split_stage = "splitting"
                job.split_progress = 25
                job.split_message = "正在分离音轨"
                publish()
                split_result = await active_orchestrator.stem_separator.split(full_path, output_dir)
                generated_paths = [output_dir / name for name in split_result.files.values()]
                stems = {
                    name: (relative_output / file_name).as_posix()
                    for name, file_name in split_result.files.items()
                }
                job.split_stage = "waveform"
                job.split_progress = 50
                job.split_message = "正在提取波形"
                publish()

                # 连同完整混音一起重新提取：本 PR 之前的任务把 "full" 存成 64 个 bin，
                # 直接与 640 个 bin 的分轨合并会让客户端拿到长度不一致的波形。
                def waveform_progress(finished: int, total: int) -> None:
                    job.split_progress = 50 + round(25 * finished / total)
                    publish()

                fresh_waveforms = await extract_waveforms(
                    {
                        "full": full_path,
                        **{
                            name: application_settings.output_dir / relative_output / file_name
                            for name, file_name in split_result.files.items()
                        },
                    },
                    progress=waveform_progress,
                )
                # 用户上传的录音（stemNames 里有显示名的音轨）不是模型产物，重新分轨时原样保留。
                previous_stems = result.get("stems")
                if not isinstance(previous_stems, dict):
                    previous_stems = {}
                previous_names = result.get("stemNames") or {}
                recordings = {
                    name: url
                    for name, url in previous_stems.items()
                    if name in previous_names and name not in stems
                }
                result["stems"] = {**stems, **recordings}
                if recordings:
                    result["stemNames"] = {name: previous_names[name] for name in recordings}
                else:
                    result.pop("stemNames", None)
                playback = _playback_dict(result)
                recording_previews = {
                    name: preview
                    for name, preview in (playback.pop("stems", None) or {}).items()
                    if name in recordings
                }
                if recording_previews:
                    playback["stems"] = dict(recording_previews)
                if variant == "original":
                    job.deleted_replaced_vocals.pop(str(song), None)
                playback.pop("replacedVocal", None)
                result.pop("replacedVocal", None)
                result.pop("_replacedVocalModel", None)
                # 旧方案的工程状态（编辑器控制与单轨原输入）按"未提交设置"丢弃。新版的
                # mixConfig 是已提交成品的凭据，而重新分轨用的是同一份 fullTrack、成品仍然
                # 成立，所以它照旧保留（文档：成功终态继续保留此配置）。
                drop_editor_state(result)
                result["audioRevision"] = result.get("audioRevision", 0) + 1
                invalidate_mix_artifact(job, result)
                result["stemUrls"] = list(result["stems"].values())
                previous_waveforms = result.get("waveforms")
                merged_waveforms = dict(fresh_waveforms)
                if isinstance(previous_waveforms, dict):
                    merged_waveforms.update(
                        (name, previous_waveforms[name])
                        for name in recordings
                        if name in previous_waveforms
                    )
                if (
                    "full" not in merged_waveforms
                    and isinstance(previous_waveforms, dict)
                    and isinstance(previous_waveforms.get("full"), list)
                ):
                    merged_waveforms["full"] = previous_waveforms["full"]
                result["waveforms"] = merged_waveforms
                result["waveforms"].pop("replaced", None)
                result["waveforms"].pop("mix", None)
                result["splitEnabled"] = True
                debug = result.get("debug")
                if not isinstance(debug, dict):
                    debug = result["debug"] = {}
                debug["splitterDurationMs"] = split_result.duration_ms
                debug["splitterEngine"] = "bs_roformer"
                debug["splitterModel"] = BS_ROFORMER_MODEL
                deleted_prefix = f"mixed:{song}:" if variant == "mixed" else f"{song}:"
                job.deleted_stems = {
                    key: deleted
                    for key, deleted in job.deleted_stems.items()
                    if not key.startswith(deleted_prefix)
                }
                job.save(application_settings.output_dir)
                wav_ready = True
                # 新分轨文件名带随机后缀，旧分轨、旧试听和已清理的删除记录不再被任何工程引用。
                stale_paths = _referenced_output_paths(
                    [previous_result, previous_deleted_stems], application_settings
                ) - _referenced_output_paths(
                    [job.result, job.deleted_stems, job.deleted_replaced_vocals],
                    application_settings,
                )
                await asyncio.to_thread(_unlink_paths, stale_paths)
                await active_orchestrator.capacity.release()
                capacity_released = True
                job.split_stage = "preview"
                job.split_progress = 75
                job.split_message = "正在导出分轨结果"
                publish()
                total_previews = len(stems)
                finished_previews = 0

                async def encode_preview(stem_path: str) -> str | None:
                    # 按已处理轨道数更新，100 只在结果可用且操作成功后发布。
                    nonlocal finished_previews
                    try:
                        return await make_playback_mp3(
                            application_settings.output_dir / stem_path,
                            application_settings.output_dir,
                        )
                    finally:
                        finished_previews += 1
                        job.split_progress = 75 + round(24 * finished_previews / total_previews)
                        publish()

                try:
                    # 单轨编码失败（返回 None）或异常都只丢掉那一轨的试听：WAV 分轨已经落盘，
                    # 不能因为试听拖垮整个分轨结果。return_exceptions 同时保证没有编码任务会
                    # 在 gather 提前抛出后继续孤独地写 playtrack/。
                    encoded = await asyncio.gather(
                        *(encode_preview(stem) for stem in stems.values()),
                        return_exceptions=True,
                    )
                    errors = [item for item in encoded if isinstance(item, BaseException)]
                    if errors:
                        logger.warning(
                            "stem previews failed job_id=%s song=%s errors=%r",
                            job_id,
                            song,
                            errors,
                        )
                    previews = {
                        name: preview
                        for name, preview in zip(stems, encoded, strict=True)
                        if isinstance(preview, str) and preview
                    }
                    if previews:
                        playback["stems"] = {**recording_previews, **previews}
                        job.save(application_settings.output_dir)
                except asyncio.CancelledError:
                    # PATCH 在 preview 阶段只取消编码器（见 README 状态契约）：WAV 分轨保留，
                    # 任务照常按 succeeded 收尾，不把取消写成终态。
                    if recording_previews:
                        playback["stems"] = dict(recording_previews)
                    else:
                        playback.pop("stems", None)
                    logger.info("stem previews cancelled job_id=%s song=%s", job_id, song)
                _mark_split_succeeded(job)
            except asyncio.CancelledError:
                if wav_ready:
                    _mark_split_succeeded(job)
                else:
                    job.split_status = "cancelled"
                    job.split_stage = "cancelled"
                    job.split_progress = None
                    job.split_message = "音轨分离已取消"
            except Exception:
                logger.exception("history split failed job_id=%s song=%s", job_id, song)
                job.split_status = "failed"
                job.split_stage = "failed"
                job.split_progress = None
                job.split_message = "音轨分离失败"
                job.split_error = "音轨分离失败，请检查服务配置后重试。"
            finally:
                if not wav_ready:
                    result.clear()
                    result.update(previous_result)
                    job.deleted_stems = previous_deleted_stems
                    job.deleted_replaced_vocals = previous_deleted_vocals
                    previous_paths = _referenced_output_paths(previous_result, application_settings)
                    _unlink_paths({path.resolve() for path in generated_paths} - previous_paths)
                if not capacity_released:
                    await active_orchestrator.capacity.release()
                publish()

        job.split_task = asyncio.create_task(execute(), name=f"split-{job_id}-{song}")
        return await _async_job_response(job, request, application_settings)

    def _restore_stashed_mix(job: GenerationJob, result: dict[str, Any]) -> None:
        """替换失败/超时/取消后，把准入时摘掉的成品引用还回去。

        人声引用按既有契约不恢复（判定失效即撤下，见 README），但成品是已经完成的产物、文件
        也没被动过：不还原的话，一次失败的重替换会让用户连上一版成品都听不到，只能干等下一次
        成功的替换。替换成功时成品才真正过期，由成功路径负责作废。
        """
        previous = job.replace_previous
        job.replace_previous = None
        restore_mix_artifact(result, previous)
        try:
            job.save(application_settings.output_dir)
        except Exception:
            logger.exception("failed to persist replacement rollback job_id=%s", job.job_id)

    @application.post(
        "/api/jobs/{job_id}/replace",
        response_model=GenerationJobResponse,
        status_code=status.HTTP_202_ACCEPTED,
        responses={
            200: {"model": GenerationJobResponse},
            404: {"model": ErrorResponse},
            409: {"model": ErrorResponse},
            413: {"model": ErrorResponse},
            429: {"model": ErrorResponse},
        },
    )
    async def replace_generation_vocal(
        job_id: str, request: Request, response: Response, song: int = 0
    ):
        job = retained_job(request, job_id)
        if job is None:
            return JSONResponse(
                status_code=status.HTTP_404_NOT_FOUND,
                content={"success": False, "message": "生成任务不存在。"},
            )
        if job.status != "succeeded":
            return JSONResponse(
                status_code=status.HTTP_409_CONFLICT,
                content={"success": False, "message": "生成任务尚未完成，无法替换人声。"},
            )
        result = _song_result(job, song)
        if result is None:
            return JSONResponse(
                status_code=status.HTTP_404_NOT_FOUND,
                content={"success": False, "message": "歌曲不存在。"},
            )
        # Checked before the cached-replacement response: a mix reads the current vocal
        # file, so a running mix must block even when replacement needs no new inference.
        if mix_busy(job):
            return JSONResponse(
                status_code=status.HTTP_409_CONFLICT,
                content={"success": False, "message": "该任务正在合轨，请稍后重试。"},
            )
        if job.split_status in {"pending", "running"}:
            return JSONResponse(
                status_code=status.HTTP_409_CONFLICT,
                content={
                    "success": False,
                    "message": f"第 {(job.split_song or 0) + 1} 首歌曲正在拆轨。",
                },
            )
        if job.replace_task is not None and not job.replace_task.done():
            if job.replace_status in {"failed", "cancelled"}:
                return JSONResponse(
                    status_code=status.HTTP_409_CONFLICT,
                    content={
                        "success": False,
                        "message": "上一次人声替换仍在安全收尾，请稍后重试。",
                    },
                )
            if job.replace_song == song:
                return await _async_job_response(job, request, application_settings)
            return JSONResponse(
                status_code=status.HTTP_409_CONFLICT,
                content={
                    "success": False,
                    "message": f"第 {(job.replace_song or 0) + 1} 首歌曲正在替换人声。",
                },
            )

        stems = result.get("stems")
        vocal_url = (stems or {}).get("replaced") or next(
            (
                value
                for name, value in (stems.items() if isinstance(stems, dict) else ())
                if is_vocal_stem(name) and isinstance(value, str)
            ),
            None,
        )
        if vocal_url is None:
            return JSONResponse(
                status_code=status.HTTP_409_CONFLICT,
                content={"success": False, "message": "该歌曲没有人声音轨，请先完成拆轨。"},
            )
        try:
            output_dir = job_song_dir(
                application_settings.output_dir, job_id, (result.get("songNumber") or song + 1) - 1
            )
            vocal_path = _output_path_from_url(vocal_url, application_settings)
            if vocal_path.parent != output_dir.resolve() or not vocal_path.is_file():
                raise OSError("missing vocal stem")
        except (OSError, ValueError):
            return JSONResponse(
                status_code=status.HTTP_409_CONFLICT,
                content={"success": False, "message": "人声音轨文件不存在，无法替换。"},
            )
        if vocal_path.stat().st_size > application_settings.rvc_max_upload_bytes:
            return JSONResponse(
                status_code=status.HTTP_413_CONTENT_TOO_LARGE,
                content={"success": False, "message": "人声音轨文件过大，无法替换。"},
            )

        fingerprint = request.app.state.rvc_model_fingerprint
        cached_url = result.get("replacedVocal")
        if isinstance(cached_url, str):
            if result.get("_replacedVocalModel") == fingerprint and _replacement_file_exists(
                cached_url, application_settings, output_dir
            ):
                job.last_operation = "replace"
                job.replace_song = song
                job.replace_status = "succeeded"
                job.replace_stage = "completed"
                job.replace_progress = REPLACE_COMPLETE_PROGRESS
                job.replace_message = "人声替换完成"
                job.replace_error = None
                response.status_code = status.HTTP_200_OK
                return await _async_job_response(job, request, application_settings)

        active_orchestrator = _orchestrator(request)
        job.replace_song = song
        job.replace_status = "pending"
        job.replace_stage = "preparing_vocal"
        job.replace_progress = 0
        job.replace_message = "正在准备输入音频"
        job.replace_error = None
        job.replace_cancel_requested = False
        try:
            await active_orchestrator.capacity.acquire()
        except CapacityExceededError:
            job.replace_song = None
            job.replace_status = None
            job.replace_stage = None
            job.replace_progress = None
            job.replace_message = None
            return JSONResponse(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                content={"success": False, "message": "当前人声替换任务过多，请稍后重试。"},
            )
        except asyncio.CancelledError:
            job.replace_song = None
            job.replace_status = None
            job.replace_stage = None
            job.replace_progress = None
            job.replace_message = None
            raise

        if isinstance(cached_url, str):
            # 缓存失效（模型指纹/文件/路径任一不符）：只把引用从结果里摘掉以触发重新推理，
            # 旧文件与新产物同名：覆盖前保留备份，直到新结果元数据提交成功后再清理。
            # 重跑失败/超时/取消时旧文件仍是"上一版还能听"的唯一退路——留着只是不再被任何
            # 结果引用、也不会对外暴露，同曲最多一份，不会堆积。
            # 先给已完成成品存档，再在内存中作废；成功提交时才落盘，避免进程中断或恢复保存
            # 失败时丢失旧成品引用。推理失败/超时/取消时还原成品；人声引用则按既有
            # 契约在判定失效时撤下、失败也不恢复（见 test_failed_rerun_keeps_previous_...）。
            job.replace_previous = capture_mix_artifact(result) or None
            result.pop("replacedVocal", None)
            result.pop("_replacedVocalModel", None)
            _playback_dict(result).pop("replacedVocal", None)
            if isinstance(result.get("waveforms"), dict):
                result["waveforms"].pop("replaced", None)
            # 判定失效的正是"合轨成品所依据的那份人声"，所以先在内存中摘掉成品引用：
            # 替换**成功**时成品就此过期（不会继续被当成最新）。替换失败/超时/取消时，由
            # _restore_stashed_mix 把这个成品引用还回来——文件没被动过，用户仍应能试听。
            invalidate_mix_artifact(job, result)

        job.last_operation = "replace"

        def publish() -> None:
            _publish_job(job, request, snapshot=True)

        async def execute() -> None:
            try:
                job.replace_status = "running"
                publish()
                # 输入准备作为第一阶段；RVC 是整体调用，不估算内部特征提取/推理百分比。
                try:
                    await asyncio.to_thread(_read_operation_input, vocal_path)
                except (GenerationError, OSError):
                    job.replace_error = "人声输入音频不可读，请检查音轨文件后重试。"
                    raise
                if job.replace_cancel_requested:
                    raise asyncio.CancelledError
                result_path = output_dir / replacement_filename(vocal_path.name)
                # 这里刻意不用 TemporaryDirectory 上下文：它在 await 被取消时会立刻 rmtree，
                # 而 RVC 推理线程无法强停、仍在写这个目录，清理与写文件会互相打架（真实 RVC 还会
                # 在推理中途读取输入/写入输出）。改成手动创建、等线程确认退出后再删。
                temp_dir = Path(tempfile.mkdtemp(prefix=".rvc-", dir=output_dir))
                backup_path = temp_dir / "previous.wav"
                previous_result = copy.deepcopy(result)
                previous_deleted = job.deleted_replaced_vocals.copy()
                installed = False
                committed = False
                try:
                    output_path = temp_dir / "converted.wav"
                    if job.replace_cancel_requested:
                        raise asyncio.CancelledError
                    job.replace_stage = "replacing_vocal"
                    job.replace_progress = 25
                    job.replace_message = "正在转换人声音色"
                    publish()
                    conversion_task = asyncio.create_task(
                        request.app.state.voice_engine.convert(
                            vocal_path, output_path, rms_mix_rate=0.0
                        )
                    )
                    try:
                        await asyncio.wait_for(
                            asyncio.shield(conversion_task),
                            timeout=application_settings.rvc_conversion_timeout_seconds,
                        )
                    except asyncio.CancelledError:
                        # 外层被取消（PATCH 取消任务或进程收尾）时不能直接返回：RVC 推理线程无法
                        # 强停，这里必须等它真正退出，否则临时目录会被提前删除、线程后续写文件失败，
                        # 引擎锁也会在推理仍占用 GPU 时被释放。conversion_task 用 shield 保护，
                        # 所以拿到的异常只反映线程自身的失败，可以安全折叠。
                        _mark_replace_cancelled(job)
                        publish()
                        await _await_stuck_replacement_worker(
                            conversion_task, job_id=job_id, song=song, reason="cancelled"
                        )
                        raise
                    except asyncio.TimeoutError:
                        logger.error(
                            "vocal replacement timed out job_id=%s song=%s timeout_seconds=%s",
                            job_id,
                            song,
                            application_settings.rvc_conversion_timeout_seconds,
                        )
                        job.replace_status = "failed"
                        job.replace_stage = "failed"
                        job.replace_progress = None
                        job.replace_message = "人声替换超时"
                        job.replace_error = "人声替换超时，请重试。"
                        publish()
                        # 失败状态已经对客户端生效，但线程仍在跑：这里继续等它退出，只为了让锁和
                        # 临时目录与线程寿命对齐。等待时长不受超时约束，若线程永不返回，这个替换
                        # 会一直占着并发额度——保持与 README 记载的语义一致，不做强制释放，
                        # 只通过日志和 /api/health 把"卡住的收尾"暴露出来。
                        await _await_stuck_replacement_worker(
                            conversion_task, job_id=job_id, song=song, reason="timed out"
                        )
                        _restore_stashed_mix(job, result)
                        return
                    if job.replace_cancel_requested:
                        raise asyncio.CancelledError
                    job.replace_stage = "creating_replacement"
                    job.replace_progress = 50
                    job.replace_message = "正在创建替换音轨"
                    publish()
                    require_readable_file(output_path, "人声替换输出文件不可读")
                    # 同一文件系统内保留旧文件，直到后处理与元数据提交都成功。
                    if result_path.exists():
                        backup_path.hardlink_to(result_path)
                    output_path.replace(result_path)
                    installed = True
                    replacement_waveforms = {}
                    try:
                        replacement_waveforms = await extract_waveforms({"replaced": result_path})
                    except Exception:
                        logger.warning(
                            "replaced vocal waveform skipped job_id=%s song=%s",
                            job_id,
                            song,
                            exc_info=True,
                        )
                    if job.replace_cancel_requested:
                        raise asyncio.CancelledError
                    job.replace_stage = "exporting_replacement"
                    job.replace_progress = 75
                    job.replace_message = "正在导出替换结果"
                    publish()
                    preview = await make_playback_mp3(result_path, application_settings.output_dir)
                    if job.replace_cancel_requested:
                        raise asyncio.CancelledError
                    job.replace_progress = 99
                    publish()
                    result["replacedVocal"] = result_path.relative_to(
                        application_settings.output_dir.resolve()
                    ).as_posix()
                    _playback_dict(result).pop("replacedVocal", None)
                    if preview:
                        result["playback"]["replacedVocal"] = preview
                    result.setdefault("waveforms", {}).pop("replaced", None)
                    result["waveforms"].update(replacement_waveforms)
                    result["_replacedVocalModel"] = fingerprint
                    job.deleted_replaced_vocals.pop(str(song), None)
                    # 人声内容已经不同，旧成品不再对应当前歌曲：作废引用，需要重新合轨。
                    invalidate_mix_artifact(job, result)
                    job.save(application_settings.output_dir)
                    committed = True
                    job.replace_previous = None
                    job.replace_status = "succeeded"
                    job.replace_stage = "completed"
                    job.replace_progress = REPLACE_COMPLETE_PROGRESS
                    job.replace_message = "人声替换完成"
                except (Exception, asyncio.CancelledError):
                    result.clear()
                    result.update(previous_result)
                    job.deleted_replaced_vocals = previous_deleted
                    if backup_path.exists():
                        backup_path.replace(result_path)
                    elif installed:
                        result_path.unlink(missing_ok=True)
                    raise
                finally:
                    # 推理已退出；若回滚失败则保留备份，避免清理时丢失旧文件。
                    if committed or not backup_path.exists():
                        shutil.rmtree(temp_dir, ignore_errors=True)
            except asyncio.CancelledError:
                job.replace_status = "cancelled"
                job.replace_stage = "cancelled"
                job.replace_progress = None
                job.replace_message = "人声替换已取消"
                _restore_stashed_mix(job, result)
            except Exception:
                logger.exception("vocal replacement failed job_id=%s song=%s", job_id, song)
                job.replace_status = "failed"
                job.replace_stage = "failed"
                job.replace_progress = None
                job.replace_message = "人声替换失败"
                job.replace_error = job.replace_error or "人声替换失败，请检查 RVC 配置后重试。"
                publish()
                _restore_stashed_mix(job, result)
            finally:
                job.replace_cancel_requested = False
                await active_orchestrator.capacity.release()
                publish()

        job.replace_task = asyncio.create_task(execute(), name=f"replace-{job_id}-{song}")
        return await _async_job_response(job, request, application_settings)

    def _mix_input_path(value: str, directory: Path) -> Path:
        """原版 `_output_audio_path` 的规则，逐条保留。

        路径必须在 output 根内、正好落在该歌曲目录、不在 `.trash`、扩展名属于媒体白名单；
        否则 400。原版就是这样信任调用方传进来的音轨路径的，不做"是否等于当前分轨"的比对。
        """
        root = application_settings.output_dir.resolve()
        target = output_path_from_url(value, root)
        relative = target.relative_to(root)
        if (
            target.parent != directory.resolve()
            or ".trash" in relative.parts
            or target.suffix.lower() not in MIX_INPUT_SUFFIXES
        ):
            raise ValueError("Invalid stem URL")
        return target

    def _readable_mix_inputs(paths: list[Path]) -> bool:
        for path in paths:
            try:
                require_readable_file(path, "混音输入文件不可读")
            except (GenerationError, OSError) as exc:
                # Path.is_file() 对 EACCES 会抛 PermissionError，不能只当 GenerationError 处理，
                # 否则一个不可读的音轨会变成 500 而不是统一的 4xx。
                logger.warning("Mix input unreadable: %s (%s)", path, exc)
                return False
        return True

    async def _start_mix(
        job: GenerationJob,
        song: int,
        request: Request,
        *,
        result: dict[str, Any],
        reference: Path,
        inputs: list[Path],
        directory: Path,
        vocal_name: str,
        mix: TrackMixRequest | None = None,
        variant: str = "original",
        song_output: dict[str, Any] | None = None,
    ) -> GenerationJob:
        """抢额度、置状态、起任务；与 /split、/replace 一样只通过 job.mix_* 汇报结果。"""
        active_orchestrator = _orchestrator(request)
        # 与 /split、/replace 同构：先占住任务，再 await 抢额度，避免两条请求同时通过检查。
        job.mix_song = song
        job.mix_status = "pending"
        job.mix_stage = "mixing"
        job.mix_progress = MIX_PROGRESS
        job.mix_message = "正在合轨"
        job.mix_error = None
        job.mix_cancel_requested = False
        try:
            await active_orchestrator.capacity.acquire()
        except CapacityExceededError:
            job.mix_song = None
            job.mix_status = None
            job.mix_stage = None
            job.mix_progress = None
            job.mix_message = None
            raise
        except asyncio.CancelledError:
            job.mix_song = None
            job.mix_status = None
            job.mix_stage = None
            job.mix_progress = None
            job.mix_message = None
            raise

        if mix is not None:
            previous = copy.deepcopy(result)
            try:
                result["mixConfig"] = mix.model_dump()
                job.save(application_settings.output_dir)
            except BaseException:
                result.clear()
                result.update(previous)
                reset_mix_state(job)
                await active_orchestrator.capacity.release()
                raise

        job.last_operation = "mix"

        def publish() -> None:
            _publish_job(job, request, snapshot=True)

        def cleanup_replaced_files() -> None:
            referenced = set()
            songs = [job.result, *(job.result.get("alternatives") or [])]
            for output in [*songs, *filter(None, (song_project(item, "mixed") for item in songs))]:
                urls = [output.get(key) for key in ("fullTrack", "mixedTrack", "replacedVocal")]
                urls.extend(output.get("stems", {}).values())
                playback = output.get("playback") or {}
                urls.extend(
                    playback.get(key) for key in ("fullTrack", "mixedTrack", "replacedVocal")
                )
                urls.extend((playback.get("stems") or {}).values())
                for url in urls:
                    if isinstance(url, str):
                        try:
                            referenced.add(_output_path_from_url(url, application_settings))
                        except ValueError:
                            pass
            # 软删除的落点有两处：第一首候选的分轨进 `.trash/{job}` 根，替换结果与其它候选进
            # `.trash/{job}/song_N`（见 _stem_trash_path / trash_result_path）。只扫 song_N
            # 会让第一首被删除的分轨永远留在回收站里；反过来扫根目录又可能删掉别的候选的
            # 可撤回文件，所以按 directory 推导出属于这一首的那一处。
            trash_root = application_settings.output_dir / ".trash" / job.job_id
            trash_folders = (trash_root,) if directory.name == "song_1" else ()
            # 两首歌共用回收站：另一首仍可撤回的文件（软删除的音轨、替换人声）不能被清掉。
            undoable = {
                Path(str(entry.get(key))).name
                for entry in [*job.deleted_stems.values(), *job.deleted_replaced_vocals.values()]
                if isinstance(entry, dict)
                for key in ("url", "mixTrack")
                if isinstance(entry.get(key), str)
            }
            for folder in (
                directory,
                directory / "playtrack",
                trash_root / directory.name,
                *trash_folders,
            ):
                if folder.is_symlink():
                    continue
                try:
                    for path in folder.glob("*"):
                        if (
                            path.suffix.lower() in MIX_INPUT_SUFFIXES
                            and path.is_file()
                            and path.resolve() not in referenced
                            and not (".trash" in path.parts and path.name in undoable)
                        ):
                            path.unlink()
                except OSError:
                    logger.warning(
                        "old mix files cleanup skipped directory=%s", folder, exc_info=True
                    )

        async def commit_mix() -> None:
            # 替换编辑器把产出写进替换后那首（第一次提交时新建）；普通编辑器覆盖它读的那首。
            # 原曲与替换后那首共用歌曲目录，替换后那首的文件带 `mixed_` 前缀。
            original = song_output if song_output is not None else result
            target_variant = "mixed" if mix.editor == "replace" else variant
            target = (
                result
                if mix.editor != "replace"
                else song_project(original, "mixed")
                or {"durationSeconds": original.get("durationSeconds")}
            )
            prefix = "mixed_" if target_variant == "mixed" else ""
            deleted_prefix = f"mixed:{song}:" if target_variant == "mixed" else f"{song}:"
            # 版本号对这首歌单调递增：替换后那首的旧版本可能高于原曲版本。
            revision = max(target.get("audioRevision", 0), mix.audioRevision) + 1
            token = uuid4().hex[:12]
            result_path = directory / f"{prefix}master_v{revision}_{token}.wav"
            split = mix.tracks[0].source != "full"
            track_paths = (
                [directory / f"{prefix}{track.id}_v{revision}_{token}.wav" for track in mix.tracks]
                if split
                else []
            )
            created = [result_path, *track_paths]
            previous = copy.deepcopy(original)
            previous_deleted_stems = copy.deepcopy(job.deleted_stems)
            previous_deleted_vocals = copy.deepcopy(job.deleted_replaced_vocals)
            previous_replace = {
                key: getattr(job, key)
                for key in (
                    "replace_song",
                    "replace_status",
                    "replace_stage",
                    "replace_progress",
                    "replace_message",
                    "replace_error",
                )
            }
            committed = False
            try:
                with tempfile.TemporaryDirectory(prefix=".mix-", dir=directory) as temp_dir:
                    output_path = Path(temp_dir) / result_path.name
                    staged_tracks = [Path(temp_dir) / path.name for path in track_paths]
                    await mix_tracks(
                        inputs,
                        reference,
                        output_path,
                        application_settings.rvc_mix_timeout_seconds,
                        gains=[10 ** (track.gainDb / 20) for track in mix.tracks],
                        track_outputs=staged_tracks if split else None,
                    )
                    job.mix_stage = "waveform"
                    job.mix_progress = WAVEFORM_PROGRESS
                    job.mix_message = "正在提取合轨波形"
                    publish()
                    fresh_waveforms = await extract_waveforms(
                        {
                            "mix": output_path,
                            **{
                                track.id: path
                                for track, path in zip(mix.tracks, staged_tracks, strict=True)
                            },
                        }
                        if split
                        else {"mix": output_path}
                    )
                    if job.mix_cancel_requested:
                        raise asyncio.CancelledError
                    output_path.replace(result_path)
                    for staged, path in zip(staged_tracks, track_paths, strict=True):
                        staged.replace(path)
                job.mix_stage = "preview"
                job.mix_progress = MIX_PREVIEW_PROGRESS
                job.mix_message = "正在生成试听音频"
                publish()
                previews = []
                for path in [result_path, *track_paths]:
                    preview = await make_playback_mp3(path, application_settings.output_dir)
                    previews.append(preview)
                    if preview:
                        created.append(application_settings.output_dir / preview)
                if job.mix_cancel_requested:
                    raise asyncio.CancelledError
                relative = result_path.relative_to(application_settings.output_dir).as_posix()
                target["fullTrack"] = relative
                target["stems"] = (
                    {
                        track.id: path.relative_to(application_settings.output_dir).as_posix()
                        for track, path in zip(mix.tracks, track_paths, strict=True)
                    }
                    if split
                    else {}
                )
                target["stemUrls"] = list(target["stems"].values())
                target["stemNames"] = {
                    track.id: result["stemNames"][track.stemId]
                    for track in mix.tracks
                    if track.stemId in result.get("stemNames", {}) and track.id in target["stems"]
                }
                target["splitEnabled"] = split
                playback = {}
                if previews[0]:
                    playback["fullTrack"] = previews[0]
                stem_previews = (
                    {
                        track.id: preview
                        for track, preview in zip(mix.tracks, previews[1:], strict=True)
                        if preview
                    }
                    if split
                    else {}
                )
                if stem_previews:
                    playback["stems"] = stem_previews
                target["playback"] = playback
                target["waveforms"] = fresh_waveforms
                if "mix" in fresh_waveforms:
                    target["waveforms"]["full"] = fresh_waveforms["mix"]
                target["audioRevision"] = revision
                target["mixConfig"] = mix.model_dump()
                # 生成记录按它显示这首歌最近一次完成创作的时间。
                target["updatedAt"] = datetime.now(timezone.utc).isoformat()
                target.setdefault("debug", {})
                drop_editor_state(target)
                if target_variant == "mixed":
                    original["mixed"] = target
                job.deleted_stems = {
                    key: value
                    for key, value in job.deleted_stems.items()
                    if not key.startswith(deleted_prefix)
                }
                if variant == "original":
                    # 替换人声是原曲上未提交的产物：替换编辑器把它写进了替换后那首，
                    # 原曲覆盖后它对应的人声已经变了。两种情况都清掉。
                    original.pop("replacedVocal", None)
                    original.pop("_replacedVocalModel", None)
                    _playback_dict(original).pop("replacedVocal", None)
                    if isinstance(original.get("waveforms"), dict):
                        original["waveforms"].pop("replaced", None)
                    job.deleted_replaced_vocals.pop(str(song), None)
                    for key in previous_replace:
                        setattr(job, key, None)
                job.save(application_settings.output_dir)
                committed = True
            finally:
                if not committed:
                    original.clear()
                    original.update(previous)
                    job.deleted_stems = previous_deleted_stems
                    job.deleted_replaced_vocals = previous_deleted_vocals
                    for key, value in previous_replace.items():
                        setattr(job, key, value)
                    for path in created:
                        path.unlink(missing_ok=True)
            # 清理放在落盘之后（异常时根本走不到这里）：进程若恰好死在这两行之间，只会留下
            # 一批无引用的旧文件，等这首下一次成功提交时再删；反过来先删后写盘，一旦写盘失败
            # 回滚回去的元数据就会指向已经被删掉的文件。
            # 清理只是尽力而为：提交已经落盘，清理失败只记日志，不能把合轨报成失败
            # （否则前端会拿着旧版本号退回编辑页，重试只会得到 409）。
            try:
                cleanup_replaced_files()
            except Exception:
                logger.warning(
                    "post-commit cleanup failed job_id=%s song=%s",
                    job.job_id,
                    song,
                    exc_info=True,
                )

        def restore_result(
            previous_track: object, previous_waveforms: object, previous_playback: object
        ) -> None:
            if previous_track is None:
                result.pop("mixedTrack", None)
            else:
                result["mixedTrack"] = previous_track
            if previous_waveforms is None:
                result.pop("waveforms", None)
            else:
                result["waveforms"] = previous_waveforms
            if previous_playback is None:
                (result.get("playback") or {}).pop("mixedTrack", None)
            else:
                _playback_dict(result)["mixedTrack"] = previous_playback

        async def execute() -> None:
            try:
                if job.mix_cancel_requested:
                    raise asyncio.CancelledError
                job.mix_status = "running"
                publish()
                if mix is not None:
                    await commit_mix()
                    job.mix_status = "succeeded"
                    job.mix_stage = "completed"
                    job.mix_progress = 100
                    job.mix_message = "合轨完成"
                    return
                # 原版命名规则：同名成品，成功后原子覆盖，失败时上一版原样保留。
                output_name = mix_output_filename(vocal_name)
                result_path = directory / output_name
                with tempfile.TemporaryDirectory(prefix=".mix-", dir=directory) as temp_dir:
                    output_path = Path(temp_dir) / "mix.wav"
                    await mix_tracks(
                        inputs,
                        reference,
                        output_path,
                        application_settings.rvc_mix_timeout_seconds,
                    )
                    job.mix_stage = "waveform"
                    job.mix_progress = WAVEFORM_PROGRESS
                    job.mix_message = "正在提取合轨波形"
                    publish()
                    fresh_waveforms = await extract_waveforms({"mix": output_path})
                    if job.mix_cancel_requested:
                        raise asyncio.CancelledError
                    previous_track = result.get("mixedTrack")
                    previous_waveforms = result.get("waveforms")
                    previous_playback = (result.get("playback") or {}).get("mixedTrack")
                    updated = (
                        dict(previous_waveforms) if isinstance(previous_waveforms, dict) else {}
                    )
                    updated.pop("mix", None)
                    updated.update(fresh_waveforms)
                    result["mixedTrack"] = result_path.relative_to(
                        application_settings.output_dir
                    ).as_posix()
                    result["waveforms"] = updated
                    # 先落元数据再发布文件：同名覆盖时路径不变，写盘失败就还没碰过成品，
                    # 上一版仍然是磁盘上的那一份。
                    try:
                        job.save(application_settings.output_dir)
                    except Exception:
                        restore_result(previous_track, previous_waveforms, previous_playback)
                        raise
                    try:
                        output_path.replace(result_path)
                    except Exception:
                        # 文件没换成，元数据要退回上一版（波形车道只对旧文件成立）。
                        restore_result(previous_track, previous_waveforms, previous_playback)
                        try:
                            job.save(application_settings.output_dir)
                        except Exception:
                            logger.exception(
                                "failed to roll back mix metadata job_id=%s", job.job_id
                            )
                        raise
                _playback_dict(result).pop("mixedTrack", None)
                job.mix_stage = "preview"
                job.mix_progress = MIX_PREVIEW_PROGRESS
                job.mix_message = "正在生成试听音频"
                publish()
                try:
                    preview = await make_playback_mp3(result_path, application_settings.output_dir)
                    if preview:
                        result["playback"]["mixedTrack"] = preview
                    job.save(application_settings.output_dir)
                except (Exception, asyncio.CancelledError):
                    # WAV 已发布；PATCH 此时只取消试听编码，不改写合轨终态。
                    result["playback"].pop("mixedTrack", None)
                    logger.warning("mix preview skipped job_id=%s", job.job_id, exc_info=True)
                # 旧版导出把成品挂在 mixedTrack 上：落成独立的替换后那首。
                if normalize_mixed_song(result):
                    job.save(application_settings.output_dir)
                job.mix_status = "succeeded"
                job.mix_stage = "completed"
                job.mix_progress = 100
                job.mix_message = "合轨完成"
                if not fresh_waveforms:
                    # 波形只是编辑器的绘制数据，提取失败不影响已经落盘的成品。
                    logger.warning(
                        "mix waveform extraction failed job_id=%s song=%s file=%s",
                        job.job_id,
                        song,
                        result_path.name,
                    )
            except asyncio.CancelledError:
                job.mix_status = job.mix_stage = "cancelled"
                job.mix_progress = None
                job.mix_message = "合轨已取消"
                job.mix_error = None
            except RVCConversionError:
                # 原版语义：混音失败（含超时）按 500 报，细节只进日志。引擎抛出的超时文案用的是
                # "剩余预算"（原版实现如此），所以这里把配置值一起记下来，排障时不会被小数字误导。
                logger.exception(
                    "Audio mixing failed job_id=%s song=%s timeout_seconds=%s",
                    job.job_id,
                    song,
                    application_settings.rvc_mix_timeout_seconds,
                )
                job.mix_status = job.mix_stage = "failed"
                job.mix_progress = None
                job.mix_message = "合轨失败"
                job.mix_error = "合轨失败，请检查音轨后重试。"
            except GenerationError as exc:
                logger.error("Audio processing is unavailable: %s", exc)
                job.mix_status = job.mix_stage = "failed"
                job.mix_progress = None
                job.mix_message = "音频处理工具不可用"
                job.mix_error = "音频处理工具不可用，请检查 FFmpeg 配置。"
            except Exception:
                logger.exception("mix failed job_id=%s song=%s", job.job_id, song)
                job.mix_status = job.mix_stage = "failed"
                job.mix_progress = None
                job.mix_message = "合轨失败"
                job.mix_error = "合轨失败，请检查音轨后重试。"
            finally:
                # 临时产物由 TemporaryDirectory 负责清理；同名成品要么已经原子替换，
                # 要么根本没被碰过，这里只剩取消标记与额度。
                job.mix_cancel_requested = False
                await active_orchestrator.capacity.release()
                publish()

        job.mix_task = asyncio.create_task(execute(), name=f"mix-{job.job_id}-{song}")
        publish()
        return job

    def _mix_conflict(job: GenerationJob) -> JSONResponse | None:
        if (
            mix_busy(job)
            or job.split_status in {"pending", "running"}
            or (job.split_task is not None and not job.split_task.done())
            or (job.replace_task is not None and not job.replace_task.done())
        ):
            return JSONResponse(
                status_code=status.HTTP_409_CONFLICT,
                content={"success": False, "message": "该任务仍有音频处理正在运行，请稍后重试。"},
            )
        return None

    @application.post(
        "/api/jobs/{job_id}/recordings",
        response_model=RecordingUploadResponse,
        status_code=status.HTTP_201_CREATED,
        responses={
            400: {"model": ErrorResponse},
            404: {"model": ErrorResponse},
            409: {"model": ErrorResponse},
            413: {"model": ErrorResponse},
            429: {"model": ErrorResponse},
            500: {"model": ErrorResponse},
        },
    )
    async def upload_recording(
        job_id: str,
        request: Request,
        name: Annotated[str, Form(min_length=1, max_length=100)],
        file: Annotated[UploadFile, File()],
        song: int = 0,
        variant: Literal["original", "mixed"] = "original",
    ):
        def error(code: int, message: str) -> JSONResponse:
            return JSONResponse(status_code=code, content={"success": False, "message": message})

        def current_project():
            job = retained_job(request, job_id)
            if job is None:
                return error(404, "生成任务不存在。")
            if job.status != "succeeded":
                return error(409, "生成任务尚未完成，无法上传录音。")
            output = _song_result(job, song)
            result = song_project(output, variant)
            if result is None:
                return error(404, "歌曲不存在。")
            conflict = _mix_conflict(job)
            if conflict is not None:
                return conflict
            if not result.get("splitEnabled"):
                return error(409, "请先完成分轨，再上传录音。")
            try:
                require_audio_revision(result, request.headers.get("X-Audio-Revision"))
            except HTTPException as exc:
                return error(exc.status_code, exc.detail)
            if len(result.get("stems", {})) >= 32:
                return error(409, "每首歌曲最多保留 32 条音轨。")
            return job, output, result

        capacity = None
        registered = False
        created: list[Path] = []
        try:
            name = name.strip()
            # Cc 控制字符与 Cf 格式字符（双向覆盖、零宽空格）都会让显示名伪装成别的内容；
            # ZWNJ/ZWJ 是波斯语、印度系文字与组合 emoji 的正常组成部分，予以放行。
            if not name or any(
                unicodedata.category(char) in {"Cc", "Cf"} and char not in "\u200c\u200d"
                for char in name
            ):
                return error(400, "录音名称须为 1–100 个字符，不能包含控制字符或不可见格式字符。")
            selected = current_project()
            if isinstance(selected, JSONResponse):
                return selected
            job, output, result = selected
            revision = result.get("audioRevision", 0)
            directory = job_song_dir(
                application_settings.output_dir, job_id, (output.get("songNumber") or song + 1) - 1
            )
            upload_capacity = _orchestrator(request).capacity
            try:
                await upload_capacity.acquire()
            except CapacityExceededError:
                return error(429, "当前音频处理任务过多，请稍后重试。")
            capacity = upload_capacity
            directory.mkdir(parents=True, exist_ok=True)
            stem_id = f"recording_{uuid4().hex}"
            prefix = "mixed_" if variant == "mixed" else ""
            with tempfile.TemporaryDirectory(prefix=".recording-", dir=directory) as staging:
                raw_path = Path(staging) / "input"
                prepared = Path(staging) / f"{prefix}{stem_id}.wav"
                written = 0
                with raw_path.open("wb") as raw:
                    while chunk := await file.read(1024 * 1024):
                        written += len(chunk)
                        if written > application_settings.recording_max_upload_mb * 1024 * 1024:
                            return error(
                                413,
                                "录音文件不能超过 "
                                f"{application_settings.recording_max_upload_mb} MB。",
                            )
                        raw.write(chunk)
                if not written:
                    return error(400, "上传的文件为空。")
                try:
                    duration = await prepare_recording_audio(
                        raw_path, prepared, application_settings.max_duration_minutes * 60
                    )
                except ReferenceInputError as exc:
                    return error(400, str(exc))
                waveforms = await extract_waveforms({stem_id: prepared})
                preview = await make_playback_mp3(prepared, application_settings.output_dir)
                # 解码期间工程可能被覆盖、删除或开始其它操作；发布前再次验证。
                selected = current_project()
                if isinstance(selected, JSONResponse):
                    return selected
                current_job, _, current_result = selected
                if (
                    current_job is not job
                    or current_result is not result
                    or (result.get("audioRevision", 0) != revision)
                ):
                    return error(409, "歌曲版本已更新，请刷新后重试。")
                target = directory / prepared.name
                prepared.replace(target)
                created.append(target)
                preview_path = None
                if preview:
                    staged_preview = application_settings.output_dir / preview
                    preview_path = directory / "playtrack" / staged_preview.name
                    preview_path.parent.mkdir(exist_ok=True)
                    staged_preview.replace(preview_path)
                    created.append(preview_path)
                previous = copy.deepcopy(result)
                try:
                    relative = target.relative_to(application_settings.output_dir).as_posix()
                    result.setdefault("stems", {})[stem_id] = relative
                    result.setdefault("stemNames", {})[stem_id] = name
                    result["stemUrls"] = list(result["stems"].values())
                    result.setdefault("waveforms", {}).update(waveforms)
                    if preview_path:
                        _playback_dict(result).setdefault("stems", {})[stem_id] = (
                            preview_path.relative_to(application_settings.output_dir).as_posix()
                        )
                    result["audioRevision"] = revision + 1
                    job.save(application_settings.output_dir)
                except BaseException:
                    result.clear()
                    result.update(previous)
                    raise
                registered = True
                _publish_job(job, request, snapshot=True)
                base_url = _public_base_url(request, application_settings)
                return {
                    "id": stem_id,
                    "name": name,
                    "audioUrl": build_public_audio_url(base_url, relative),
                    "playbackUrl": build_public_audio_url(
                        base_url, preview_path.relative_to(application_settings.output_dir)
                    )
                    if preview_path
                    else None,
                    "durationSeconds": duration,
                    "waveform": waveforms.get(stem_id, []),
                    "audioRevision": revision + 1,
                }
        except (GenerationError, OSError, ValueError) as exc:
            logger.exception("recording upload failed job_id=%s", job_id)
            return error(500, public_error_reason(exc))
        finally:
            if not registered:
                for path in created:
                    path.unlink(missing_ok=True)
            await file.close()
            if capacity is not None:
                await capacity.release()

    @application.post(
        "/api/jobs/{job_id}/mix",
        response_model=GenerationJobResponse,
        status_code=status.HTTP_202_ACCEPTED,
        responses={
            400: {"model": ErrorResponse},
            404: {"model": ErrorResponse},
            409: {"model": ErrorResponse},
            429: {"model": ErrorResponse},
        },
    )
    async def mix_generation_job(
        job_id: str,
        request: Request,
        song: int = 0,
        variant: Literal["original", "mixed"] = "original",
    ):
        """任务级合轨：与 /split、/replace 同构，进度走任务状态与 SSE。

        `variant` 选输入来自哪首歌：原曲或替换后那首。替换编辑器（editor=replace）读原曲的
        音轨与替换人声，产出写进替换后那首；普通编辑器读哪首就覆盖哪首。
        """
        job = retained_job(request, job_id)
        if job is None:
            return JSONResponse(
                status_code=status.HTTP_404_NOT_FOUND,
                content={"success": False, "message": "生成任务不存在。"},
            )
        if job.status != "succeeded":
            return JSONResponse(
                status_code=status.HTTP_409_CONFLICT,
                content={"success": False, "message": "生成任务尚未完成，无法合轨。"},
            )
        result = _song_result(job, song)
        if result is None:
            return JSONResponse(
                status_code=status.HTTP_404_NOT_FOUND,
                content={"success": False, "message": "歌曲不存在。"},
            )
        song_output = result
        result = song_project(song_output, variant)
        if result is None:
            return JSONResponse(
                status_code=status.HTTP_404_NOT_FOUND,
                content={"success": False, "message": "替换后的歌曲不存在。"},
            )
        mix = None
        body = await request.body()
        if body:
            try:
                mix = TrackMixRequest.model_validate_json(body)
            except ValidationError:
                return JSONResponse(
                    status_code=400,
                    content={"success": False, "message": "合轨设置无效，请检查音轨和音量。"},
                )
            if mix.editor == "replace" and variant != "original":
                return JSONResponse(
                    status_code=400,
                    content={"success": False, "message": "替换人声只能基于原曲提交。"},
                )
            try:
                require_audio_revision(result, mix.audioRevision)
            except HTTPException as exc:
                return JSONResponse(
                    status_code=exc.status_code,
                    content={"success": False, "message": exc.detail},
                )
        elif variant != "original" or result.get("audioRevision", 0) > 0:
            return JSONResponse(
                status_code=409,
                content={"success": False, "message": "请提交当前工程的音轨设置和版本号。"},
            )
        conflict = _mix_conflict(job)
        if conflict is not None:
            return conflict
        try:
            directory = job_song_dir(
                application_settings.output_dir,
                job_id,
                (song_output.get("songNumber") or song + 1) - 1,
            )
        except ValueError:
            return JSONResponse(
                status_code=status.HTTP_400_BAD_REQUEST,
                content={"success": False, "message": "合轨音轨路径无效。"},
            )
        if (
            mix is not None
            and any(track.source == "full" for track in mix.tracks)
            and result.get("splitEnabled")
        ):
            return JSONResponse(
                status_code=400,
                content={
                    "success": False,
                    "message": "完整音轨只适用于未分轨的单轨歌曲。",
                },
            )
        full_track = result.get("fullTrack")
        # 替换人声只属于原曲：替换后那首里的 `replaced` 已经是普通音轨（source=stem）。
        vocal_url = result.get("replacedVocal") if variant == "original" else None
        stems = result.get("stems")
        # 旧无请求体导出仍要求原四轨伴奏，同时保留六轨模型及录音产生的额外音轨。
        backing_names = ["drums", "bass", "other"]
        if isinstance(stems, dict):
            backing_names.extend(name for name in stems if name not in {*backing_names, "vocal"})
        stem_urls = [stems.get(name) if isinstance(stems, dict) else None for name in backing_names]
        selected_urls = [vocal_url, *stem_urls]
        if mix is not None:
            selected_urls = [
                (stems if isinstance(stems, dict) else {}).get(track.stemId)
                if track.source == "stem"
                else vocal_url
                if track.source == "replaced"
                else full_track
                for track in mix.tracks
            ]
        if not isinstance(full_track, str) or not all(
            isinstance(value, str) for value in selected_urls
        ):
            return JSONResponse(
                status_code=status.HTTP_409_CONFLICT,
                content={
                    "success": False,
                    "message": "该歌曲缺少所选工程输入，无法合轨。",
                },
            )
        recorded_model = song_output.get("_replacedVocalModel")
        uses_replacement = mix is None or any(track.source == "replaced" for track in mix.tracks)
        if (
            uses_replacement
            and isinstance(recorded_model, str)
            and (recorded_model != request.app.state.rvc_model_fingerprint)
        ):
            if not _rvc_assets_present(application_settings):
                # 资产不在时指纹无法验证：这台实例根本读不到模型，把它当成"模型已变更"会让
                # 本来只用 ffmpeg 的合轨失败，还给出一个必然失败的补救动作（重跑 /replace）。
                logger.warning(
                    "mix allowed with unverifiable replacement model job_id=%s song=%s: "
                    "RVC assets are missing",
                    job_id,
                    song,
                )
            else:
                # 与 /replace 同一判据：指纹不符说明这份替换人声已经过期，
                # 拿它合轨会产出"看起来正常、其实是旧模型"的成品。
                return JSONResponse(
                    status_code=status.HTTP_409_CONFLICT,
                    content={
                        "success": False,
                        "message": "替换人声使用的模型已变更，请先重新替换人声。",
                    },
                )
        try:
            reference = _mix_input_path(full_track, directory)
            inputs = [_mix_input_path(value, directory) for value in selected_urls]
        except ValueError:
            return JSONResponse(
                status_code=status.HTTP_400_BAD_REQUEST,
                content={"success": False, "message": "合轨音轨路径无效。"},
            )
        if not _readable_mix_inputs([reference, *inputs]):
            return JSONResponse(
                status_code=status.HTTP_404_NOT_FOUND,
                content={"success": False, "message": "混音输入文件不可读。"},
            )
        try:
            job = await _start_mix(
                job,
                song,
                request,
                result=result,
                reference=reference,
                inputs=inputs,
                directory=directory,
                vocal_name=inputs[0].name,
                mix=mix,
                variant=variant,
                song_output=song_output,
            )
        except CapacityExceededError:
            return JSONResponse(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                content={"success": False, "message": "当前音频处理任务过多，请稍后重试。"},
            )
        except Exception:
            logger.exception("failed to save mix draft job_id=%s song=%s", job_id, song)
            return JSONResponse(
                status_code=500,
                content={"success": False, "message": "保存合轨设置失败，请重试。"},
            )
        return await _async_job_response(job, request, application_settings)

    @application.patch(
        "/api/jobs/{job_id}",
        response_model=GenerationJobResponse,
        responses={404: {"model": ErrorResponse}},
    )
    async def update_generation_job(
        job_id: str,
        payload: UpdateGenerationJobRequest,
        request: Request,
    ):
        job = retained_job(request, job_id, allow_active=True)
        if job is None:
            return JSONResponse(
                status_code=status.HTTP_404_NOT_FOUND,
                content={"success": False, "message": "生成任务不存在。"},
            )
        if job.split_task is not None and not job.split_task.done():
            job.split_task.cancel()
            # 取消只有在 WAV 落盘之前（模型分离、波形提取）才是终态：那时没有任何分轨
            # 产物对外可见。preview 阶段的取消只作用于编码器——WAV 分轨与波形都已落盘，任务
            # 本身会照常按 succeeded 收尾（见 README 状态契约）。此处改写 split_status 会让
            # PATCH 的响应与随后的最终状态互相矛盾：客户端把 "cancelled" 当终态就再也看不到
            # 已经生成的 WAV 分轨。收尾（stage=completed）期间同样不改写，避免同样的矛盾。
            if job.split_stage in {"starting_split", "splitting", "waveform"}:
                job.split_status = "cancelled"
                job.split_stage = "cancelled"
                job.split_progress = None
                job.split_message = "音轨分离已取消"
                _publish_job(job, request, snapshot=True)
        if (
            job.mix_status in {"pending", "running"}
            and job.mix_task is not None
            and not job.mix_task.done()
        ):
            job.mix_task.cancel()
            # preview 时 WAV 已发布，只中止编码器；之前的阶段才把取消作为终态。
            if job.mix_stage in {"mixing", "waveform"}:
                job.mix_cancel_requested = True
                job.mix_status = job.mix_stage = "cancelled"
                job.mix_progress = None
                job.mix_message = "合轨已取消"
                job.mix_error = None
            _publish_job(job, request, snapshot=True)
        if (
            job.replace_status in {"pending", "running"}
            and job.replace_task is not None
            and not job.replace_task.done()
        ):
            # ponytail: RVC runs in a worker thread and has no safe stop API; mark cancellation
            # now, keep capacity reserved, then discard its output when inference returns.
            _mark_replace_cancelled(job)
            _publish_job(job, request, snapshot=True)
        if job.task is not None and not job.task.done():
            job.task.cancel()
            job.status = "cancelled"
            job.stage = "cancelled"
            job.message = "任务已取消"
            job.progress = job.step = job.total_steps = None
            job.received_audio_seconds = job.expected_audio_seconds = None
            job.finish_generation()
            job.save(application_settings.output_dir)
            _publish_job(job, request)
        return await _async_job_response(job, request, application_settings)

    @application.delete(
        "/api/jobs/{job_id}/stems/{stem_name}",
        status_code=status.HTTP_204_NO_CONTENT,
        responses={404: {"model": ErrorResponse}, 409: {"model": ErrorResponse}},
    )
    async def delete_generation_stem(
        job_id: str,
        stem_name: str,
        request: Request,
        song: int = 0,
        variant: Literal["original", "mixed"] = "original",
    ):
        job = retained_job(request, job_id)
        if job is None:
            return JSONResponse(
                status_code=status.HTTP_404_NOT_FOUND,
                content={"success": False, "message": "生成任务不存在。"},
            )
        song_output = _song_result(job, song)
        result = song_project(song_output, variant)
        if job.status != "succeeded" or result is None:
            return JSONResponse(
                status_code=status.HTTP_409_CONFLICT,
                content={"success": False, "message": "生成任务尚未完成，无法删除音轨。"},
            )
        if not result.get("splitEnabled"):
            return JSONResponse(
                status_code=status.HTTP_409_CONFLICT,
                content={"success": False, "message": "完整混音不能作为分轨删除。"},
            )
        if mix_busy(job):
            return JSONResponse(
                status_code=status.HTTP_409_CONFLICT,
                content={"success": False, "message": "该任务正在合轨，请稍后重试。"},
            )
        if job.split_status in {"pending", "running"}:
            # 分轨期间（含 preview 编码窗口）result["stems"] 已经对外可见，但文件仍在被编码器
            # 读写：此时删除会丢播放引用、恢复会永久降级成 WAV，一律先拒绝。
            return JSONResponse(
                status_code=status.HTTP_409_CONFLICT,
                content={"success": False, "message": "该歌曲正在拆轨，请稍后重试。"},
            )

        try:
            require_audio_revision(
                result,
                request.headers.get("X-Audio-Revision"),
            )
        except HTTPException as exc:
            return JSONResponse(
                status_code=exc.status_code,
                content={"success": False, "message": exc.detail},
            )
        stems = result.get("stems", {})
        stem_url = stems.get(stem_name)
        if not stem_url:
            return JSONResponse(
                status_code=status.HTTP_404_NOT_FOUND,
                content={"success": False, "message": "音轨不存在或已被删除。"},
            )
        is_vocal = is_vocal_stem(stem_name)
        try:
            target = _output_path_from_url(stem_url, application_settings)
            if not target.is_file():
                raise OSError("stem file is missing")
            directory_song = (song_output.get("songNumber") or song + 1) - 1
            trash = _stem_trash_path(target, application_settings, job_id, directory_song)
            trash.parent.mkdir(parents=True, exist_ok=True)
            trash.unlink(missing_ok=True)
            target.replace(trash)
        except (OSError, ValueError):
            logger.exception("failed to delete stem job_id=%s stem=%s", job_id, stem_name)
            return JSONResponse(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                content={"success": False, "message": "删除音轨文件失败。"},
            )
        stem_index = list(stems).index(stem_name)
        waveforms = result.get("waveforms")
        # 两首歌各记各的撤回：替换后那首的键带 `mixed:` 前缀。
        deleted_key = f"mixed:{song}:{stem_name}" if variant == "mixed" else f"{song}:{stem_name}"
        deleted_entry: dict[str, Any] = {
            "url": stem_url,
            "index": stem_index,
            "waveform": waveforms.get(stem_name) if isinstance(waveforms, dict) else None,
        }
        if stem_name in result.get("stemNames", {}):
            deleted_entry["name"] = result["stemNames"].pop(stem_name)
        playback = result.get("playback")
        stem_playback = playback.get("stems") if isinstance(playback, dict) else None
        if isinstance(stem_playback, dict):
            deleted_entry["playback"] = stem_playback.get(stem_name)
        # 分轨是合轨的输入：删掉它就等于让成品不再对应当前歌曲。引用连同车道一起作废，
        # 但先存进撤回记录，PUT 恢复分轨时能把成品一并还原（与其它撤回字段同一机制）。
        deleted_entry.update(capture_mix_artifact(result))
        job.deleted_stems[deleted_key] = deleted_entry
        del stems[stem_name]
        if isinstance(stem_playback, dict):
            stem_playback.pop(stem_name, None)
        result["stemUrls"] = list(stems.values())
        if isinstance(waveforms, dict):
            waveforms.pop(stem_name, None)
        invalidate_mix_artifact(job, result)
        if is_vocal and variant == "original":
            if (
                job.replace_song == song
                and job.replace_status in {"pending", "running"}
                and job.replace_task is not None
                and not job.replace_task.done()
            ):
                _mark_replace_cancelled(job)
                _publish_job(job, request, snapshot=True)
        job.message = f"音轨 {stem_name} 已删除"
        job.save(application_settings.output_dir)
        return Response(status_code=status.HTTP_204_NO_CONTENT)

    @application.put(
        "/api/jobs/{job_id}/stems/{stem_name}",
        response_model=GenerationJobResponse,
        responses={404: {"model": ErrorResponse}, 409: {"model": ErrorResponse}},
    )
    async def restore_generation_stem(
        job_id: str,
        stem_name: str,
        request: Request,
        song: int = 0,
        variant: Literal["original", "mixed"] = "original",
    ):
        job = retained_job(request, job_id)
        if job is None:
            return JSONResponse(
                status_code=status.HTTP_404_NOT_FOUND,
                content={"success": False, "message": "生成任务不存在。"},
            )
        song_output = _song_result(job, song)
        result = song_project(song_output, variant)
        if job.status != "succeeded" or result is None:
            return JSONResponse(
                status_code=status.HTTP_409_CONFLICT,
                content={"success": False, "message": "生成任务尚未完成，无法恢复音轨。"},
            )
        stems = result.get("stems", {})
        if mix_busy(job):
            return JSONResponse(
                status_code=status.HTTP_409_CONFLICT,
                content={"success": False, "message": "该任务正在合轨，请稍后重试。"},
            )
        if job.split_status in {"pending", "running"}:
            # 与删除分轨同一理由：编码器还在写 playtrack/，恢复出来的引用会指向被顶替的文件。
            return JSONResponse(
                status_code=status.HTTP_409_CONFLICT,
                content={"success": False, "message": "该歌曲正在拆轨，请稍后重试。"},
            )
        try:
            require_audio_revision(
                result,
                request.headers.get("X-Audio-Revision"),
            )
        except HTTPException as exc:
            return JSONResponse(
                status_code=exc.status_code,
                content={"success": False, "message": exc.detail},
            )
        if stem_name in stems:
            return await _async_job_response(job, request, application_settings)
        # 两首歌各记各的撤回：替换后那首的键带 `mixed:` 前缀。
        deleted_key = f"mixed:{song}:{stem_name}" if variant == "mixed" else f"{song}:{stem_name}"
        deleted = job.deleted_stems.get(deleted_key)
        if deleted is None:
            return JSONResponse(
                status_code=status.HTTP_404_NOT_FOUND,
                content={"success": False, "message": "没有可恢复的音轨。"},
            )

        try:
            target = _output_path_from_url(deleted["url"], application_settings)
            directory_song = (song_output.get("songNumber") or song + 1) - 1
            trash = _stem_trash_path(target, application_settings, job_id, directory_song)
            if not trash.is_file():
                del job.deleted_stems[deleted_key]
                return JSONResponse(
                    status_code=status.HTTP_404_NOT_FOUND,
                    content={"success": False, "message": "删除的音轨文件已不存在，无法恢复。"},
                )
            target.parent.mkdir(parents=True, exist_ok=True)
            trash.replace(target)
        except (OSError, ValueError):
            logger.exception("failed to restore stem job_id=%s stem=%s", job_id, stem_name)
            return JSONResponse(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                content={"success": False, "message": "恢复音轨文件失败。"},
            )

        items = list(stems.items())
        items.insert(min(deleted["index"], len(items)), (stem_name, deleted["url"]))
        result["stems"] = dict(items)
        if isinstance(deleted.get("name"), str):
            result.setdefault("stemNames", {})[stem_name] = deleted["name"]
        if isinstance(deleted.get("playback"), str):
            _playback_dict(result).setdefault("stems", {})[stem_name] = deleted["playback"]
        result["stemUrls"] = list(result["stems"].values())
        waveforms = result.get("waveforms")
        if isinstance(waveforms, dict) and deleted["waveform"] is not None:
            waveforms[stem_name] = deleted["waveform"]
        # 这条入口只恢复本车道。替换人声（`/api/voice/result` DELETE 写下的记录）由它自己的
        # PUT 负责：拼在一起会让"撤回一条普通音轨"顺手复活刚删掉的替换结果，
        # 而删除侧早已不再联动（文档第 10 条只要求单向不连带删除）。
        if stored_path_exists(application_settings.output_dir, deleted.get("mixTrack")):
            # 删除分轨时作废的成品：文件还在（同名覆盖只会由下一次合轨写），恢复引用与车道。
            # 用 helper 而不是裸写：它自带"已有更新的成品就不覆盖"的防护，并会一并写回 mix
            # 车道（裸写只判断暂存键，会拿过期车道覆盖掉更新的成品）。
            restore_mix_artifact(result, deleted)
        del job.deleted_stems[deleted_key]
        job.message = f"音轨 {stem_name} 已恢复"
        job.save(application_settings.output_dir)
        return await _async_job_response(job, request, application_settings)

    @application.get("/output/{file_path:path}")
    async def audio_file(file_path: str, request: Request) -> StreamingResponse:
        root = application_settings.output_dir.resolve()
        try:
            target = (root / file_path).resolve()
        except OSError as exc:
            raise _audio_io_error(exc, root / file_path) from exc
        except RuntimeError as exc:
            raise HTTPException(status_code=404, detail="Audio file not found") from exc
        try:
            relative = target.relative_to(root)
        except ValueError as exc:
            raise HTTPException(status_code=404, detail="Audio file not found") from exc
        # 隐藏目录一律不对外：.trash 是软删除区，.mix-* 是合轨的中间产物。
        if (
            any(part.startswith(".") for part in relative.parts)
            or relative.parts[:1] in {("rvc",), ("logs",)}
            # Only the reference preview clips are public, not the index.
            or (relative.parts[:1] == ("references",) and relative.suffix != ".mp3")
        ):
            raise HTTPException(status_code=404, detail="Audio file not found")
        expiry = None
        if len(relative.parts) >= 2 and relative.parts[0] == "jobs":
            job = request.app.state.jobs.get(relative.parts[1])
            if retention_enforced(application_settings):
                if job is None:
                    raise HTTPException(status_code=404, detail="任务元数据不可用。")
                expiry = expires_at(job, application_settings.song_retention_days)
                if expiry is None:
                    raise HTTPException(status_code=404, detail="任务创建时间不可用。")
                if is_expired(job, application_settings.song_retention_days):
                    raise HTTPException(status_code=404, detail="歌曲已过期。")
        preview_match = (
            re.fullmatch(r"(.+)\.playback-\d+-\d+\.mp3", target.name)
            if target.parent.name == "playtrack"
            else None
        )
        if preview_match:
            source_path = target.parent.parent
            try:
                for source_path in target.parent.parent.iterdir():
                    if source_path.stem != preview_match[1] or source_path.suffix.lower() != ".wav":
                        continue
                    source_path = source_path.resolve()
                    source_path.relative_to(root)
                    if playback_matches(source_path, target):
                        break
                else:
                    raise HTTPException(status_code=404, detail="Audio file not found")
            except (ValueError, RuntimeError) as exc:
                raise HTTPException(status_code=404, detail="Audio file not found") from exc
            except OSError as exc:
                raise _audio_io_error(exc, source_path) from exc
        try:
            descriptor = os.open(target, os.O_RDONLY | os.O_NONBLOCK)
            try:
                metadata = os.fstat(descriptor)
                if not stat.S_ISREG(metadata.st_mode):
                    raise HTTPException(status_code=404, detail="Audio file not found")
                os.set_blocking(descriptor, True)
                source = os.fdopen(descriptor, "rb")
            except BaseException:
                os.close(descriptor)
                raise
        except OSError as exc:
            raise _audio_io_error(exc, target) from exc
        try:
            file_size = metadata.st_size
            start, end = _parse_byte_range(request.headers.get("range"), file_size)
            media_type = detect_audio_content_type(target, header=source.read(12))
        except OSError as exc:
            source.close()
            raise _audio_io_error(exc, target) from exc
        except BaseException:
            source.close()
            raise
        response_status = status.HTTP_206_PARTIAL_CONTENT if request.headers.get("range") else 200
        cache_seconds = (
            max(0, int((expiry - datetime.now(timezone.utc)).total_seconds())) if expiry else 0
        )
        headers = {
            "Accept-Ranges": "bytes",
            "Cache-Control": (
                f"public, max-age={cache_seconds if expiry else 31536000}, immutable"
                if len(relative.parts) >= 2
                and relative.parts[-2] == "playtrack"
                and re.fullmatch(r".+\.playback-\d+-\d+\.mp3", target.name)
                and (expiry is not None or not retention_enforced(application_settings))
                else "no-store"
            ),
            "Content-Length": str(end - start + 1),
        }
        if response_status == status.HTTP_206_PARTIAL_CONTENT:
            headers["Content-Range"] = f"bytes {start}-{end}/{file_size}"
        return StreamingResponse(
            _stream_file(source, start, end),
            background=BackgroundTask(source.close),
            status_code=response_status,
            media_type=media_type,
            headers=headers,
        )

    return application


def _orchestrator(request: Request) -> GenerationOrchestrator:
    active = getattr(request.app.state, "orchestrator", None)
    if active is None:
        raise HTTPException(status_code=503, detail="Application is not ready")
    return active


def _audio_io_error(exc: OSError, path: Path) -> HTTPException:
    if exc.errno in {errno.ENOENT, errno.ENOTDIR, errno.EISDIR, errno.ELOOP, errno.ENXIO}:
        return HTTPException(status_code=404, detail="Audio file not found")
    if isinstance(exc, PermissionError) or exc.errno in {errno.EACCES, errno.EPERM}:
        return HTTPException(status_code=403, detail="Audio file access denied")
    logger.error("Audio file IO failure path=%s errno=%s", path, exc.errno, exc_info=True)
    if exc.errno in {errno.EMFILE, errno.ENFILE}:
        return HTTPException(
            status_code=503, detail="Audio service busy", headers={"Retry-After": "5"}
        )
    return HTTPException(status_code=500, detail="Audio file IO failure")


def _generation_parameters(
    payload: GenerateRequest, settings: Settings
) -> tuple[str, int | float | None]:
    prompt = payload.prompt.strip()
    if not prompt:
        raise ValueError("prompt 不能为空，请输入歌曲风格描述。")
    if len(prompt) > settings.prompt_max_chars:
        raise ValueError(f"prompt 不能超过 {settings.prompt_max_chars} 个字符。")
    if not any(split_generation_prompt(prompt)):
        raise ValueError("歌词和风格至少需要提供一项。")
    if payload.durationSeconds is not None:
        return prompt, payload.durationSeconds / 60
    if payload.durationMinutes is None or (
        isinstance(payload.durationMinutes, str)
        and payload.durationMinutes.strip().lower() == "auto"
    ):
        return prompt, None
    return prompt, settings.normalize_duration(payload.durationMinutes)


def _audio_reference(payload: GenerateRequest, settings: Settings) -> AudioReference | None:
    if payload.referenceId is None:
        return None
    provider = payload.provider or settings.music_provider
    if provider not in {"elevenlabs_music", "mureka_music"}:
        raise ValueError("音频参考仅支持 ElevenLabs 和 Mureka 模型，MiniMax 不支持。")
    if settings.music_api_mode != "real":
        raise ValueError("音频参考需要 MUSIC_API_MODE=real。")
    entry = find_reference(settings.output_dir, payload.referenceId)
    if entry is None:
        raise ValueError("参考音频不存在或已删除，请重新选择。")
    if entry.get("provider") != provider:
        raise ValueError("参考音频不属于当前模型，请为当前模型重新上传。")
    if provider == "mureka_music":
        if settings.mureka_model == "mureka-o2":
            raise ValueError("Mureka mureka-o2 模型不支持音频参考。")
        # Mureka uploads the local 30 s clip itself; duration and strength are ElevenLabs-only.
        return AudioReference(payload.referenceId, REFERENCE_MAX_SECONDS * 1000, "")
    if settings.elevenlabs_music_model_id not in ELEVENLABS_REFERENCE_MODELS:
        raise ValueError("音频参考仅支持 ElevenLabs music_v2 / music_v2_5 模型。")
    if settings.elevenlabs_force_instrumental:
        raise ValueError("音频参考不能与 ELEVENLABS_FORCE_INSTRUMENTAL=true 同时使用。")
    duration_ms = int(min(float(entry["durationSeconds"]), REFERENCE_MAX_SECONDS) * 1000)
    return AudioReference(
        entry["id"], duration_ms, settings.elevenlabs_reference_condition_strength
    )


def _public_base_url(request: Request, settings: Settings) -> str:
    return settings.public_base_url or str(request.base_url).rstrip("/")


def _current_operation(job: GenerationJob) -> str | None:
    for operation in ("mix", "replace", "split"):
        if getattr(job, f"{operation}_status") in {"pending", "running"}:
            return operation
    return job.last_operation


def _operation_status(job: GenerationJob) -> str | None:
    operation = _current_operation(job)
    return getattr(job, f"{operation}_status") if operation else None


def _publish_job(job: GenerationJob, request: Request, *, snapshot: bool = False) -> None:
    queues = request.app.state.job_subscribers.get(job.job_id, ())
    if not queues:
        return
    event = None
    if snapshot:
        event = copy.copy(job)
        event.result = copy.deepcopy(job.result)
    for queue in queues:
        # A queued event already wakes the consumer; a generation sentinel can coalesce
        # because it reads the live job. Operation snapshots always enqueue their state.
        if not snapshot and not queue.empty():
            continue
        if queue.full():
            queue.get_nowait()  # Bound memory for slow clients; keep the newest updates.
        queue.put_nowait(event)


def _read_operation_input(path: Path) -> None:
    require_readable_file(path, "音频输入文件不可读")
    with path.open("rb") as audio:
        audio.read(12)


def _mark_split_succeeded(job: GenerationJob) -> None:
    """分轨成功收尾的四个字段：WAV 落盘后正常结束与"编码中被取消"两条路径共用。"""
    job.split_status = "succeeded"
    job.split_stage = "completed"
    job.split_progress = SPLIT_COMPLETE_PROGRESS
    job.split_message = "音轨分离完成"


def _mark_replace_cancelled(job: GenerationJob) -> None:
    job.replace_cancel_requested = True
    job.replace_status = "cancelled"
    job.replace_stage = "cancelled"
    job.replace_progress = None
    job.replace_message = "人声替换已取消"
    job.replace_error = None


def _reported_mix_status(job: GenerationJob) -> tuple[str | None, int | None]:
    """对外汇报的合轨状态与曲目序号：有运行态就报它，否则从结果里推导。

    与替换人声同一套理由：`mixStatus` 不落盘（重启后回到 None），而结果里的 `mixedTrack`
    还在。只看运行态会让重启后的客户端收到"有成品、mixStatus 却是 null"的矛盾响应，把已经
    可以播放下载的成品判为失败。这里以结果为准补齐 succeeded，并给出成品所在曲目序号，
    让多曲目任务的客户端能定位到是哪一首。
    """
    if job.mix_status is not None:
        return job.mix_status, job.mix_song
    result = job.result
    if not isinstance(result, dict):
        return None, None
    mixed_songs = [
        index
        for index, song in enumerate([result, *(result.get("alternatives") or [])])
        if isinstance(song, dict)
        and (isinstance(song.get("mixedTrack"), str) or song_project(song, "mixed") is not None)
    ]
    if not mixed_songs:
        return None, None
    # 重启后没有"最后一次合的是哪首"的信息：只有唯一一首有成品时才敢报序号，
    # 多首都有成品时报 None —— 客户端应直接看每首歌自己的 mixedTrack。
    return "succeeded", mixed_songs[0] if len(mixed_songs) == 1 else None


def _replacement_file_exists(audio_url: str, settings: Settings, directory: Path) -> bool:
    """替换缓存与重启恢复只认可对应歌曲目录内仍存在的文件。"""
    try:
        target = _output_path_from_url(audio_url, settings)
        return target.parent == directory.resolve() and target.is_file()
    except (OSError, ValueError):
        return False


def _reported_replace_status(
    job: GenerationJob, settings: Settings, *, fingerprint: str | None = None
) -> tuple[str | None, int | None]:
    """对外汇报的替换状态：有在途操作就报它，否则从结果里推导。

    只依赖 job.replace_status 会有两个洞：它不落盘（重启后回到 None），也不被"恢复被删除的
    替换结果"这条路径重新赋值。于是客户端会收到"result 里有 replacedVocal、replaceStatus 却
    是 null"的矛盾响应，前端的替换状态机会直接判失败（"生成服务未返回人声替换状态"），
    撤回删除后再点替换就再也走不通。这里以结果为准补齐终态：

    - replacedVocal 文件在对应歌曲目录内存在、且模型指纹一致（或旧数据没有指纹）→ succeeded；
    - 指纹不一致且模型资产在场，说明缓存确实失效，交给 /replace 重新推理，不在这里谎报成功；
    - 指纹不一致但资产不在（缺挂载的实例）：指纹根本无从验证，此时既不能判失效也不该让
      客户端去跑一个必然失败的 /replace，按 succeeded 汇报，与合轨入口的逃逸一致。
    """
    if job.replace_status is not None:
        return job.replace_status, job.replace_song
    result = job.result
    if not isinstance(result, dict):
        return None, None
    replaced_songs = []
    for index, song in enumerate([result, *(result.get("alternatives") or [])]):
        if not isinstance(song, dict) or not isinstance(song.get("replacedVocal"), str):
            continue
        try:
            directory = job_song_dir(
                settings.output_dir, job.job_id, (song.get("songNumber") or index + 1) - 1
            )
        except ValueError:
            continue
        if not _replacement_file_exists(song["replacedVocal"], settings, directory):
            continue
        recorded = song.get("_replacedVocalModel")
        if isinstance(recorded, str) and recorded != fingerprint:
            if _rvc_assets_present(settings):
                continue
            logger.warning(
                "replacement model fingerprint is unverifiable (RVC assets missing); "
                "reporting the stored replacement as usable job_id=%s song=%s",
                job.job_id,
                index,
            )
        replaced_songs.append(index)
    if not replaced_songs:
        return None, None
    # Like mix, identify the song only when exactly one usable replacement survives.
    return "succeeded", replaced_songs[0] if len(replaced_songs) == 1 else None


async def _await_stuck_replacement_worker(
    conversion_task: asyncio.Task[None], *, job_id: str, song: int, reason: str
) -> None:
    """等待已经进入终态、但 RVC 线程仍在跑的替换收尾。

    额度按设计保留到线程真正退出（详见 README），所以这里的等待没有上限；为了让运维能发现
    "线程不返回、额度一直被占"的情况，等待期间计数并写日志，线程退出后再撤销。
    """
    global _replacement_wind_down_active
    _replacement_wind_down_active += 1
    logger.warning(
        "vocal replacement worker still running after %s; holding capacity until it exits "
        "job_id=%s song=%s",
        reason,
        job_id,
        song,
    )
    try:
        await conversion_task
    except Exception:
        logger.warning(
            "vocal replacement worker failed while winding down job_id=%s song=%s",
            job_id,
            song,
            exc_info=True,
        )
    finally:
        _replacement_wind_down_active = max(0, _replacement_wind_down_active - 1)
        logger.warning("vocal replacement worker exited job_id=%s song=%s", job_id, song)


@lru_cache(maxsize=8)
def _file_sha256(path: str, _size: int, _mtime_ns: int) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _rvc_assets_present(settings: Settings) -> bool:
    """模型/索引文件此刻是否真的读得到。

    `_rvc_model_fingerprint` 在文件缺失时把 "missing" 拼进摘要，所以"模型被换掉"和
    "资产不在"都会得到一个与记录值不同的指纹。替换人声只有在资产存在时才能跑成功，
    因此结果里记录的指纹一定是真实模型的摘要；反过来，当前资产缺失只说明这台实例读不到
    模型（合轨本身只用 ffmpeg，不需要 RVC），不能据此判定模型被换过。
    """
    if not settings.rvc_model_path.is_file():
        return False
    index_path = settings.rvc_index_path
    return index_path is None or index_path.is_file()


def _rvc_model_fingerprint(settings: Settings) -> str:
    paths = [settings.rvc_model_path, settings.rvc_index_path]
    parts = [settings.rvc_model_version, "rms_mix_rate=0"]
    for path in paths:
        if path is None:
            parts.append("")
            continue
        try:
            stat = path.stat()
            parts.append(_file_sha256(str(path.resolve()), stat.st_size, stat.st_mtime_ns))
        except OSError:
            parts.append("missing")
    return f"v1:{hashlib.sha256(chr(0).join(parts).encode()).hexdigest()}"


async def _async_job_response(
    job: GenerationJob, request: Request, settings: Settings
) -> dict[str, Any]:
    return _job_response(
        job, request, settings, fingerprint=request.app.state.rvc_model_fingerprint
    )


def _job_response(
    job: GenerationJob,
    request: Request,
    settings: Settings,
    *,
    include_operations: bool = True,
    include_waveforms: bool = True,
    fingerprint: str | None = None,
) -> dict[str, Any]:
    response = job.response()
    if settings.song_retention_enabled:
        expiry = (
            expires_at(job, settings.song_retention_days) if retention_enforced(settings) else None
        )
        response["expiresAt"] = expiry.isoformat() if expiry else None
        expired = expiry is not None and expiry <= datetime.now(timezone.utc)
        if settings.song_retention_dry_run:
            retention_state = "dry_run"
        elif expiry is None:
            retention_state = "unmanaged"
        elif expired:
            retention_state = "expired_active" if is_job_active(job) else "expired"
        else:
            retention_state = "retained"
        response["retentionState"] = retention_state
        response["audioAvailable"] = bool(job.result) and (
            not retention_enforced(settings) or (expiry is not None and not expired)
        )
    if job.result is not None:
        response["result"] = _render_result_urls(
            job.result,
            _public_base_url(request, settings),
            settings,
            include_waveforms=include_waveforms,
        )
    if not include_operations:
        return response

    response.update(
        operation=None,
        operationStatus=None,
        operationSong=None,
        operationStage=None,
        operationProgress=None,
        operationMessage=None,
    )
    operation = _current_operation(job)
    if operation:
        operation_status = getattr(job, f"{operation}_status")
        if operation_status is not None:
            response.update(
                operation=operation,
                operationStatus=operation_status,
                operationSong=getattr(job, f"{operation}_song"),
                operationStage=getattr(job, f"{operation}_stage"),
                operationProgress=getattr(job, f"{operation}_progress"),
                operationMessage=getattr(job, f"{operation}_message"),
            )
    if response["operationStatus"] in {"pending", "running"}:
        # Keep the released clients' running-operation projection. Terminal operations
        # never overwrite the persisted music generation task's success or failure.
        response.update(
            status=response["operationStatus"],
            stage=response["operationStage"],
            progress=response["operationProgress"],
            message=response["operationMessage"],
        )
    if job.split_status is not None:
        response["splitStatus"] = job.split_status
        response["splitSong"] = job.split_song
        response["splitError"] = job.split_error
    replace_status, replace_song = _reported_replace_status(job, settings, fingerprint=fingerprint)
    if replace_status is not None:
        response["replaceStatus"] = replace_status
        response["replaceSong"] = replace_song
        response["replaceError"] = job.replace_error
    mix_status, mix_song = _reported_mix_status(job)
    if mix_status is not None:
        response["mixStatus"] = mix_status
        response["mixSong"] = mix_song
        response["mixError"] = job.mix_error
    if response["operationStatus"] is None:
        # No runtime operation survived restart. Reuse the existing artifact-derived
        # statuses; a mix is downstream of replacement, so prefer it when both exist.
        recovered = (
            "mix"
            if mix_status == "succeeded"
            else ("replace" if replace_status == "succeeded" else None)
        )
        if recovered:
            response.update(
                operation=recovered,
                operationStatus="succeeded",
                operationSong=response[f"{recovered}Song"],
                operationStage="completed",
                operationProgress=100,
                operationMessage="合轨完成" if recovered == "mix" else "人声替换完成",
            )
    return response


def _render_result_urls(
    result: dict[str, Any],
    base_url: str,
    settings: Settings,
    *,
    include_waveforms: bool = True,
) -> dict[str, Any]:
    rendered = copy.deepcopy(result if include_waveforms else _without_waveforms(result))
    stem_order = {name: index for index, name in enumerate(STEM_NAMES)}
    songs = [rendered, *rendered.get("alternatives", [])]
    for song in songs:
        normalize_mixed_song(song)
    for output in [*songs, *(song_project(song, "mixed") for song in songs)]:
        if not isinstance(output, dict):
            continue
        output.setdefault("audioRevision", 0)
        drop_editor_state(output)
        full_track = output.get("fullTrack")
        if isinstance(full_track, str):
            output["fullTrack"] = _public_audio_url(full_track, base_url, settings)
        stems = output.get("stems")
        if isinstance(stems, dict):
            output["stems"] = {
                name: _public_audio_url(url, base_url, settings)
                for name, url in sorted(
                    stems.items(),
                    key=lambda item: stem_order.get(item[0].rsplit(":", 1)[-1], len(stem_order)),
                )
                if isinstance(url, str)
            }
            output["stemUrls"] = list(output["stems"].values())
        replaced_vocal = output.get("replacedVocal")
        if isinstance(replaced_vocal, str):
            output["replacedVocal"] = _public_audio_url(replaced_vocal, base_url, settings)
        mixed_track = output.get("mixedTrack")
        if isinstance(mixed_track, str):
            output["mixedTrack"] = _public_audio_url(mixed_track, base_url, settings)

        def valid_preview(source: object) -> str | None:
            if not isinstance(source, str):
                return None
            try:
                preview = existing_playback_mp3(_output_path_from_url(source, settings))
                if preview:
                    return build_public_audio_url(
                        base_url, preview.relative_to(settings.output_dir.resolve())
                    )
            except (OSError, ValueError):
                pass
            return None

        valid = {}
        for key in ("fullTrack", "replacedVocal", "mixedTrack"):
            url = valid_preview(output.get(key))
            if url:
                valid[key] = url
        stems = output.get("stems")
        if isinstance(stems, dict):
            valid_stems = {}
            for name, source in stems.items():
                url = valid_preview(source)
                if url:
                    valid_stems[name] = url
            if valid_stems:
                valid["stems"] = valid_stems
        if valid or "playback" in output:
            output["playback"] = valid
        for key in [key for key in output if isinstance(key, str) and key.startswith("_")]:
            del output[key]
    for song in songs:
        # 兼容字段：替换后那首的成品同时以 mixedTrack 暴露（生成记录、旧客户端按它判断有无成品）。
        mixed = song_project(song, "mixed")
        if mixed is not None and isinstance(mixed.get("fullTrack"), str):
            song["mixedTrack"] = mixed["fullTrack"]
            preview = (mixed.get("playback") or {}).get("fullTrack")
            if isinstance(preview, str):
                _playback_dict(song)["mixedTrack"] = preview
            waveform = (mixed.get("waveforms") or {}).get("full")
            if isinstance(waveform, list) and isinstance(song.get("waveforms"), dict):
                song["waveforms"]["mix"] = waveform
    return rendered


def _without_waveforms(result: dict[str, Any]) -> dict[str, Any]:
    """浅拷贝一份任务结果，把波形包络清空。

    列表不绘制编辑器车道，音频编辑操作的 SSE 中间帧也不需要重复发送波形。
    生成中的 SSE 只在歌曲完成帧和终态携带波形，其余帧清空键以保持响应结构稳定。
    需要真实波形时可查询 `GET /api/jobs/{job_id}`。浅拷贝不会改动持久化结果。
    """
    projected = {key: value for key, value in result.items() if key != "waveforms"}
    projected["waveforms"] = {}
    if isinstance(result.get("mixed"), dict):
        projected["mixed"] = _without_waveforms(result["mixed"])
    alternatives = result.get("alternatives")
    if isinstance(alternatives, list):
        projected["alternatives"] = [
            _without_waveforms(item) if isinstance(item, dict) else item for item in alternatives
        ]
    return projected


def _public_audio_url(audio_url: str, base_url: str, settings: Settings) -> str:
    try:
        target = _output_path_from_url(audio_url, settings)
    except ValueError:
        return audio_url
    return build_public_audio_url(base_url, target.relative_to(settings.output_dir.resolve()))


def _output_path_from_url(audio_url: str, settings: Settings):
    return output_path_from_url(audio_url, settings.output_dir)


def _referenced_output_paths(value: Any, settings: Settings) -> set[Path]:
    """Collect every output audio file referenced anywhere inside a job structure."""
    found: set[Path] = set()
    pending = [value]
    while pending:
        item = pending.pop()
        if isinstance(item, dict):
            pending.extend(item.values())
        elif isinstance(item, list):
            pending.extend(entry for entry in item if isinstance(entry, (dict, list, str)))
        elif isinstance(item, str) and item.lower().endswith(tuple(MIX_INPUT_SUFFIXES)):
            with suppress(ValueError):
                found.add(_output_path_from_url(item, settings))
    return found


def _unlink_paths(paths: set[Path]) -> None:
    for path in paths:
        try:
            path.unlink(missing_ok=True)
        except OSError:
            logger.warning("stale audio cleanup skipped path=%s", path, exc_info=True)


def _stem_trash_path(target, settings: Settings, job_id: str, song: int = 0):
    root = settings.output_dir.resolve() / ".trash" / job_id
    return root / (f"song_{song + 1}" if song else "") / target.name


def _parse_byte_range(value: str | None, file_size: int) -> tuple[int, int]:
    if not value:
        return 0, file_size - 1
    if not value.startswith("bytes=") or "," in value:
        raise HTTPException(status_code=416, detail="Invalid byte range")
    start_text, separator, end_text = value[6:].partition("-")
    if not separator:
        raise HTTPException(status_code=416, detail="Invalid byte range")
    try:
        if not start_text:
            suffix_length = int(end_text)
            if suffix_length <= 0:
                raise ValueError
            start = max(0, file_size - suffix_length)
            end = max(0, file_size - 1)
        else:
            start = int(start_text)
            end = int(end_text) if end_text else max(0, file_size - 1)
    except ValueError as exc:
        raise HTTPException(status_code=416, detail="Invalid byte range") from exc
    if start < 0 or end < start or start >= file_size:
        raise HTTPException(status_code=416, detail="Invalid byte range")
    return start, min(end, file_size - 1)


async def _stream_file(source, start: int, end: int) -> AsyncIterator[bytes]:
    remaining = end - start + 1
    with source:
        source.seek(start)
        while remaining > 0:
            chunk = source.read(min(64 * 1024, remaining))
            if not chunk:
                break
            remaining -= len(chunk)
            yield chunk


def run() -> None:
    settings = Settings()
    configure_logging()
    uvicorn.run("app.main:app", host=settings.host, port=settings.port, reload=False)


app = create_app()
