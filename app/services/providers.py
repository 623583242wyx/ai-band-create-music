from __future__ import annotations

import asyncio
import json
import logging
import re
import time
import wave
from collections.abc import AsyncIterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol
from urllib.parse import quote
from uuid import NAMESPACE_URL, uuid5

import httpx

from app.core.config import PROJECT_ROOT, Settings
from app.core.errors import GenerationError, ProviderGlobalError
from app.services.audio_files import download_audio, require_readable_file, write_stream_atomically
from app.services.job_files import job_song_dir, update_provider_diagnostic
from app.services.prompt import (
    _http_failure_message,
    split_generation_prompt,
)
from app.services.references import (
    REFERENCE_MAX_SECONDS,
    find_reference,
    loop_reference_audio,
    reference_audio_name,
    references_dir,
    remote_reference_id,
)

logger = logging.getLogger(__name__)

MINIMAX_GENERATE_PATH = "/v1/audio/jobs"
GLOBAL_PROVIDER_HTTP_STATUSES = {401, 402, 403, 429}
MINIMAX_LYRIC_LINE_LIMIT = 32
# The 4100-character prompt limit applies to both compose and stream endpoints:
# https://elevenlabs.io/docs/api-reference/music/compose
# https://elevenlabs.io/docs/api-reference/music/stream
ELEVENLABS_PROMPT_MAX_CHARS = 4100
# Auto output validation follows the compose API's 3-second minimum song length.
ELEVENLABS_AUTO_MIN_DURATION_SECONDS = 3
# Live /v1/music compose check (2026-09-23), music_v2 pcm_44100:
# a 3s request returned 529,200 bytes.
# (44,100 Hz × 2 channels × 2 bytes × 3s), with distinct signals in both channels.
# This account received HTTP 200; the wrapped WAV decoded as 16-bit stereo, 3.000s.
# Live /v1/music/stream check (2026-09-29), music_v2, output_format=pcm_44100:
# music_length_ms=3000, force_instrumental=true, short piano prompt; HTTP 200,
# 529,200 bytes in 408 chunks; wrapped WAV: 44,100 Hz, 16-bit stereo, 3.000s.
# Audio reference / conditioning_ref is documented for these models only:
# https://elevenlabs.io/docs/overview/capabilities/music (Audio Reference)
ELEVENLABS_REFERENCE_MODELS = frozenset({"music_v2", "music_v2_5"})
ELEVENLABS_PCM_CHANNELS = 2
ELEVENLABS_PCM_SAMPLE_WIDTH = 2
# https://platform.mureka.ai/docs/api/operations/post-v1-song-generate.html
MUREKA_LYRICS_MAX_CHARS = 5000
MUREKA_PROMPT_MAX_CHARS = 1024
LYRIC_BREAK_PATTERN = re.compile(r"(?<=[，。！？；、,.!?;:：])")


def _reference_upload_http_error(
    provider: str, credential: str, error: httpx.HTTPStatusError
) -> GenerationError:
    status = error.response.status_code
    detail = _http_failure_message(error)
    if status in {401, 403}:
        return ProviderGlobalError(
            f"{provider} 参考音频上传失败：鉴权或权限不足（{detail}），请检查 {credential}。"
        )
    if status == 402:
        return ProviderGlobalError(
            f"{provider} 参考音频上传失败：账户额度或套餐不可用（{detail}）。"
        )
    if status == 429:
        return ProviderGlobalError(
            f"{provider} 参考音频上传失败：请求过于频繁（{detail}），请稍后重试。"
        )
    if status == 413:
        return GenerationError(
            f"{provider} 参考音频上传失败：处理后的文件超过服务限制（{detail}）。"
        )
    if status in {415, 422}:
        return GenerationError(
            f"{provider} 参考音频上传失败：文件格式或时长不被接受（{detail}）。"
        )
    if status >= 500:
        return GenerationError(
            f"{provider} 参考音频上传失败：服务暂时不可用（{detail}），请稍后重试。"
        )
    return GenerationError(f"{provider} 参考音频上传失败：{detail}")


class ProviderProgressCallback(Protocol):
    async def __call__(
        self,
        stage: str,
        step: int | None,
        total_steps: int | None,
        *,
        received_audio_seconds: float | None = None,
        expected_audio_seconds: float | None = None,
    ) -> None: ...


def build_elevenlabs_prompt(
    structured_prompt: str,
    duration_minutes: float | None,
    clear_chinese_vocal_mode: bool,
    lyrics: str = "",
    require_lyrics: bool = False,
) -> str:
    requirements = [structured_prompt]
    if duration_minutes is not None:
        requirements.append(f"Target duration: {duration_minutes:g} minutes.")
    if clear_chinese_vocal_mode:
        requirements.append(
            "Chinese clear vocal mode: Use Mandarin Chinese lead vocals with clear articulation "
            "and natural phrasing. Keep lyric density moderate with short singable lines and "
            "breathing space between phrases. Make the lead vocal forward in the mix. Use light "
            "reverb, minimal backing vocals, and avoid mumbling or swallowed syllables."
        )
    if lyrics:
        requirements.extend(["Original lyrics (preserve wording and line order):", lyrics])
    if require_lyrics:
        # 收起页没有用户歌词：明确要求写唱，避免风格里的“无歌词/哼唱”被理解成纯音乐。
        requirements.append(
            "Write and sing original lyrics with a clear lead vocal. "
            "Do not generate an instrumental or wordless vocal track. "
            "Hums or vocal chops may only be backing texture; "
            "the lead vocal must sing actual lyrics."
        )
    return "\n".join(requirements)


def _composition_lyrics(plan: object) -> str | None:
    """从 ElevenLabs 详细响应的 composition plan 里取实际唱出的歌词。"""
    if not isinstance(plan, dict):
        return None
    chunks = plan.get("chunks")
    if isinstance(chunks, list):
        texts = [
            chunk.get("text")
            for chunk in chunks
            if isinstance(chunk, dict) and isinstance(chunk.get("text"), str)
        ]
        text = "\n".join(value.strip() for value in texts if value.strip())
        return text if _has_lyric_text(text) else None
    sections = plan.get("sections")
    if isinstance(sections, list):
        lines: list[str] = []
        for section in sections:
            section_lines = section.get("lines") if isinstance(section, dict) else None
            if isinstance(section_lines, list):
                lines.extend(str(line).strip() for line in section_lines if str(line).strip())
        text = "\n".join(lines)
        return text if _has_lyric_text(text) else None
    return None


def _has_lyric_text(text: str) -> bool:
    """去掉段落标签和花括号提示后还有没有真正要唱的字。"""
    return bool(re.sub(r"\[[^\]]+\]|\{[^}]+\}", "", text).strip())


async def _detailed_music_audio(
    response: httpx.Response, metadata: dict[str, Any]
) -> AsyncIterator[bytes]:
    """增量解析 detailed multipart；按 part 的 Content-Type 选 JSON/音频，其它段跳过。"""
    content_type = response.headers.get("content-type", "")
    marker = "boundary="
    boundary = (
        content_type.split(marker, 1)[1].split(";", 1)[0].strip().strip('"')
        if marker in content_type
        else ""
    )
    if not boundary:
        raise GenerationError("ElevenLabs 音乐生成失败：详细响应缺少 multipart boundary。")
    delimiter = f"--{boundary}".encode()
    separator = b"\r\n" + delimiter
    buffer = b""
    state = "preamble"
    async for chunk in response.aiter_bytes():
        if not chunk:
            continue
        buffer += chunk
        while True:
            if state == "preamble":
                index = buffer.find(delimiter)
                if index < 0:
                    buffer = buffer[-len(delimiter) :]
                    break
                buffer = buffer[index + len(delimiter) :]
                state = "after_boundary"
            elif state == "after_boundary":
                if len(buffer) < 2:
                    break
                if buffer.startswith(b"--"):
                    return
                index = buffer.find(b"\r\n")
                if index < 0:
                    break
                buffer = buffer[index + 2 :]
                state = "headers"
            elif state == "headers":
                index = buffer.find(b"\r\n\r\n")
                if index < 0:
                    break
                header = buffer[:index].lower()
                buffer = buffer[index + 4 :]
                if b"application/json" in header:
                    state = "json"
                elif b"text/" in header or b"image/" in header:
                    state = "skip"
                else:
                    # 未识别的二进制段按音频处理，兼容 binary/octet-stream 等类型。
                    state = "audio"
            elif state == "json":
                index = buffer.find(separator)
                if index < 0:
                    break
                body = buffer[:index]
                buffer = buffer[index + len(separator) :]
                try:
                    parsed = json.loads(body.decode("utf-8"))
                except (UnicodeDecodeError, ValueError) as exc:
                    raise GenerationError(
                        "ElevenLabs 音乐生成失败：详细响应元数据无效。"
                    ) from exc
                if isinstance(parsed, dict):
                    metadata.update(parsed)
                # 数组等非对象 JSON 段（例如时间戳）跳过，不影响音频。
                state = "after_boundary"
            elif state == "audio":
                index = buffer.find(separator)
                if index < 0:
                    hold = len(separator)
                    if len(buffer) > hold:
                        yield buffer[:-hold]
                        buffer = buffer[-hold:]
                    break
                audio = buffer[:index]
                if audio:
                    yield audio
                buffer = buffer[index + len(separator) :]
                state = "after_boundary"
            else:  # skip：时间戳/波形等附加段，不写进 WAV
                index = buffer.find(separator)
                if index < 0:
                    buffer = buffer[-len(separator) :]
                    break
                buffer = buffer[index + len(separator) :]
                state = "after_boundary"
    if state == "preamble":
        raise GenerationError("ElevenLabs 音乐生成失败：详细响应未找到 multipart boundary。")
    raise GenerationError("ElevenLabs 音乐生成失败：详细响应结构无效。")


def _detailed_music_text(metadata: dict[str, Any]) -> tuple[str | None, str | None]:
    song = metadata.get("song_metadata") or metadata.get("songMetadata") or {}
    plan = metadata.get("composition_plan") or metadata.get("compositionPlan") or {}
    title = song.get("title") if isinstance(song, dict) else None
    lyrics = song.get("lyrics") if isinstance(song, dict) else None
    if not isinstance(lyrics, str) or not lyrics.strip() or not _has_lyric_text(lyrics):
        lyrics = _composition_lyrics(plan)
    return (
        title.strip() if isinstance(title, str) and title.strip() else None,
        lyrics.strip() if isinstance(lyrics, str) and lyrics.strip() else None,
    )




def _audio_target(
    settings: Settings, fallback_name: str, job_id: str | None, variation: int
) -> Path:
    if job_id:
        suffix = Path(fallback_name).suffix
        return job_song_dir(settings.output_dir, job_id, variation) / (
            f"full_song_{job_id}_{variation + 1}{suffix}"
        )
    return settings.output_dir / fallback_name


@dataclass(frozen=True, slots=True)
class AudioReference:
    """A locally stored audio reference supplied to a music provider."""

    song_id: str
    duration_ms: int
    strength: str


@dataclass(frozen=True, slots=True)
class MusicResult:
    audio_path: Path
    debug: dict[str, Any]
    # Create 收起页没有用户歌词，服务商返回什么就存什么。
    title: str | None = None
    lyrics: str | None = None


class MusicProvider(Protocol):
    async def generate(
        self,
        structured_prompt: str,
        duration_seconds: int | None,
        user_prompt: str,
        *,
        variation: int = 0,
        progress: ProviderProgressCallback | None = None,
        job_id: str | None = None,
    ) -> MusicResult: ...


def _wrap_long_lyric_lines(lyrics: str) -> str:
    output: list[str] = []
    for line in lyrics.splitlines():
        if len(line) <= MINIMAX_LYRIC_LINE_LIMIT or (
            line.strip().startswith("[") and line.strip().endswith("]")
        ):
            output.append(line)
            continue
        current = ""
        for part in LYRIC_BREAK_PATTERN.split(line):
            if not part:
                continue
            if current and len(current) + len(part) > MINIMAX_LYRIC_LINE_LIMIT:
                output.append(current)
                current = ""
            while len(part) > MINIMAX_LYRIC_LINE_LIMIT:
                output.append(part[:MINIMAX_LYRIC_LINE_LIMIT])
                part = part[MINIMAX_LYRIC_LINE_LIMIT:]
            current += part
        if current:
            output.append(current)
    return "\n".join(output)


def _response_diagnostic(response: httpx.Response) -> dict[str, Any]:
    try:
        body = response.json()
    except ValueError:
        body = response.text
    return {
        "statusCode": response.status_code,
        "headers": dict(response.headers),
        "body": body,
    }


def extract_task_id(data: Any) -> str | None:
    if not isinstance(data, dict):
        return None
    nested = data.get("data") if isinstance(data.get("data"), dict) else {}
    return next(
        (
            str(value)
            for value in (
                data.get("taskId"),
                data.get("task_id"),
                data.get("id"),
                nested.get("taskId"),
                nested.get("task_id"),
                nested.get("id"),
            )
            if value
        ),
        None,
    )


def extract_task_status(data: Any) -> str:
    if not isinstance(data, dict):
        return ""
    nested = data.get("data") if isinstance(data.get("data"), dict) else {}
    return str(
        data.get("status") or data.get("state") or nested.get("status") or nested.get("state") or ""
    ).lower()


def extract_generated_audio_url(data: Any) -> str | None:
    if not isinstance(data, dict):
        return None
    nested = data.get("data")
    candidates: list[Any] = [data]
    if isinstance(nested, dict):
        candidates.append(nested)
    elif isinstance(nested, list) and nested and isinstance(nested[0], dict):
        candidates.append(nested[0])
    for candidate in candidates:
        for key in ("audioUrl", "audio_url", "url"):
            if candidate.get(key):
                return str(candidate[key])
    return None


def _suno_clips(data: Any) -> list[dict[str, Any]]:
    if isinstance(data, list):
        return [item for item in data if isinstance(item, dict)]
    if not isinstance(data, dict):
        return []
    for key in ("data", "clips", "items"):
        value = data.get(key)
        if isinstance(value, list):
            return [item for item in value if isinstance(item, dict)]
    return []


def extract_suno_clip_ids(data: Any) -> list[str]:
    return [str(clip["id"]) for clip in _suno_clips(data) if clip.get("id")]


def extract_suno_audio_url(data: Any) -> str | None:
    clips = _suno_clips(data)
    for clip in clips:
        status = str(clip.get("status", "")).lower()
        if clip.get("audio_url") and status in {"streaming", "complete", "completed", "succeeded"}:
            return str(clip["audio_url"])
    for clip in clips:
        if clip.get("audio_url"):
            return str(clip["audio_url"])
    return extract_generated_audio_url(data)


class MockMusicProvider:
    def __init__(self, settings: Settings) -> None:
        self._settings = settings

    async def generate(
        self,
        structured_prompt: str,
        duration_seconds: int | None,
        user_prompt: str,
        *,
        variation: int = 0,
        progress: ProviderProgressCallback | None = None,
        job_id: str | None = None,
    ) -> MusicResult:
        require_readable_file(
            self._settings.mock_full_song_path,
            "音乐生成 mock 文件不存在，请放置本地完整歌曲或设置 MOCK_FULL_SONG_PATH",
        )
        return MusicResult(self._settings.mock_full_song_path, {"provider": "mock"})


class GenericMusicProvider:
    def __init__(self, settings: Settings, client: httpx.AsyncClient) -> None:
        self._settings = settings
        self._client = client

    def _headers(self) -> dict[str, str]:
        if self._settings.music_api_key is None:
            raise GenerationError("音乐生成失败：MUSIC_PROVIDER=generic 时必须配置 MUSIC_API_KEY。")
        return {
            "Authorization": f"Bearer {self._settings.music_api_key.get_secret_value()}",
            "Content-Type": "application/json",
        }

    async def generate(
        self,
        structured_prompt: str,
        duration_seconds: int | None,
        user_prompt: str,
        *,
        variation: int = 0,
        progress: ProviderProgressCallback | None = None,
        job_id: str | None = None,
    ) -> MusicResult:
        if not self._settings.music_api_base_url:
            raise GenerationError(
                "音乐生成失败：MUSIC_API_MODE=real 时必须配置 MUSIC_API_BASE_URL。"
            )
        try:
            response = await self._client.post(
                f"{self._settings.music_api_base_url}{self._settings.music_generate_path}",
                json={
                    "prompt": structured_prompt,
                    "duration_minutes": duration_seconds / 60,
                    "duration_seconds": duration_seconds,
                    "model": self._settings.music_model or None,
                    "callback_url": self._settings.music_callback_url or None,
                },
                headers=self._headers(),
                timeout=self._settings.music_api_timeout_seconds,
            )
            response.raise_for_status()
            data = response.json()
            audio_url = extract_generated_audio_url(data)
            if audio_url:
                audio_path = await self._download(
                    audio_url, f"full_song_{time.time_ns()}.mp3", job_id, variation
                )
                return MusicResult(audio_path, {"provider": "generic", "mode": "direct_audio_url"})
            task_id = extract_task_id(data)
            if not task_id:
                raise GenerationError(
                    "音乐生成接口未返回 taskId 或 audioUrl，请检查代理接口响应结构。"
                )
            audio_path = await self._poll(task_id, job_id, variation)
            return MusicResult(
                audio_path, {"provider": "generic", "mode": "polling", "taskId": task_id}
            )
        except GenerationError:
            raise
        except (httpx.HTTPError, ValueError) as exc:
            raise GenerationError(f"音乐生成失败：{_http_failure_message(exc)}") from exc

    async def _poll(self, task_id: str, job_id: str | None, variation: int) -> Path:
        url = f"{self._settings.music_api_base_url}{self._settings.music_status_path}"
        for _ in range(self._settings.music_max_poll_attempts):
            await asyncio.sleep(self._settings.music_poll_interval_seconds)
            response = await self._client.get(
                url,
                params={"taskId": task_id},
                headers=self._headers(),
                timeout=self._settings.music_api_timeout_seconds,
            )
            response.raise_for_status()
            data = response.json()
            status = extract_task_status(data)
            audio_url = extract_generated_audio_url(data)
            if audio_url and status in {
                "success",
                "succeeded",
                "completed",
                "complete",
                "done",
                "",
            }:
                return await self._download(
                    audio_url, f"full_song_{task_id}.mp3", job_id, variation
                )
            if status in {"failed", "error", "canceled", "cancelled"}:
                raise GenerationError(f"音乐生成任务失败，taskId={task_id}，status={status}")
        raise GenerationError(f"音乐生成任务超时，taskId={task_id}")

    async def _download(
        self, audio_url: str, name: str, job_id: str | None, variation: int
    ) -> Path:
        return await download_audio(
            self._client,
            audio_url,
            _audio_target(self._settings, name, job_id, variation),
            timeout=self._settings.music_download_timeout_seconds,
            trusted_local_root=PROJECT_ROOT,
        )


class ElevenLabsMusicProvider:
    def __init__(self, settings: Settings, client: httpx.AsyncClient) -> None:
        self._settings = settings
        self._client = client

    def _api_key(self) -> str:
        secret = self._settings.elevenlabs_api_key or self._settings.music_api_key
        if secret is None:
            raise ProviderGlobalError("ElevenLabs 音乐生成失败：缺少 ELEVENLABS_API_KEY 环境变量。")
        return secret.get_secret_value()

    def _headers(self) -> dict[str, str]:
        return {"xi-api-key": self._api_key(), "Content-Type": "application/json"}

    async def upload_reference(self, path: Path, filename: str) -> str:
        """Upload a prepared clip and return its song_id (billed like one generation)."""
        url = f"{self._settings.elevenlabs_music_base_url}/v1/music/upload"
        try:
            with path.open("rb") as audio:
                response = await self._client.post(
                    url,
                    files={"file": (filename, audio, "audio/mpeg")},
                    headers={"xi-api-key": self._api_key()},
                    timeout=self._settings.music_api_timeout_seconds,
                )
            response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            raise _reference_upload_http_error("ElevenLabs", "ELEVENLABS_API_KEY", exc) from exc
        except httpx.TimeoutException as exc:
            raise GenerationError("ElevenLabs 参考音频上传失败：请求超时，请稍后重试。") from exc
        except httpx.ConnectError as exc:
            raise GenerationError(
                "ElevenLabs 参考音频上传失败：无法连接 ElevenLabs 服务。"
            ) from exc
        except httpx.HTTPError as exc:
            raise GenerationError(
                f"ElevenLabs 参考音频上传失败：{_http_failure_message(exc)}"
            ) from exc
        try:
            data = response.json()
        except ValueError as exc:
            raise GenerationError(
                "ElevenLabs 参考音频上传失败：服务返回的响应不是有效 JSON。"
            ) from exc
        song_id = data.get("song_id") if isinstance(data, dict) else None
        if not isinstance(song_id, str) or not re.fullmatch(r"[A-Za-z0-9_-]{1,128}", song_id):
            raise GenerationError("ElevenLabs 参考音频上传失败：接口未返回有效的 song_id。")
        return song_id

    async def _reference_song_id(self, reference_id: str) -> str:
        entry = find_reference(self._settings.output_dir, reference_id)
        # References created before local-only uploads already contain ElevenLabs song ids.
        if not entry or not entry.get("localOnly"):
            return reference_id
        return await remote_reference_id(
            self._settings.output_dir,
            reference_id,
            "elevenlabs_music",
            f"{self._settings.elevenlabs_music_base_url}:{self._api_key()}",
            self._upload_local_reference,
        )

    async def _upload_local_reference(self, entry: dict[str, Any]) -> str:
        audio = entry.get("audio")
        if not isinstance(audio, str) or not audio:
            raise GenerationError("参考音频不存在或已删除，请重新选择。")
        path = references_dir(self._settings.output_dir) / Path(audio).name
        require_readable_file(path, "参考音频文件不可读")
        looped = None
        try:
            if float(entry["durationSeconds"]) < 10:
                looped = path.parent / f".elevenlabs-{reference_audio_name()}"
                await loop_reference_audio(path, looped)
            return await self.upload_reference(looped or path, path.name)
        except ValueError as exc:
            raise GenerationError(f"ElevenLabs 参考音频处理失败：{exc}") from exc
        finally:
            if looped is not None:
                looped.unlink(missing_ok=True)

    async def _reference_plan(
        self,
        prompt: str,
        duration_seconds: int | None,
        reference: AudioReference,
        job_id: str | None,
        variation: int,
    ) -> tuple[dict[str, Any], float]:
        """Build a free composition plan from the prompt and condition its first chunk.

        conditioning_ref only exists on composition-plan chunks, which cannot be combined with
        `prompt`; /v1/music/plan turns the same prompt into chunks without spending credits.
        """
        url = f"{self._settings.elevenlabs_music_base_url}/v1/music/plan"
        body: dict[str, Any] = {
            "prompt": prompt,
            "model_id": self._settings.elevenlabs_music_model_id,
        }
        if duration_seconds is not None:
            body["music_length_ms"] = duration_seconds * 1000
        response = await self._client.post(
            url,
            json=body,
            headers=self._headers(),
            timeout=self._settings.music_api_timeout_seconds,
        )
        update_provider_diagnostic(
            self._settings.output_dir,
            job_id,
            variation,
            planRequest={"method": "POST", "url": url, "body": body},
            planResponse=_response_diagnostic(response),
        )
        response.raise_for_status()
        plan = response.json()
        chunks = plan.get("chunks") if isinstance(plan, dict) else None
        if (
            not isinstance(chunks, list)
            or not chunks
            or not all(isinstance(chunk, dict) for chunk in chunks)
            or "text" not in chunks[0]
        ):
            raise GenerationError("ElevenLabs 音乐生成失败：composition plan 响应结构无效。")
        durations = [chunk.get("duration_ms") for chunk in chunks]
        if not all(type(value) is int and value > 0 for value in durations):
            raise GenerationError("ElevenLabs 音乐生成失败：composition plan 缺少有效的段落时长。")
        chunks[0] = {
            **chunks[0],
            "conditioning_ref": {
                "song_id": reference.song_id,
                "range": {"start_ms": 0, "end_ms": reference.duration_ms},
            },
            "condition_strength": reference.strength,
        }
        return {**plan, "chunks": chunks}, sum(durations) / 1000

    async def generate(
        self,
        structured_prompt: str,
        duration_seconds: int | None,
        user_prompt: str,
        *,
        variation: int = 0,
        progress: ProviderProgressCallback | None = None,
        job_id: str | None = None,
        reference: AudioReference | None = None,
        direct_prompt: bool = False,
    ) -> MusicResult:
        if duration_seconds is not None and duration_seconds <= 0:
            raise GenerationError("ElevenLabs 音乐生成失败：目标时长必须大于 0 秒。")
        force_instrumental = self._settings.elevenlabs_force_instrumental
        if reference is not None:
            if force_instrumental:
                raise GenerationError(
                    "ElevenLabs 音乐生成失败：音频参考需要 composition plan，"
                    "不能与 ELEVENLABS_FORCE_INSTRUMENTAL=true 同时使用。"
                )
            if self._settings.elevenlabs_music_model_id not in ELEVENLABS_REFERENCE_MODELS:
                raise GenerationError(
                    "ElevenLabs 音乐生成失败：音频参考仅支持 music_v2 / music_v2_5，"
                    f"当前 ELEVENLABS_MUSIC_MODEL_ID={self._settings.elevenlabs_music_model_id}。"
                )
        clear_chinese = (
            self._settings.elevenlabs_clear_chinese_vocal_mode and not force_instrumental
        )
        lyrics, _ = split_generation_prompt(user_prompt)
        prompt = build_elevenlabs_prompt(
            structured_prompt,
            duration_seconds / 60 if duration_seconds is not None else None,
            clear_chinese,
            lyrics,
            require_lyrics=direct_prompt and not lyrics and not force_instrumental,
        )
        output_format = self._settings.elevenlabs_music_output_format
        sample_rate = int(output_format.removeprefix("pcm_"))
        request_body = {
            "prompt": prompt,
            "model_id": self._settings.elevenlabs_music_model_id,
            "force_instrumental": force_instrumental,
        }
        if duration_seconds is not None:
            request_body["music_length_ms"] = duration_seconds * 1000
        url = (
            f"{self._settings.elevenlabs_music_base_url}/v1/music/detailed"
            if direct_prompt
            else f"{self._settings.elevenlabs_music_base_url}/v1/music/stream"
        )
        request_diagnostic = {
            "method": "POST",
            "url": url,
            "params": {"output_format": output_format},
            "body": request_body,
        }
        update_provider_diagnostic(
            self._settings.output_dir,
            job_id,
            variation,
            request=request_diagnostic,
        )
        try:
            if len(prompt) > ELEVENLABS_PROMPT_MAX_CHARS:
                message = (
                    "ElevenLabs 音乐生成失败：完整歌词与风格 Prompt 长度为 "
                    f"{len(prompt)}，超过接口上限 {ELEVENLABS_PROMPT_MAX_CHARS}；"
                    "用户提供歌词时请缩短歌词或风格要求；自动写词时请检查生成歌词预算和标签长度。"
                )
                update_provider_diagnostic(
                    self._settings.output_dir,
                    job_id,
                    variation,
                    request=request_diagnostic,
                    validationError=message,
                )
                raise GenerationError(message)

            expected_seconds: float | None = duration_seconds
            if reference is not None:
                await self._report_progress(progress, "preparing_reference")
                reference = AudioReference(
                    await self._reference_song_id(reference.song_id),
                    reference.duration_ms,
                    reference.strength,
                )
                plan, expected_seconds = await self._reference_plan(
                    prompt, duration_seconds, reference, job_id, variation
                )
                # composition_plan excludes prompt, music_length_ms and force_instrumental.
                request_body = {
                    "composition_plan": plan,
                    "model_id": self._settings.elevenlabs_music_model_id,
                }
                request_diagnostic = {**request_diagnostic, "body": request_body}
                update_provider_diagnostic(
                    self._settings.output_dir,
                    job_id,
                    variation,
                    request=request_diagnostic,
                    audioReference={
                        "songId": reference.song_id,
                        "durationMs": reference.duration_ms,
                        "strength": reference.strength,
                    },
                )

            target = _audio_target(
                self._settings,
                f"full_song_elevenlabs_{time.time_ns()}.wav",
                job_id,
                variation,
            )
            await self._report_progress(
                progress, "generating_music", expected_audio_seconds=expected_seconds
            )
            title = None
            generated_lyrics = None
            for attempt in range(2):
                if direct_prompt:
                    # detailed 是唯一同时返回音频与 song_metadata/歌词的接口。
                    async with self._client.stream(
                        "POST",
                        url,
                        json=request_body,
                        params={"output_format": output_format},
                        headers=self._headers(),
                        timeout=self._settings.music_api_timeout_seconds,
                    ) as response:
                        if not response.is_success:
                            await response.aread()
                        response.raise_for_status()
                        metadata: dict[str, Any] = {}
                        actual_duration_seconds = await self._write_pcm_wav(
                            _detailed_music_audio(response, metadata),
                            target,
                            sample_rate,
                            expected_seconds,
                            progress=progress,
                        )
                    title, generated_lyrics = _detailed_music_text(metadata)
                    break
                async with self._client.stream(
                    "POST",
                    url,
                    json=request_body,
                    params={"output_format": output_format},
                    headers=self._headers(),
                    timeout=self._settings.music_api_timeout_seconds,
                ) as response:
                    if not response.is_success:
                        await response.aread()
                    # Only retry explicit rejections before any audio is accepted; never retry
                    # a timeout or broken audio stream, which may already have been billed.
                    if attempt == 0 and response.status_code in {402, 403, 404, 405}:
                        logger.warning(
                            "ElevenLabs stream endpoint rejected, falling back to compose "
                            "status=%s job_id=%s",
                            response.status_code,
                            job_id,
                        )
                        url = url.removesuffix("/stream")
                        update_provider_diagnostic(
                            self._settings.output_dir,
                            job_id,
                            variation,
                            streamFallbackStatus=response.status_code,
                            streamFallbackResponse=_response_diagnostic(response),
                            request={**request_diagnostic, "url": url},
                        )
                        continue
                    response.raise_for_status()
                    actual_duration_seconds = await self._write_pcm_wav(
                        response.aiter_bytes(),
                        target,
                        sample_rate,
                        expected_seconds,
                        progress=progress,
                    )
                    break
            else:
                raise GenerationError("ElevenLabs 音乐生成失败：未能取得音频。")
            return MusicResult(
                target,
                {
                    "provider": "elevenlabs_music",
                    "modelId": self._settings.elevenlabs_music_model_id,
                    "outputFormat": output_format,
                    "mode": "prompt_pcm_wav",
                    "endpoint": url,
                    "clearChineseVocalMode": clear_chinese,
                    "durationSeconds": actual_duration_seconds,
                    **(
                        {"audioReferenceId": reference.song_id, "mode": "reference_plan_pcm_wav"}
                        if reference is not None
                        else {}
                    ),
                },
                title=title,
                lyrics=generated_lyrics,
            )
        except GenerationError:
            raise
        except httpx.HTTPStatusError as exc:
            error_type = (
                ProviderGlobalError
                if exc.response.status_code in GLOBAL_PROVIDER_HTTP_STATUSES
                else GenerationError
            )
            raise error_type(self._failure_message(exc)) from exc
        except (httpx.HTTPError, ValueError) as exc:
            raise GenerationError(f"ElevenLabs 音乐生成失败：{_http_failure_message(exc)}") from exc

    @staticmethod
    async def _report_progress(
        progress: ProviderProgressCallback | None,
        stage: str,
        *,
        received_audio_seconds: float | None = None,
        expected_audio_seconds: float | None = None,
    ) -> None:
        if progress is None:
            return
        try:
            await progress(
                stage,
                None,
                None,
                received_audio_seconds=received_audio_seconds,
                expected_audio_seconds=expected_audio_seconds,
            )
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.warning("ElevenLabs progress report failed stage=%s", stage, exc_info=True)

    @staticmethod
    async def _write_pcm_wav(
        chunks: AsyncIterator[bytes],
        target: Path,
        sample_rate: int,
        expected_duration_seconds: float | None,
        *,
        progress: ProviderProgressCallback | None = None,
    ) -> float:
        if expected_duration_seconds is not None and expected_duration_seconds <= 0:
            raise GenerationError("ElevenLabs 音乐生成失败：目标时长必须大于 0 秒。")
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_name(f"{target.name}.{time.time_ns()}.part")
        written = 0
        bytes_per_second = sample_rate * ELEVENLABS_PCM_CHANNELS * ELEVENLABS_PCM_SAMPLE_WIDTH
        last_reported_bytes = 0
        last_reported_at = None

        async def report_received() -> None:
            nonlocal last_reported_bytes, last_reported_at
            if progress:
                await ElevenLabsMusicProvider._report_progress(
                    progress,
                    "receiving_audio",
                    received_audio_seconds=round(written / bytes_per_second, 3),
                    expected_audio_seconds=expected_duration_seconds,
                )
                last_reported_bytes = written
                last_reported_at = time.monotonic()

        try:
            # write_stream_atomically cannot write RIFF sizes; wave patches them on close
            # before the completed temporary file is atomically renamed.
            with wave.open(str(temporary), "wb") as audio:
                audio.setparams(
                    (
                        ELEVENLABS_PCM_CHANNELS,
                        ELEVENLABS_PCM_SAMPLE_WIDTH,
                        sample_rate,
                        0,
                        "NONE",
                        "not compressed",
                    )
                )
                async for chunk in chunks:
                    if chunk:
                        written += len(chunk)
                        audio.writeframesraw(chunk)
                        if progress and (
                            last_reported_at is None or time.monotonic() - last_reported_at >= 1
                        ):
                            await report_received()
            if written == 0:
                raise GenerationError("ElevenLabs 音乐生成接口返回空音频。")
            frame_size = ELEVENLABS_PCM_CHANNELS * ELEVENLABS_PCM_SAMPLE_WIDTH
            if written % frame_size:
                raise GenerationError("ElevenLabs 音乐生成接口返回了不完整的 PCM 音频帧。")
            actual_duration_seconds = written / (sample_rate * frame_size)
            if (
                expected_duration_seconds is None
                and actual_duration_seconds < ELEVENLABS_AUTO_MIN_DURATION_SECONDS
            ):
                raise GenerationError(
                    "ElevenLabs 音乐生成接口返回的自动时长音频过短："
                    f"至少需要 {ELEVENLABS_AUTO_MIN_DURATION_SECONDS} 秒，"
                    f"实际 {actual_duration_seconds:.3f} 秒。"
                )
            # ponytail: ratio only catches severe mismatch; use provider metadata when available.
            if expected_duration_seconds is not None and not (
                0.6 <= actual_duration_seconds / expected_duration_seconds <= 1.6
            ):
                raise GenerationError(
                    "ElevenLabs 音乐生成接口返回的 PCM 时长异常："
                    f"期望 {expected_duration_seconds:g} 秒，"
                    f"实际 {actual_duration_seconds:.3f} 秒。"
                )
            temporary.replace(target)
            if written != last_reported_bytes:
                await report_received()
            return round(actual_duration_seconds, 3)
        finally:
            temporary.unlink(missing_ok=True)

    @staticmethod
    def _failure_message(error: httpx.HTTPStatusError) -> str:
        detail = _http_failure_message(error)
        status = error.response.status_code
        if status == 401:
            return f"ElevenLabs 音乐生成失败：{detail}。请检查 API Key 的 music_generation 权限。"
        if status == 402:
            return f"ElevenLabs 音乐生成失败：{detail}。当前套餐可能不支持 Music API。"
        if status == 403:
            return f"ElevenLabs 音乐生成失败：{detail}。请检查账号套餐、区域和 Music API 权限。"
        return f"ElevenLabs 音乐生成失败：{detail}"


DEFAULT_SUNO_MODELS = (
    "v4.5-all",
    "chirp-v4-5-all",
    "chirp-v4-5",
    "chirp-v4",
    "chirp-v3-5",
    "chirp-v3-0",
)


class SunoMusicProvider:
    def __init__(self, settings: Settings, client: httpx.AsyncClient) -> None:
        self._settings = settings
        self._client = client

    def _headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self._settings.music_api_key:
            headers["Authorization"] = f"Bearer {self._settings.music_api_key.get_secret_value()}"
        return headers

    def _models(self) -> list[str]:
        configured = [
            item.strip() for item in self._settings.suno_model_fallbacks.split(",") if item.strip()
        ]
        values = [self._settings.suno_model, *configured]
        if self._settings.suno_use_default_model_fallbacks:
            values.extend(DEFAULT_SUNO_MODELS)
        return list(dict.fromkeys(value for value in values if value))

    async def generate(
        self,
        structured_prompt: str,
        duration_seconds: int | None,
        user_prompt: str,
        *,
        variation: int = 0,
        progress: ProviderProgressCallback | None = None,
        job_id: str | None = None,
    ) -> MusicResult:
        if not self._settings.music_api_base_url:
            raise GenerationError(
                "音乐生成失败：MUSIC_PROVIDER=suno_api 时必须配置 MUSIC_API_BASE_URL。"
            )
        duration_minutes = duration_seconds / 60
        failures: list[str] = []
        data: Any = None
        for model in self._models():
            try:
                response = await self._client.post(
                    f"{self._settings.music_api_base_url}{self._settings.music_generate_path}",
                    json={
                        "prompt": structured_prompt,
                        "duration": duration_minutes,
                        "duration_minutes": duration_minutes,
                        "make_instrumental": self._settings.suno_make_instrumental,
                        "model": model,
                        "wait_audio": False,
                    },
                    headers=self._headers(),
                    timeout=self._settings.suno_create_timeout_seconds,
                )
                response.raise_for_status()
                data = response.json()
                break
            except (httpx.HTTPError, ValueError) as exc:
                failures.append(f"{model}: {_http_failure_message(exc)}")
        if data is None:
            raise GenerationError(f"Suno 音乐生成失败：无法创建任务。{' | '.join(failures)}")

        audio_url = extract_suno_audio_url(data)
        if audio_url:
            path = await self._download(
                audio_url, f"full_song_suno_{time.time_ns()}.mp3", job_id, variation
            )
            return MusicResult(path, {"provider": "suno_api", "mode": "direct_audio_url"})
        clip_ids = extract_suno_clip_ids(data)
        if not clip_ids:
            raise GenerationError("Suno 音乐生成失败：suno-api 未返回 clip id。")
        path = await self._poll(clip_ids, job_id, variation)
        return MusicResult(path, {"provider": "suno_api", "mode": "polling", "clipIds": clip_ids})

    async def _poll(self, clip_ids: list[str], job_id: str | None, variation: int) -> Path:
        ids = ",".join(clip_ids)
        url = f"{self._settings.music_api_base_url}{self._settings.music_status_path}"
        for _ in range(self._settings.music_max_poll_attempts):
            await asyncio.sleep(self._settings.music_poll_interval_seconds)
            response = await self._client.get(
                url,
                params={"ids": ids},
                headers=self._headers(),
                timeout=self._settings.music_api_timeout_seconds,
            )
            response.raise_for_status()
            audio_url = extract_suno_audio_url(response.json())
            if audio_url:
                return await self._download(
                    audio_url, f"full_song_suno_{clip_ids[0]}.mp3", job_id, variation
                )
        raise GenerationError(f"suno-api 生成任务超时，ids={ids}")

    async def _download(self, url: str, name: str, job_id: str | None, variation: int) -> Path:
        return await download_audio(
            self._client,
            url,
            _audio_target(self._settings, name, job_id, variation),
            timeout=self._settings.music_download_timeout_seconds,
            trusted_local_root=PROJECT_ROOT,
        )


class MiniMaxMusicProvider:
    """Client for the self-hosted MiniMax Music 3 service."""

    def __init__(
        self,
        settings: Settings,
        client: httpx.AsyncClient,
    ) -> None:
        self._settings = settings
        self._client = client

    async def generate(
        self,
        structured_prompt: str,
        duration_seconds: int | None,
        user_prompt: str,
        *,
        variation: int = 0,
        progress: ProviderProgressCallback | None = None,
        job_id: str | None = None,
    ) -> MusicResult:
        if not self._settings.minimax_base_url:
            raise ProviderGlobalError("MiniMax 音乐生成失败：MINIMAX_BASE_URL 不能为空。")
        # MiniMax decides its own natural duration; fixed values from shared callers are ignored.
        del duration_seconds
        try:
            lyrics, _ = split_generation_prompt(user_prompt)
            if not lyrics:
                raise GenerationError("MiniMax 音乐生成失败：缺少歌词。")
            lyrics = _wrap_long_lyric_lines(lyrics)
            if len(lyrics) < 10:
                raise GenerationError("MiniMax 音乐生成失败：生成的歌词不足 10 个字符。")
            target = _audio_target(
                self._settings,
                f"full_song_minimax_{time.time_ns()}.wav",
                job_id,
                variation,
            )
            request_body = {
                "model": self._settings.minimax_model,
                "input": lyrics,
                "instructions": structured_prompt,
                "seed": self._settings.minimax_seed + variation,
                "num_inference_steps": self._settings.minimax_num_inference_steps,
                "response_format": "wav",
                "stream": False,
            }
            if job_id:
                request_body["jobId"] = uuid5(NAMESPACE_URL, f"{job_id}:{variation}").hex
            request_body["instructions"] += (
                "\n[Lyric Fidelity: Sing every supplied non-tag lyric line exactly once, "
                "verbatim, and in order. Never omit, repeat, paraphrase, invent, or replace "
                "any lyric words.]"
                "\n[Timing: Choose the natural complete song duration. Sing every supplied "
                "lyric without cutting off the ending, then stop all vocals and finish with "
                "only a short natural outro.]"
                "\n[Vocal Ending: Never fill unused time with repeated or invented vocals, "
                "humming, chants, ad-libs, or an extended instrumental outro.]"
            )
            url = f"{self._settings.minimax_base_url}{MINIMAX_GENERATE_PATH}"
            request_diagnostic = {
                "method": "POST",
                "url": url,
                "timeoutSeconds": self._settings.minimax_timeout_seconds,
                "body": request_body,
            }
            update_provider_diagnostic(
                self._settings.output_dir,
                job_id,
                variation,
                provider="minimax_music",
                request=request_diagnostic,
            )
            response = await self._client.post(
                url,
                json=request_body,
                timeout=httpx.Timeout(self._settings.minimax_timeout_seconds, connect=10),
            )
            create_response = _response_diagnostic(response)
            update_provider_diagnostic(
                self._settings.output_dir,
                job_id,
                variation,
                createResponse=create_response,
            )
            response.raise_for_status()
            create_body = create_response["body"]
            remote_id = create_body.get("jobId") if isinstance(create_body, dict) else None
            if not isinstance(remote_id, str) or not remote_id:
                raise GenerationError("MiniMax 未返回 jobId。")

            job_url = f"{url}/{quote(str(remote_id), safe='')}"
            started = time.monotonic()
            try:
                while True:
                    remaining = self._settings.minimax_timeout_seconds - (
                        time.monotonic() - started
                    )
                    if remaining <= 0:
                        raise GenerationError("MiniMax 音乐生成任务超时。")
                    response = await self._client.get(job_url, timeout=min(30, remaining))
                    status_response = _response_diagnostic(response)
                    update_provider_diagnostic(
                        self._settings.output_dir,
                        job_id,
                        variation,
                        statusResponse=status_response,
                    )
                    response.raise_for_status()
                    state = status_response["body"]
                    if not isinstance(state, dict):
                        raise GenerationError("MiniMax 返回的任务状态不是 JSON 对象。")
                    step, total = state.get("step"), state.get("totalSteps")
                    if step is not None and (type(step) is not int or step < 0):
                        raise GenerationError("MiniMax 返回无效 step。")
                    if total is not None and (type(total) is not int or total < 0):
                        raise GenerationError("MiniMax 返回无效 totalSteps。")
                    if progress:
                        await progress(state.get("stage") or "generating_music", step, total)
                    if state["status"] == "succeeded":
                        break
                    if state["status"] in {"failed", "cancelled"}:
                        raise GenerationError(state.get("error") or "MiniMax 音乐生成失败。")
                    if state["status"] not in {"pending", "running"}:
                        raise GenerationError("MiniMax 返回未知任务状态。")
                    await asyncio.sleep(min(self._settings.music_poll_interval_seconds, remaining))
            except (asyncio.CancelledError, Exception):
                try:
                    await self._client.delete(job_url, timeout=10)
                except httpx.HTTPError:
                    pass
                raise
            try:
                async with self._client.stream(
                    "GET",
                    f"{job_url}/audio",
                    timeout=httpx.Timeout(self._settings.minimax_timeout_seconds, connect=10),
                ) as response:
                    if not response.is_success:
                        await response.aread()
                    audio_response = {
                        "statusCode": response.status_code,
                        "headers": dict(response.headers),
                    }
                    if not response.is_success:
                        audio_response["body"] = response.text
                    update_provider_diagnostic(
                        self._settings.output_dir,
                        job_id,
                        variation,
                        audioResponse=audio_response,
                    )
                    response.raise_for_status()
                    await write_stream_atomically(
                        response.aiter_bytes(), target, "MiniMax Music 3 服务返回空音频。"
                    )
                with wave.open(str(target), "rb") as audio:
                    actual_duration_seconds = round(audio.getnframes() / audio.getframerate(), 3)
                update_provider_diagnostic(
                    self._settings.output_dir,
                    job_id,
                    variation,
                    actualDurationSeconds=actual_duration_seconds,
                )
            finally:
                # The durable local copy belongs to history; release the remote temporary WAV.
                try:
                    await self._client.delete(job_url, timeout=10)
                except httpx.HTTPError:
                    pass
            debug = {
                "provider": "minimax_music",
                "modelId": self._settings.minimax_model,
                "mode": "self_hosted_wav",
                "jobId": remote_id,
                "step": step,
                "totalSteps": total,
                "durationSeconds": actual_duration_seconds,
                "seed": self._settings.minimax_seed + variation,
                "numInferenceSteps": self._settings.minimax_num_inference_steps,
                "request": request_diagnostic,
                "createResponse": create_response,
                "statusResponse": status_response,
                "audioResponse": audio_response,
            }
            return MusicResult(target, debug)
        except GenerationError:
            raise
        except httpx.ConnectError as exc:
            raise GenerationError(
                "MiniMax Music 3 服务无法直连："
                f"{self._settings.minimax_base_url}{MINIMAX_GENERATE_PATH}。"
                "请求已绕过系统代理；请确认服务监听地址、端口和防火墙配置。"
            ) from exc
        except httpx.HTTPStatusError as exc:
            error_type = (
                ProviderGlobalError
                if exc.response.status_code in GLOBAL_PROVIDER_HTTP_STATUSES
                else GenerationError
            )
            raise error_type(f"MiniMax Music 3 服务调用失败：{_http_failure_message(exc)}") from exc
        except (httpx.HTTPError, ValueError, wave.Error) as exc:
            raise GenerationError(
                f"MiniMax Music 3 服务调用失败：{_http_failure_message(exc)}"
            ) from exc


class MurekaMusicProvider:
    """Mureka song API: upload reference → /v1/song/generate → poll /v1/song/query/{id}.

    https://platform.mureka.ai/docs/api/operations/post-v1-song-generate.html
    """

    def __init__(self, settings: Settings, client: httpx.AsyncClient) -> None:
        self._settings = settings
        self._client = client

    def _headers(self) -> dict[str, str]:
        secret = self._settings.mureka_api_key
        if secret is None:
            raise ProviderGlobalError("Mureka 音乐生成失败：缺少 MUREKA_API_KEY 环境变量。")
        return {"Authorization": f"Bearer {secret.get_secret_value()}"}

    async def _generate_lyrics(self, prompt: str) -> tuple[str | None, str]:
        # 收起页直传没有歌词；Mureka 的歌词接口会同时给出歌名。
        url = f"{self._settings.mureka_base_url}/v1/lyrics/generate"
        try:
            response = await self._client.post(
                url,
                json={"prompt": prompt},
                headers=self._headers(),
                timeout=self._settings.music_api_timeout_seconds,
            )
            response.raise_for_status()
            data = response.json()
        except httpx.HTTPStatusError as exc:
            error_type = (
                ProviderGlobalError
                if exc.response.status_code in GLOBAL_PROVIDER_HTTP_STATUSES
                else GenerationError
            )
            raise error_type(f"Mureka 歌词生成失败：{_http_failure_message(exc)}") from exc
        except (httpx.HTTPError, ValueError) as exc:
            raise GenerationError(f"Mureka 歌词生成失败：{_http_failure_message(exc)}") from exc
        lyrics = data.get("lyrics") if isinstance(data, dict) else None
        title = data.get("title") if isinstance(data, dict) else None
        if not isinstance(lyrics, str) or not lyrics.strip():
            raise GenerationError("Mureka 歌词生成失败：接口未返回歌词。")
        return (
            title.strip() if isinstance(title, str) and title.strip() else None,
            lyrics.strip(),
        )

    async def _reference_file_id(self, reference_id: str) -> str:
        return await remote_reference_id(
            self._settings.output_dir,
            reference_id,
            "mureka_music",
            f"{self._settings.mureka_base_url}:{self._headers()['Authorization']}",
            self._upload_local_reference,
        )

    async def _upload_local_reference(self, entry: dict[str, Any]) -> str:
        audio = entry.get("audio")
        if not isinstance(audio, str) or not audio:
            raise GenerationError("参考音频不存在或已删除，请重新选择。")
        directory = references_dir(self._settings.output_dir)
        path = directory / Path(audio).name
        require_readable_file(path, "参考音频文件不可读")
        # purpose=reference accepts mp3/m4a of exactly 30 s; longer is trimmed by Mureka, shorter
        # local clips (3–30 s) are looped to 30 s in a temporary file first.
        looped = None
        try:
            if float(entry["durationSeconds"]) < REFERENCE_MAX_SECONDS:
                looped = directory / f".mureka-{reference_audio_name()}"
                try:
                    await loop_reference_audio(path, looped)
                except ValueError as exc:
                    raise GenerationError(f"Mureka 参考音频处理失败：{exc}") from exc
            try:
                with (looped or path).open("rb") as handle:
                    response = await self._client.post(
                        f"{self._settings.mureka_base_url}/v1/files/upload",
                        data={"purpose": "reference"},
                        files={"file": (path.name, handle, "audio/mpeg")},
                        headers=self._headers(),
                        timeout=self._settings.music_api_timeout_seconds,
                    )
                response.raise_for_status()
            except httpx.HTTPStatusError as exc:
                raise _reference_upload_http_error("Mureka", "MUREKA_API_KEY", exc) from exc
            except httpx.TimeoutException as exc:
                raise GenerationError("Mureka 参考音频上传失败：请求超时，请稍后重试。") from exc
            except httpx.ConnectError as exc:
                raise GenerationError("Mureka 参考音频上传失败：无法连接 Mureka 服务。") from exc
            except httpx.HTTPError as exc:
                raise GenerationError(
                    f"Mureka 参考音频上传失败：{_http_failure_message(exc)}"
                ) from exc
            try:
                data = response.json()
            except ValueError as exc:
                raise GenerationError(
                    "Mureka 参考音频上传失败：服务返回的响应不是有效 JSON。"
                ) from exc
        finally:
            if looped is not None:
                looped.unlink(missing_ok=True)
        file_id = data.get("id") if isinstance(data, dict) else None
        if not isinstance(file_id, str) or not file_id:
            raise GenerationError("Mureka 参考音频上传失败：接口未返回文件 id。")
        return file_id

    async def generate(
        self,
        structured_prompt: str,
        duration_seconds: int | None,
        user_prompt: str,
        *,
        variation: int = 0,
        progress: ProviderProgressCallback | None = None,
        job_id: str | None = None,
        reference: AudioReference | None = None,
    ) -> MusicResult:
        # Mureka has no length parameter; the song length follows the lyrics (max 5m30s).
        del duration_seconds
        lyrics, _ = split_generation_prompt(user_prompt)
        song_title = None
        if not lyrics:
            song_title, lyrics = await self._generate_lyrics(
                structured_prompt[:MUREKA_PROMPT_MAX_CHARS]
            )
        if len(lyrics) > MUREKA_LYRICS_MAX_CHARS:
            raise GenerationError(
                f"Mureka 音乐生成失败：歌词长度 {len(lyrics)} 超过接口上限 "
                f"{MUREKA_LYRICS_MAX_CHARS} 字符。"
            )
        base_url = self._settings.mureka_base_url
        try:
            request_body: dict[str, Any] = {
                "lyrics": lyrics,
                "model": self._settings.mureka_model,
                # ponytail: the API rejects longer prompts; trimming keeps the head of the
                # structured style prompt, which carries genre, mood and vocal tags.
                "prompt": structured_prompt[:MUREKA_PROMPT_MAX_CHARS],
                # n defaults to 2 and every song is billed; the orchestrator loops per variation.
                "n": 1,
            }
            if reference is not None:
                if progress:
                    await progress("preparing_reference", None, None)
                # Docs list reference_id + vocal_id and prompt + vocal_id as combinations;
                # prompt + reference_id is unverified and needs a live check.
                request_body["reference_id"] = await self._reference_file_id(reference.song_id)
            url = f"{base_url}/v1/song/generate"
            request_diagnostic = {"method": "POST", "url": url, "body": request_body}
            update_provider_diagnostic(
                self._settings.output_dir,
                job_id,
                variation,
                provider="mureka_music",
                request=request_diagnostic,
            )
            response = await self._client.post(
                url,
                json=request_body,
                headers=self._headers(),
                timeout=self._settings.music_api_timeout_seconds,
            )
            create_response = _response_diagnostic(response)
            update_provider_diagnostic(
                self._settings.output_dir, job_id, variation, createResponse=create_response
            )
            response.raise_for_status()
            task = create_response["body"]
            task_id = task.get("id") if isinstance(task, dict) else None
            if not isinstance(task_id, str) or not task_id:
                raise GenerationError("Mureka 音乐生成失败：接口未返回任务 id。")

            query_url = f"{base_url}/v1/song/query/{quote(task_id, safe='')}"
            started = time.monotonic()
            while True:
                status_name = str(task.get("status") or "")
                if progress:
                    await progress(
                        "generating_music"
                        if status_name in {"running", "succeeded"}
                        else "waiting_music",
                        None,
                        None,
                    )
                if status_name == "succeeded":
                    break
                if status_name in {"failed", "timeouted", "cancelled"}:
                    reason = task.get("failed_reason") or status_name
                    raise GenerationError(f"Mureka 音乐生成失败：{reason}")
                remaining = self._settings.mureka_timeout_seconds - (time.monotonic() - started)
                if remaining <= 0:
                    raise GenerationError(f"Mureka 音乐生成任务超时，task_id={task_id}。")
                await asyncio.sleep(min(self._settings.music_poll_interval_seconds, remaining))
                response = await self._client.get(
                    query_url, headers=self._headers(), timeout=min(30, max(remaining, 1))
                )
                status_response = _response_diagnostic(response)
                update_provider_diagnostic(
                    self._settings.output_dir, job_id, variation, statusResponse=status_response
                )
                response.raise_for_status()
                task = status_response["body"]
                if not isinstance(task, dict):
                    raise GenerationError("Mureka 返回的任务状态不是 JSON 对象。")

            choices = task.get("choices")
            choice = choices[0] if isinstance(choices, list) and choices else None
            if not isinstance(choice, dict):
                raise GenerationError("Mureka 音乐生成失败：任务成功但未返回歌曲。")
            # WAV keeps the stem separator input lossless; mp3 `url` is the documented fallback.
            if choice.get("wav_url"):
                audio_url, suffix = choice["wav_url"], ".wav"
            else:
                audio_url, suffix = choice.get("url"), ".mp3"
            if not isinstance(audio_url, str) or not audio_url.startswith("https://"):
                raise GenerationError("Mureka 音乐生成失败：歌曲下载地址无效。")
            target = _audio_target(
                self._settings, f"full_song_mureka_{task_id}{suffix}", job_id, variation
            )
            path = await download_audio(
                self._client,
                audio_url,
                target,
                timeout=self._settings.music_download_timeout_seconds,
                trusted_local_root=PROJECT_ROOT,
            )
            duration_ms = choice.get("duration")
            return MusicResult(
                path,
                {
                    "provider": "mureka_music",
                    "modelId": task.get("model") or self._settings.mureka_model,
                    "mode": "polling",
                    "taskId": task_id,
                    "songId": choice.get("id"),
                    "styleTruncated": len(structured_prompt) > MUREKA_PROMPT_MAX_CHARS,
                    **(
                        {"durationSeconds": duration_ms / 1000}
                        if isinstance(duration_ms, (int, float))
                        else {}
                    ),
                    **({"audioReferenceId": reference.song_id} if reference is not None else {}),
                },
                title=song_title,
                lyrics=lyrics,
            )
        except GenerationError:
            raise
        except httpx.HTTPStatusError as exc:
            error_type = (
                ProviderGlobalError
                if exc.response.status_code in GLOBAL_PROVIDER_HTTP_STATUSES
                else GenerationError
            )
            raise error_type(f"Mureka 音乐生成失败：{_http_failure_message(exc)}") from exc
        except (httpx.HTTPError, ValueError) as exc:
            raise GenerationError(f"Mureka 音乐生成失败：{_http_failure_message(exc)}") from exc


def create_music_provider(
    settings: Settings,
    client: httpx.AsyncClient,
    direct_client: httpx.AsyncClient | None = None,
    provider: str | None = None,
) -> MusicProvider:
    if settings.music_api_mode == "mock":
        return MockMusicProvider(settings)
    selected = provider or settings.music_provider
    if selected == "elevenlabs_music":
        elevenlabs_client = (
            direct_client
            if settings.elevenlabs_bypass_global_proxy and direct_client is not None
            else client
        )
        return ElevenLabsMusicProvider(settings, elevenlabs_client)
    if selected == "suno_api":
        return SunoMusicProvider(settings, client)
    if selected == "minimax_music":
        return MiniMaxMusicProvider(settings, direct_client or client)
    if selected == "mureka_music":
        mureka_client = (
            direct_client
            if settings.mureka_bypass_global_proxy and direct_client is not None
            else client
        )
        return MurekaMusicProvider(settings, mureka_client)
    return GenericMusicProvider(settings, client)
