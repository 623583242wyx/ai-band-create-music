"""ElevenLabs 音频参考：本地预处理与已上传参考的索引。

ElevenLabs 文档（inpainting 指南）规定 conditioning 参考最长 30 秒，上传接口按一次生成计费，
所以每个文件只上传一次：先在本地裁到 30 秒并转成 MP3，上传后把 song_id 记进索引，
之后的生成直接复用。ElevenLabs 没有公开删除接口，删除只移除本地记录与试听文件。
"""

from __future__ import annotations

import asyncio
import json
import logging
import math
import os
import re
import tempfile
from contextlib import suppress
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

from app.core.errors import GenerationError

logger = logging.getLogger(__name__)

REFERENCE_DIR = "references"
REFERENCE_INDEX = "index.json"
# https://elevenlabs.io/docs/eleven-api/guides/how-to/music/inpainting
# "A conditioning reference can be at most 30 seconds (30,000ms) long."
REFERENCE_MAX_SECONDS = 30
# ponytail: ElevenLabs documents no minimum reference length; 3 s mirrors the API's
# shortest chunk/song length so near-empty clips fail here instead of after a billed upload.
REFERENCE_MIN_SECONDS = 3
REFERENCE_NAME_MAX_CHARS = 100
REFERENCE_FFMPEG_TIMEOUT_SECONDS = 120
REFERENCE_ID_PATTERN = r"^[A-Za-z0-9_-]{1,128}$"


class ReferenceInputError(ValueError):
    """The uploaded file cannot be used as an audio reference."""


def references_dir(output_dir: Path) -> Path:
    return output_dir / REFERENCE_DIR


def clean_reference_name(filename: str | None) -> str:
    name = Path(filename or "").name.strip()
    name = re.sub(r"[\x00-\x1f\x7f]", "", name)
    return name[:REFERENCE_NAME_MAX_CHARS] or "参考音频"


def load_references(output_dir: Path) -> list[dict[str, Any]]:
    path = references_dir(output_dir) / REFERENCE_INDEX
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return []
    except (OSError, ValueError):
        logger.exception("audio reference index unreadable path=%s", path)
        return []
    if not isinstance(data, list):
        return []
    return [
        item
        for item in data
        if isinstance(item, dict)
        and isinstance(item.get("id"), str)
        and re.fullmatch(REFERENCE_ID_PATTERN, item["id"])
        and isinstance(item.get("durationSeconds"), (int, float))
    ]


def find_reference(output_dir: Path, reference_id: str) -> dict[str, Any] | None:
    return next((item for item in load_references(output_dir) if item["id"] == reference_id), None)


def save_references(output_dir: Path, references: list[dict[str, Any]]) -> None:
    directory = references_dir(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(dir=directory, prefix=".index.", suffix=".json")
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(references, handle, ensure_ascii=False, indent=2)
        Path(name).replace(directory / REFERENCE_INDEX)
    finally:
        Path(name).unlink(missing_ok=True)


def new_reference_entry(
    song_id: str, name: str, duration_seconds: float, audio: str
) -> dict[str, Any]:
    return {
        "id": song_id,
        "name": name,
        "durationSeconds": duration_seconds,
        "audio": audio,
        "createdAt": datetime.now(timezone.utc).isoformat(),
    }


async def prepare_reference_audio(source: Path, target: Path) -> float:
    """Trim to the first 30 s, encode MP3 and return the encoded duration in seconds."""
    from app.services.stems import prepare_ffmpeg_environment

    environment = prepare_ffmpeg_environment()
    target.parent.mkdir(parents=True, exist_ok=True)
    await run_audio_command(
        [
            "ffmpeg",
            "-v",
            "error",
            "-nostdin",
            "-y",
            "-i",
            str(source),
            "-map",
            "0:a:0",
            "-vn",
            "-t",
            str(REFERENCE_MAX_SECONDS),
            "-ac",
            "2",
            "-ar",
            "44100",
            "-codec:a",
            "libmp3lame",
            "-b:a",
            "192k",
            "-f",
            "mp3",
            str(target),
        ],
        environment,
        "无法解析该音频文件，请上传常见格式（MP3、WAV、FLAC、M4A 等）的音频。",
    )
    output = await run_audio_command(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "json",
            str(target),
        ],
        environment,
        "无法读取音频时长。",
        capture=True,
    )
    try:
        duration = float(json.loads(output)["format"]["duration"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ReferenceInputError("无法读取音频时长。") from exc
    if not math.isfinite(duration) or duration < REFERENCE_MIN_SECONDS:
        raise ReferenceInputError(f"参考音频至少需要 {REFERENCE_MIN_SECONDS} 秒。")
    return round(min(duration, REFERENCE_MAX_SECONDS), 3)


async def loop_reference_audio(source: Path, target: Path) -> None:
    """Loop a short prepared clip to exactly REFERENCE_MAX_SECONDS.

    Mureka only accepts reference uploads of exactly 30 s, while local references may be 3–30 s.
    """
    from app.services.stems import prepare_ffmpeg_environment

    await run_audio_command(
        [
            "ffmpeg",
            "-v",
            "error",
            "-nostdin",
            "-y",
            "-stream_loop",
            "-1",
            "-i",
            str(source),
            "-t",
            str(REFERENCE_MAX_SECONDS),
            "-ac",
            "2",
            "-ar",
            "44100",
            "-codec:a",
            "libmp3lame",
            "-b:a",
            "192k",
            "-f",
            "mp3",
            str(target),
        ],
        prepare_ffmpeg_environment(),
        f"无法把参考音频补足到 {REFERENCE_MAX_SECONDS} 秒。",
    )


async def run_audio_command(
    command: list[str], environment: dict[str, str], message: str, *, capture: bool = False
) -> str:
    try:
        process = await asyncio.create_subprocess_exec(
            *command,
            env=environment,
            stdin=asyncio.subprocess.DEVNULL,
            stdout=asyncio.subprocess.PIPE if capture else asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.PIPE,
        )
    except OSError as exc:
        raise GenerationError(f"音频处理工具无法启动：{exc}") from exc
    try:
        stdout, stderr = await asyncio.wait_for(
            process.communicate(), timeout=REFERENCE_FFMPEG_TIMEOUT_SECONDS
        )
    except (asyncio.TimeoutError, asyncio.CancelledError) as exc:
        with suppress(ProcessLookupError):
            process.kill()
        await process.wait()
        if isinstance(exc, asyncio.CancelledError):
            raise
        raise ReferenceInputError("音频处理超时，请换一个较短的文件。") from exc
    if process.returncode:
        logger.warning(
            "audio reference ffmpeg failed command=%s stderr=%s",
            command[0],
            stderr.decode(errors="replace").strip()[:500],
        )
        raise ReferenceInputError(message)
    return (stdout or b"").decode(errors="replace")


def reference_audio_name() -> str:
    return f"{uuid4().hex}.mp3"
