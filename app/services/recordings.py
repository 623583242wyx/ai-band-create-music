from __future__ import annotations

import asyncio
import wave
from pathlib import Path

from app.services.references import ReferenceInputError, run_audio_command
from app.services.stems import prepare_ffmpeg_environment


async def prepare_recording_audio(source: Path, target: Path, max_seconds: int) -> float:
    """Decode browser recordings into bounded PCM WAV without trusting MIME or filename."""
    await run_audio_command(
        [
            "ffmpeg",
            "-v",
            "error",
            "-nostdin",
            "-y",
            "-protocol_whitelist",
            "file,pipe",
            "-format_whitelist",
            "wav,mp3,flac,ogg,mov,matroska,webm,aac",
            "-i",
            str(source),
            "-map",
            "0:a:0",
            "-vn",
            "-t",
            str(max_seconds + 1),
            "-ac",
            "2",
            "-ar",
            "44100",
            "-c:a",
            "pcm_s16le",
            "-f",
            "wav",
            str(target),
        ],
        await asyncio.to_thread(prepare_ffmpeg_environment),
        "无法解析录音，请上传有效音频文件（WebM、MP4/M4A、WAV、MP3、OGG 等）。",
    )
    with wave.open(str(target), "rb") as audio:
        duration = audio.getnframes() / audio.getframerate()
    if duration <= 0:
        raise ReferenceInputError("录音没有可播放的音频内容。")
    if duration > max_seconds:
        raise ReferenceInputError(f"录音时长不能超过 {max_seconds} 秒。")
    return duration
