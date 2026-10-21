from __future__ import annotations

import asyncio
import math
import re
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


END_TOLERANCE_SECONDS = 0.05


def committed_recording_ids(project: dict) -> list[str]:
    if project.get("committedRecordingIds") is not None:
        return project["committedRecordingIds"] or []
    # Legacy commits have rendered stem filenames, even if a later mixConfig failed.
    return sorted(
        stem
        for stem, url in project.get("stems", {}).items()
        if stem.startswith("recording_") and re.search(r"_v[0-9]+_[0-9a-f]{12}\.wav$", url)
    )


def recording_position_editable(project: dict, stem: str) -> bool:
    if stem not in committed_recording_ids(project):
        return True
    return bool(project.get("replacedVocal")) and bool(
        project.get("recordingClips", {}).get(stem, {}).get("source")
    )


def move_recording_audio(
    source: Path,
    target: Path,
    position: float,
    clip: dict | None,
    song_duration: float,
    max_seconds: float,
    waveform: list[float] | None = None,
) -> dict:
    """Move exactly the retained PCM samples; never accumulate padding across moves."""
    with wave.open(str(source), "rb") as audio:
        rate = audio.getframerate()
        old_position = 0 if (clip or {}).get("source") else (clip or {}).get("position", 0)
        duration = (clip or {}).get("duration", audio.getnframes() / rate)
        if not all(
            math.isfinite(value) for value in (position, old_position, duration, song_duration)
        ):
            raise ReferenceInputError("请输入有效的放置位置。")
        start, frames, offset = (
            round(old_position * rate),
            round(duration * rate),
            round(position * rate),
        )
        if position < 0 or frames <= 0 or start < 0 or start + frames > audio.getnframes():
            raise ReferenceInputError("录音片段无效，请重新打开工程。")
        limit = round(min(song_duration, max_seconds) * rate)
        if offset + frames > limit:
            # Client timelines use MP3 previews, whose encoder padding can run slightly past the
            # master; snap drops within that margin to the end instead of rejecting them.
            if offset + frames - limit > round(END_TOLERANCE_SECONDS * rate):
                raise ReferenceInputError("录音超出歌曲结尾，请向左移动。")
            offset = limit - frames
            if offset < 0:
                raise ReferenceInputError("录音超出歌曲结尾，请向左移动。")
        clip_waveform = (clip or {}).get("waveform")
        if not clip_waveform and waveform:
            count = len(waveform)
            clip_waveform = waveform[
                math.floor(start / audio.getnframes() * count) : math.ceil(
                    (start + frames) / audio.getnframes() * count
                )
            ]
        audio.setpos(start)
        with wave.open(str(target), "wb") as output:
            output.setparams(audio.getparams())
            frame_bytes = audio.getnchannels() * audio.getsampwidth()
            remaining = offset
            while remaining:
                count = min(remaining, 65536)
                output.writeframesraw(bytes(count * frame_bytes))
                remaining -= count
            remaining = frames
            while remaining:
                count = min(remaining, 65536)
                data = audio.readframes(count)
                if len(data) != count * frame_bytes:
                    raise ReferenceInputError("录音文件不完整，请重新打开工程。")
                output.writeframesraw(data)
                remaining -= count
    result = {"position": offset / rate, "duration": frames / rate}
    if clip_waveform is not None:
        # Moving changes silence padding, never the retained clip's waveform bins.
        result["waveform"] = clip_waveform
    if (clip or {}).get("source"):
        result["source"] = clip["source"]
    return result


async def recording_song_duration(source: Path) -> float:
    """Read the actual master, including MP3 jobs with absent or estimated duration hints."""
    output = await run_audio_command(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            str(source),
        ],
        await asyncio.to_thread(prepare_ffmpeg_environment),
        "无法读取歌曲时长，请重新打开工程。",
        capture=True,
    )
    duration = float(output.strip())
    if not math.isfinite(duration) or duration <= 0:
        raise ReferenceInputError("无法读取歌曲时长，请重新打开工程。")
    return duration
