from __future__ import annotations

import asyncio
import logging
import os
import shutil
import tempfile
import weakref
from collections.abc import AsyncIterator
from pathlib import Path
from urllib.parse import quote, unquote, urlsplit

import httpx

from app.core.errors import GenerationError

logger = logging.getLogger(__name__)
PLAYBACK_ENCODE_TIMEOUT_SECONDS = 120
# 试听编码是纯 CPU 的 ffmpeg 子进程：分轨任务的容量配额在编码前就让给了下一个 BS-RoFormer
# 任务，所以这里必须自己限并发，否则 4–6 个 ffmpeg 会和 BS-RoFormer 抢同一批核。
PLAYBACK_ENCODE_CONCURRENCY = os.cpu_count() or 2
_encoder_gates: weakref.WeakKeyDictionary[asyncio.AbstractEventLoop, asyncio.Semaphore] = (
    weakref.WeakKeyDictionary()
)


def _encoder_gate() -> asyncio.Semaphore:
    """按事件循环懒建编码闸门（模块级信号量跨循环复用会绑定到已关闭的循环）。"""
    loop = asyncio.get_running_loop()
    gate = _encoder_gates.get(loop)
    if gate is None:
        gate = asyncio.Semaphore(PLAYBACK_ENCODE_CONCURRENCY)
        _encoder_gates[loop] = gate
    return gate


def _playback_name(source: Path) -> str:
    version = source.stat()
    return f"{source.stem}.playback-{version.st_mtime_ns}-{version.st_size}.mp3"


async def make_playback_mp3(source: Path, output_dir: Path) -> str | None:
    """Encode one WAV version; a failed preview never changes the source or its result."""
    if source.suffix.lower() != ".wav":
        return None
    async with _encoder_gate():
        return await _encode_playback_mp3(source, output_dir)


async def _encode_playback_mp3(source: Path, output_dir: Path) -> str | None:
    temporary = None
    process = None
    try:
        # Local import breaks the audio_files <-> stems import cycle.
        from app.services.stems import prepare_ffmpeg_environment

        with source.open("rb") as audio:
            header = audio.read(12)
        if header[:4] != b"RIFF" or header[8:12] != b"WAVE":
            return None
        playback_dir = source.parent / "playtrack"
        playback_dir.mkdir(exist_ok=True)
        target = playback_dir / _playback_name(source)
        if target.is_file() and target.stat().st_size:
            return target.relative_to(output_dir).as_posix()
        descriptor, name = tempfile.mkstemp(
            dir=playback_dir, prefix=f".{source.stem}.", suffix=".mp3"
        )
        os.close(descriptor)
        temporary = Path(name)
        process = await asyncio.create_subprocess_exec(
            "ffmpeg",
            "-v",
            "error",
            "-nostdin",
            "-y",
            "-i",
            str(source),
            "-codec:a",
            "libmp3lame",
            "-q:a",
            "2",
            "-f",
            "mp3",
            str(temporary),
            env=prepare_ffmpeg_environment(),
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.PIPE,
        )
        _, stderr = await asyncio.wait_for(
            process.communicate(), timeout=PLAYBACK_ENCODE_TIMEOUT_SECONDS
        )
        if process.returncode or not temporary.is_file() or not temporary.stat().st_size:
            raise RuntimeError(stderr.decode(errors="replace").strip() or "empty MP3")
        if target.name != _playback_name(source):
            return None
        temporary.replace(target)
        # Older previews may still be named in an undo stash; restoring then falls back to WAV.
        for old in playback_dir.glob(f"{source.stem}.playback-*.mp3"):
            if old != target:
                old.unlink(missing_ok=True)
        return target.relative_to(output_dir).as_posix()
    except asyncio.CancelledError:
        raise
    except Exception:
        logger.warning("playback encoding skipped source=%s", source, exc_info=True)
        return None
    finally:
        # 临时文件先清掉再回收子进程：取消（PATCH 在 preview 阶段就是取消编码器）会在下面的
        # await 上再抛一次 CancelledError，把清理留在后面就会在 playtrack/ 里留下残渣。
        if temporary is not None:
            temporary.unlink(missing_ok=True)
        if process is not None and process.returncode is None:
            process.kill()
            await process.wait()


def playback_matches(source: Path, preview: Path) -> bool:
    if not source.is_file() or not preview.is_file():
        return False
    return (
        preview.stat().st_size > 0
        and preview.name == _playback_name(source)
        and preview.parent == source.parent / "playtrack"
    )


def existing_playback_mp3(source: Path) -> Path | None:
    """Reuse a current preview, or the original MP3 of a legacy song."""
    if not source.is_file() or not source.stat().st_size:
        return None
    if source.suffix.lower() == ".mp3":
        return source
    if source.suffix.lower() == ".wav":
        preview = source.parent / "playtrack" / _playback_name(source)
        if playback_matches(source, preview):
            return preview
    return None


def detect_audio_content_type(path: Path, *, header: bytes | None = None) -> str:
    try:
        if header is None:
            with path.open("rb") as audio:
                header = audio.read(12)
        if header[:4] == b"RIFF" and header[8:12] == b"WAVE":
            return "audio/wav"
        if header[:3] == b"ID3" or (len(header) >= 2 and header[0] == 0xFF):
            return "audio/mpeg"
    except OSError:
        pass
    return "audio/wav" if path.suffix.lower() == ".wav" else "audio/mpeg"


def require_readable_file(path: Path, message: str) -> None:
    if not path.is_file() or path.stat().st_size <= 0:
        raise GenerationError(f"{message}: {path}")


def build_public_audio_url(base_url: str, relative_path: str | Path) -> str:
    encoded = "/".join(
        quote(part, safe="") for part in Path(relative_path).parts if part not in {"/", ""}
    )
    return f"{base_url.rstrip('/')}/output/{encoded}"


def output_path_from_url(audio_url: str, output_dir: Path) -> Path:
    parsed = urlsplit(audio_url)
    path = unquote(parsed.path)
    if "/output/" in path:
        path = path.split("/output/", 1)[1]
    elif parsed.scheme or path.startswith("/"):
        raise ValueError("Not an output URL")
    root = output_dir.resolve()
    target = (root / path).resolve()
    target.relative_to(root)
    return target


async def write_stream_atomically(
    chunks: AsyncIterator[bytes],
    target_path: Path,
    empty_message: str,
) -> Path:
    target_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = target_path.with_name(f"{target_path.name}.{os.getpid()}.part")
    written = 0
    try:
        # Writes are small sequential chunks to a local file. Keeping the handle in
        # this coroutine avoids thread-pool cancellation deadlocks while streaming.
        with temporary_path.open("wb") as output:
            async for chunk in chunks:
                if not chunk:
                    continue
                written += len(chunk)
                output.write(chunk)
        if written == 0:
            raise GenerationError(empty_message)
        temporary_path.replace(target_path)
    finally:
        temporary_path.unlink(missing_ok=True)
    return target_path


async def download_audio(
    client: httpx.AsyncClient,
    audio_url: str,
    target_path: Path,
    *,
    timeout: float,
    trusted_local_root: Path,
) -> Path:
    if not audio_url.lower().startswith(("http://", "https://")):
        trusted_local_root = trusted_local_root.resolve()
        local_path = Path(audio_url).expanduser()
        if not local_path.is_absolute():
            local_path = trusted_local_root / local_path
        local_path = local_path.resolve()
        try:
            local_path.relative_to(trusted_local_root)
        except ValueError as exc:
            raise GenerationError("音乐生成接口返回了不受信任的本地音频路径。") from exc
        require_readable_file(local_path, "音乐生成接口返回的本地音频文件不可读")
        return local_path

    try:
        async with client.stream(
            "GET", audio_url, timeout=timeout, follow_redirects=True
        ) as response:
            response.raise_for_status()
            return await write_stream_atomically(
                response.aiter_bytes(), target_path, "音乐生成音频下载结果为空。"
            )
    except httpx.HTTPError as exc:
        raise GenerationError(f"音乐生成音频下载失败：{exc}") from exc


async def ensure_file_under_root(
    source: Path,
    root: Path,
    target_name: str,
) -> Path:
    source = source.resolve()
    root = root.resolve()
    require_readable_file(source, "完整音频不可读")
    try:
        source.relative_to(root)
        return source
    except ValueError:
        target = root / target_name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        return target
