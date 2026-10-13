from __future__ import annotations

import asyncio
import importlib.util
import os
import re
import shutil
import sys
import tempfile
import wave
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol
from uuid import uuid4

from app.core.config import Settings
from app.core.errors import GenerationError
from app.services.audio_files import require_readable_file

STEM_NAMES = ("vocal", "drums", "bass", "guitar", "piano", "other")
BS_ROFORMER_MODEL = "roformer-model-bs-roformer-sw-by-jarredou"
BS_ROFORMER_SAMPLE_RATE = 44_100


def is_vocal_stem(name: str) -> bool:
    return name.lower().rsplit(":", 1)[-1] in {"vocal", "vocals", "voice", "replaced"}


@dataclass(frozen=True, slots=True)
class SplitResult:
    stdout: str
    stderr: str
    duration_ms: int
    files: dict[str, str]


class StemSeparator(Protocol):
    async def split(self, input_path: Path, output_dir: Path) -> SplitResult: ...


def require_bs_roformer_runtime(device: str) -> None:
    missing = [
        name
        for name in ("numpy", "torch", "bs_roformer", "soundfile")
        if importlib.util.find_spec(name) is None
    ]
    if missing:
        raise GenerationError(
            f"BS-RoFormer 运行依赖缺失：{', '.join(missing)}。请执行 uv sync --frozen。"
        )
    if device.startswith("cuda"):
        import torch

        index = int(device.split(":", 1)[1]) if ":" in device else 0
        if not torch.cuda.is_available() or index >= torch.cuda.device_count():
            raise GenerationError(
                f"BS_ROFORMER_DEVICE={device} 不可用，请检查 NVIDIA 驱动和 CUDA GPU。"
            )


def build_bs_roformer_command(settings: Settings, input_dir: Path, work_dir: Path) -> list[str]:
    command = [
        sys.executable,
        "-m",
        "bs_roformer.inference",
        "--model",
        BS_ROFORMER_MODEL,
        "--models_dir",
        str(settings.bs_roformer_models_dir),
        "--input_folder",
        str(input_dir),
        "--store_dir",
        str(work_dir),
    ]
    if settings.bs_roformer_device != "auto":
        command.extend(["--device", settings.bs_roformer_device])
    return command


def prepare_ffmpeg_environment() -> dict[str, str]:
    environment = dict(os.environ)
    if shutil.which("ffmpeg") and shutil.which("ffprobe"):
        return environment
    try:
        import static_ffmpeg

        static_ffmpeg.add_paths(weak=True)
        if shutil.which("ffmpeg") and shutil.which("ffprobe"):
            environment["PATH"] = os.environ.get("PATH", "")
            return environment
    except Exception:
        pass
    raise GenerationError("音频处理需要 ffmpeg 和 ffprobe，请安装 static-ffmpeg 或系统 ffmpeg。")


def stem_output_files(input_path: Path) -> dict[str, str]:
    base_name = re.sub(r"[^\w\-. ]+", "_", input_path.stem, flags=re.UNICODE).strip(" ._")
    base_name = base_name[:160] or "music"
    return {name: f"{base_name}_{name}.wav" for name in STEM_NAMES}


def publish_stems(
    work_dir: Path, output_dir: Path, output_files: dict[str, str], expected_frames: int
) -> None:
    """Validate all six outputs, then move them to fresh (uuid-named) targets."""
    import soundfile as sf

    sources = {
        name: work_dir / f"source_{'vocals' if name == 'vocal' else name}.wav"
        for name in STEM_NAMES
    }
    for source in sources.values():
        require_readable_file(source, "BS-RoFormer 未生成完整的六轨结果")
        try:
            audio = sf.info(source)
        except RuntimeError as exc:
            raise GenerationError(f"BS-RoFormer 输出不是有效 WAV：{source.name}") from exc
        if (
            audio.format not in {"WAV", "WAVEX", "RF64"}
            or audio.samplerate != BS_ROFORMER_SAMPLE_RATE
            or audio.channels != 2
            or audio.frames != expected_frames
            or audio.frames <= 0
        ):
            raise GenerationError(f"BS-RoFormer 输出的格式或时长不匹配：{source.name}")
    published: list[Path] = []
    try:
        for name, source in sources.items():
            target = output_dir / output_files[name]
            source.replace(target)
            published.append(target)
    except BaseException:
        for target in published:
            target.unlink(missing_ok=True)
        raise


class BSRoFormerStemSeparator:
    def __init__(self, settings: Settings) -> None:
        self._settings = settings

    async def _run(
        self, command: list[str], environment: dict[str, str], deadline: float, label: str
    ) -> tuple[str, str]:
        remaining = deadline - asyncio.get_running_loop().time()
        if remaining <= 0:
            raise GenerationError(
                f"BS-RoFormer 拆轨超时（{self._settings.bs_roformer_timeout_seconds:g} 秒）。"
            )
        process = await asyncio.create_subprocess_exec(
            *command,
            env=environment,
            stdin=asyncio.subprocess.DEVNULL,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        try:
            stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=remaining)
        except (asyncio.TimeoutError, asyncio.CancelledError) as exc:
            with suppress(ProcessLookupError):
                process.kill()
            await process.wait()
            if isinstance(exc, asyncio.CancelledError):
                raise
            raise GenerationError(
                f"BS-RoFormer 拆轨超时（{self._settings.bs_roformer_timeout_seconds:g} 秒）。"
            ) from exc
        stdout = stdout.decode("utf-8", errors="replace").strip()
        stderr = stderr.decode("utf-8", errors="replace").strip()
        if process.returncode != 0:
            raise GenerationError(
                f"{label}失败，退出码：{process.returncode}，stderr={stderr[-4000:]}"
            )
        return stdout, stderr

    async def split(self, input_path: Path, output_dir: Path) -> SplitResult:
        require_readable_file(input_path, "拆轨输入音频不可读")
        require_bs_roformer_runtime(self._settings.bs_roformer_device)
        output_dir.mkdir(parents=True, exist_ok=True)
        # static-ffmpeg may download binaries on first use; keep that off the event loop.
        environment = await asyncio.to_thread(prepare_ffmpeg_environment)
        environment["PYTHONUNBUFFERED"] = "1"
        started = asyncio.get_running_loop().time()
        deadline = started + self._settings.bs_roformer_timeout_seconds
        # New files keep a failed/cancelled rebuild from overwriting the current project.
        output_files = stem_output_files(Path(f"{input_path.stem[:120]}_{uuid4().hex[:12]}.wav"))
        with tempfile.TemporaryDirectory(prefix=".bs-roformer-", dir=output_dir) as temporary:
            work_dir = Path(temporary)
            input_dir = work_dir / "input"
            input_dir.mkdir()
            prepared = input_dir / "source.wav"
            await self._run(
                [
                    "ffmpeg",
                    "-v",
                    "error",
                    "-nostdin",
                    "-y",
                    "-i",
                    str(input_path),
                    "-map",
                    "0:a:0",
                    "-vn",
                    "-ac",
                    "2",
                    "-ar",
                    str(BS_ROFORMER_SAMPLE_RATE),
                    "-c:a",
                    "pcm_s16le",
                    "-f",
                    "wav",
                    str(prepared),
                ],
                environment,
                deadline,
                "分轨输入音频解码",
            )
            require_readable_file(prepared, "分轨输入音频解码结果为空")
            with wave.open(str(prepared), "rb") as audio:
                frames = audio.getnframes()
            if frames <= 0:
                raise GenerationError("分轨输入音频没有可播放的内容。")
            stdout, stderr = await self._run(
                build_bs_roformer_command(self._settings, input_dir, work_dir),
                environment,
                deadline,
                "BS-RoFormer 六轨分离",
            )
            # No await during publication: cancellation cannot leave half a set of stems.
            publish_stems(work_dir, output_dir, output_files, frames)
        return SplitResult(
            stdout, stderr, int((asyncio.get_running_loop().time() - started) * 1000), output_files
        )
