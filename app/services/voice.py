from __future__ import annotations

import asyncio
import json
import logging
import math
import re
import tempfile
from contextlib import suppress
from pathlib import Path
from typing import Annotated
from urllib.parse import unquote, urlsplit

from fastapi import FastAPI, Form, HTTPException, Request

from app.core.config import Settings
from app.services.audio_files import output_path_from_url
from app.services.job_files import (
    capture_mix_artifact,
    invalidate_mix_artifact,
    job_song_dir,
    require_audio_revision,
    restore_mix_artifact,
    stored_path_exists,
)
from app.services.retention import retained_job
from app.services.stems import prepare_ffmpeg_environment

logger = logging.getLogger(__name__)


def build_mix_filter(input_count: int) -> str:
    labels = "[vocal]" + "".join(f"[{index}:a]" for index in range(1, input_count))
    return (
        "[0:a]pan=stereo|c0=c0|c1=c0,equalizer=f=3000:t=q:w=1:g=2.5[vocal];"
        f"{labels}amix=inputs={input_count}:duration=longest:"
        "dropout_transition=0:normalize=0[premix]"
    )


MIX_FILTER = build_mix_filter(4)
LIMIT_FILTER = (
    "volume={gain_db:.3f}dB,alimiter=limit=0.891251:attack=5:release=50:level=false:latency=true"
)


class RVCConversionError(RuntimeError):
    """Raised when the local RVC inference engine cannot convert an input file."""


class MixTimeoutError(RVCConversionError):
    """内部信号：某个 ffmpeg 步骤用光了剩余预算。

    对外统一由 :func:`mix_tracks` 报成配置的总超时，避免日志与持久化错误里出现
    "剩余 12.3 秒"这种与 RVC_MIX_TIMEOUT_SECONDS 对不上的数字。
    """


def mix_busy(job) -> bool:
    """True while a mix of this job is queued or running.

    Lives here, rather than in the mix endpoint module, because every voice and job
    endpoint has to refuse work that would race the mix's input files, and this module
    is the one both of them already import (importing the mix module from here would be
    a cycle).
    """
    if job is None:
        return False
    return getattr(job, "mix_status", None) in {"pending", "running"} or (
        getattr(job, "mix_task", None) is not None and not job.mix_task.done()
    )


class RVCEngine:
    """Lazy, concurrency-safe adapter around rvc-python."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._inference = None
        self._lock = asyncio.Lock()

    @property
    def loaded(self) -> bool:
        return self._inference is not None

    async def convert(
        self,
        input_path: Path,
        output_path: Path,
        *,
        f0_up_key: int = 0,
        f0_method: str = "rmvpe",
        index_rate: float = 0.75,
        filter_radius: int = 3,
        resample_sr: int = 0,
        rms_mix_rate: float = 1.0,
        protect: float = 0.33,
    ) -> None:
        async with self._lock:
            await asyncio.to_thread(
                self._convert_sync,
                input_path,
                output_path,
                f0_up_key=f0_up_key,
                f0_method=f0_method,
                index_rate=index_rate,
                filter_radius=filter_radius,
                resample_sr=resample_sr,
                rms_mix_rate=rms_mix_rate,
                protect=protect,
            )

    def _load_sync(self):
        if self._inference is not None:
            return self._inference

        self._validate_assets()
        try:
            import torch
            from rvc_python import infer as rvc_infer

            if self._settings.rvc_device.startswith("cuda") and not torch.cuda.is_available():
                raise RuntimeError("CUDA is unavailable; set RVC_DEVICE=cpu to use CPU inference")
            if self._settings.rvc_device.startswith("cpu"):
                torch.set_num_threads(self._settings.rvc_cpu_threads)

            # rvc-python stores its large base models beside the installed package
            # and has no directory option. Link only those assets into this
            # environment; code and mutable configs remain owned by generate.
            base_model_dir = self._settings.rvc_base_model_dir
            if base_model_dir is not None:
                _reuse_base_models(rvc_infer, base_model_dir)

            inference = rvc_infer.RVCInference(
                device=self._settings.rvc_device,
                model_path=str(self._settings.rvc_model_path),
                index_path=(
                    str(self._settings.rvc_index_path)
                    if self._settings.rvc_index_path is not None
                    else ""
                ),
                version=self._settings.rvc_model_version,
            )
            _load_hubert_safely(inference, torch)
        except Exception as exc:
            raise RVCConversionError(f"Failed to load RVC model: {exc}") from exc

        self._inference = inference
        return inference

    def _convert_sync(self, input_path: Path, output_path: Path, **params) -> None:
        inference = self._load_sync()
        try:
            rvc_params = {
                **params,
                "f0up_key": params["f0_up_key"],
                "f0method": params["f0_method"],
            }
            del rvc_params["f0_up_key"]
            del rvc_params["f0_method"]
            inference.set_params(**rvc_params)
            inference.infer_file(str(input_path), str(output_path))
        except Exception as exc:
            raise RVCConversionError(f"RVC inference failed: {exc}") from exc

        if not output_path.is_file() or output_path.stat().st_size == 0:
            raise RVCConversionError("RVC inference did not produce an output file")

    def _validate_assets(self) -> None:
        if not self._settings.rvc_model_path.is_file():
            raise RuntimeError(f"RVC model not found: {self._settings.rvc_model_path}")
        index_path = self._settings.rvc_index_path
        if index_path is not None and not index_path.is_file():
            raise RuntimeError(f"RVC index not found: {index_path}")
        base_model_dir = self._settings.rvc_base_model_dir
        if base_model_dir is not None:
            missing = [
                name
                for name in ("hubert_base.pt", "rmvpe.pt", "rmvpe.onnx")
                if not (base_model_dir / name).is_file()
            ]
            if missing:
                raise RuntimeError(f"RVC base models are missing: {', '.join(missing)}")


def install_voice_api(
    application: FastAPI,
    settings: Settings,
    engine: RVCEngine | None = None,
) -> RVCEngine:
    active_engine = engine or RVCEngine(settings)
    application.state.voice_engine = active_engine

    @application.delete("/api/voice/result")
    async def delete_result(
        request: Request,
        filename: Annotated[str, Form(min_length=1, max_length=260)],
        job_id: Annotated[str, Form(min_length=1, max_length=100)],
        song: Annotated[int, Form(ge=0)] = 0,
        audio_revision: Annotated[str | None, Form()] = None,
    ) -> dict[str, bool]:
        job = retained_job(request, job_id)
        job_result = _job_song_result(request, job_id, song)
        require_audio_revision(job_result, audio_revision)
        if mix_busy(job):
            raise HTTPException(status_code=409, detail="正在合轨，请稍后重试。")
        directory_song = (job_result.get("songNumber") or song + 1) - 1
        result_path = _result_path(settings, filename, job_id, directory_song)
        # 这个入口只管理"派生音频"的软删除。母带删掉整首歌就没法播放；提交后的保留音轨
        # （stems 里的 *_v<rev>_*.wav）该走 /stems 的软删除 —— 那里才记撤回、才检查版本。
        # 两者都必须拒绝，否则会留下"结果仍引用、文件已进回收站、还没有撤回记录"的死链。
        protected = [job_result.get("fullTrack"), job_result.get("editorFullTrack")]
        for project in (job_result, job_result.get("mixed")):
            stems = project.get("stems") if isinstance(project, dict) else None
            if isinstance(stems, dict):
                protected.extend(stems.values())
        if isinstance(job_result.get("mixed"), dict):
            protected.append(job_result["mixed"].get("fullTrack"))
        if any(_resolves_to(settings, value, result_path) for value in protected):
            raise HTTPException(status_code=409, detail="完整音频或保留音轨不能作为替换产物删除。")
        if not result_path.is_file():
            raise HTTPException(status_code=404, detail="Converted audio not found")
        trash_path = trash_result_path(settings, filename, job_id, directory_song)
        trash_path.parent.mkdir(parents=True, exist_ok=True)
        trash_path.unlink(missing_ok=True)
        result_path.replace(trash_path)
        deleted = dict(getattr(job, "deleted_replaced_vocals", {}).get(str(song)) or {})
        # 删的可能是替换人声本身，也可能是当初合出来的成品；两者都会让"当前状态"失效，
        # 而且都可能被 PUT 还原，所以按被删的文件决定要记什么、要作废什么。
        target_is_replacement = _resolves_to(settings, job_result.get("replacedVocal"), result_path)
        target_is_mix = _resolves_to(settings, job_result.get("mixedTrack"), result_path)
        if target_is_replacement:
            deleted["url"] = job_result.pop("replacedVocal")
            playback = job_result.get("playback")
            if isinstance(playback, dict):
                deleted["playback"] = playback.pop("replacedVocal", None)
            waveforms = job_result.get("waveforms")
            if isinstance(waveforms, dict):
                deleted["waveform"] = waveforms.pop("replaced", None)
            model = job_result.pop("_replacedVocalModel", None)
            if isinstance(model, str):
                deleted["model"] = model
        if (target_is_replacement or target_is_mix) and isinstance(
            job_result.get("mixedTrack"), str
        ):
            # 人声是合轨的输入，成品本身也是这个入口管理的产物：作废引用并留下还原材料。
            deleted.update(capture_mix_artifact(job_result))
            invalidate_mix_artifact(job, job_result)
        if target_is_replacement or target_is_mix:
            job.deleted_replaced_vocals[str(song)] = deleted
            if (
                target_is_replacement
                and job.replace_song == song
                and (job.replace_task is None or job.replace_task.done())
            ):
                job.replace_song = None
                job.replace_status = None
                job.replace_stage = None
                job.replace_progress = None
                job.replace_message = None
                job.replace_error = None
            job.save(settings.output_dir)
        return {"success": True}

    @application.put("/api/voice/result")
    async def restore_result(
        request: Request,
        filename: Annotated[str, Form(min_length=1, max_length=260)],
        job_id: Annotated[str, Form(min_length=1, max_length=100)],
        song: Annotated[int, Form(ge=0)] = 0,
        audio_revision: Annotated[str | None, Form()] = None,
    ) -> dict[str, bool]:
        job = retained_job(request, job_id)
        job_result = _job_song_result(request, job_id, song)
        require_audio_revision(job_result, audio_revision)
        if mix_busy(job):
            raise HTTPException(status_code=409, detail="正在合轨，请稍后重试。")
        directory_song = (job_result.get("songNumber") or song + 1) - 1
        result_path = _result_path(settings, filename, job_id, directory_song)
        if not result_path.is_file():
            trash_path = trash_result_path(settings, filename, job_id, directory_song)
            if not trash_path.is_file():
                raise HTTPException(status_code=404, detail="Deleted audio not found")
            result_path.parent.mkdir(parents=True, exist_ok=True)
            trash_path.replace(result_path)
        deleted_replaced_vocals = getattr(job, "deleted_replaced_vocals", {})
        deleted = deleted_replaced_vocals.get(str(song))
        if isinstance(deleted, dict):
            # 恢复的可能是替换人声，也可能是当初被软删除的成品；按实际恢复的文件决定还原什么。
            target_is_replacement = _resolves_to(settings, deleted.get("url"), result_path)
            target_is_mix = _resolves_to(settings, deleted.get("mixTrack"), result_path)
            if target_is_replacement:
                job_result["replacedVocal"] = deleted.pop("url")
                if isinstance(deleted.get("playback"), str):
                    job_result.setdefault("playback", {})["replacedVocal"] = deleted.pop("playback")
                else:
                    deleted.pop("playback", None)
                if isinstance(deleted.get("waveform"), list):
                    job_result.setdefault("waveforms", {})["replaced"] = deleted.pop("waveform")
                else:
                    deleted.pop("waveform", None)
                model = deleted.pop("model", None)
                if isinstance(model, str):
                    job_result["_replacedVocalModel"] = model
            if target_is_replacement or target_is_mix:
                # 撤回删除时把当初作废的成品一并还原。两份存档可能分两次 PUT 回来（例如先撤回
                # 人声、再撤回成品），所以**只有真正还原了才消费存档**：成品还在回收站里时把键
                # 留在记录里，等它自己的 PUT 再还回去。
                if isinstance(job_result.get("mixedTrack"), str):
                    # 期间已经重新合过轨：存档里那份成品已经过期，丢掉，别让它在后续 PUT 里复活。
                    deleted.pop("mixTrack", None)
                    deleted.pop("mixWaveform", None)
                    deleted.pop("mixPlayback", None)
                elif target_is_mix or stored_path_exists(
                    settings.output_dir, deleted.get("mixTrack")
                ):
                    if restore_mix_artifact(job_result, deleted):
                        deleted.pop("mixTrack", None)
                        deleted.pop("mixWaveform", None)
                        deleted.pop("mixPlayback", None)
                if not deleted:
                    deleted_replaced_vocals.pop(str(song), None)
                job.save(settings.output_dir)
        return {"success": True}

    return active_engine


def _job_song_result(request: Request, job_id: str, song: int) -> dict:
    job = retained_job(request, job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Generation job not found")
    if job.status != "succeeded":
        raise HTTPException(status_code=409, detail="生成任务尚未完成，无法编辑音频。")
    if not job.result:
        raise HTTPException(status_code=409, detail="Generation job has no audio result")
    results = [job.result, *(job.result.get("alternatives") or [])]
    if song >= len(results) or not isinstance(results[song], dict):
        raise HTTPException(status_code=404, detail="Song not found in generation job")
    return results[song]


def _reuse_base_models(rvc_infer, source_dir: Path) -> None:
    package_dir = Path(rvc_infer.__file__).resolve().parent
    target_dir = package_dir / "base_model"
    target_dir.mkdir(parents=True, exist_ok=True)
    for name in ("hubert_base.pt", "rmvpe.pt", "rmvpe.onnx"):
        source = source_dir / name
        target = target_dir / name
        if target.is_symlink() and target.resolve() != source.resolve():
            target.unlink()
        if not target.exists():
            target.symlink_to(source)


def _load_hubert_safely(inference, torch) -> None:
    from fairseq.data.dictionary import Dictionary
    from rvc_python.modules.vc.utils import load_hubert

    # Torch 2.6+ defaults to weights_only=True. This checkpoint needs exactly
    # Fairseq's Dictionary class, so allowlist that class instead of disabling
    # safe loading for the whole generate process.
    with torch.serialization.safe_globals([Dictionary]):
        inference.vc.hubert_model = load_hubert(inference.config, inference.lib_dir)


def _safe_song_name(value: str) -> str:
    name = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "_", value).strip(" .")
    name = re.sub(r"\s+", " ", name)
    return name[:120].rstrip(" .") or "converted"


def _source_song_name(filename: str | None, fallback: str) -> str:
    source = Path(filename).stem if filename else fallback
    source = re.sub(r"_(?:rvc_)?(?:vocal|vocals|voice)$", "", source, flags=re.IGNORECASE)
    return _safe_song_name(source)


def replacement_filename(filename: str | None, fallback: str = "converted") -> str:
    return f"{_source_song_name(filename, fallback)}_rvc_vocal.wav"


def _result_filename(value: str) -> str:
    """从 `filename` 里取出结果文件名，容忍各种 URL 形态。

    前端把 job 结果里的 `replacedVocal` 原样回传，而它可能是裸文件名、站内代理路径
    （`/api/music/output/jobs/.../x.wav`）或生成服务的绝对 URL。代理层历史上做过这层
    归一化，一旦代理不再处理（或换了调用方），后端直接 400 就会让"删除替换人声 → 撤回"
    这类操作失败。这里统一剥掉 scheme/host/目录，只保留最后一段。
    """
    parsed = urlsplit(value)
    path = unquote(parsed.path) if parsed.scheme or parsed.netloc else value
    segments = [segment for segment in path.split("/") if segment not in ("", ".")]
    # 归一化之前先拒绝穿越形状：不要靠"最后一段恰好不存在"来兜底，那样既可能 500，
    # 也失去了"明确拒绝非法输入"这条回归护栏。
    if not segments or any(
        segment == ".." or "\\" in segment or "\x00" in segment for segment in segments
    ):
        raise HTTPException(status_code=400, detail="Invalid result filename")
    name = segments[-1]
    if name != Path(name).name or Path(name).suffix.lower() != ".wav":
        raise HTTPException(status_code=400, detail="Invalid result filename")
    return name


def _result_path(
    settings: Settings,
    filename: str,
    job_id: str,
    song: int = 0,
) -> Path:
    try:
        root = job_song_dir(settings.output_dir, job_id, song)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail="Invalid job output path") from exc
    return root / _result_filename(filename)


def trash_result_path(
    settings: Settings,
    filename: str,
    job_id: str,
    song: int = 0,
) -> Path:
    return settings.output_dir / ".trash" / job_id / f"song_{song + 1}" / _result_filename(filename)


def mix_output_filename(vocal_filename: str) -> str:
    """原版的成品命名规则：`{人声名}_rvc_mix.wav`，同名原子覆盖。"""
    return f"{_source_song_name(vocal_filename, 'completed')}_rvc_mix.wav"


async def mix_tracks(
    inputs: list[Path],
    reference_path: Path,
    output_path: Path,
    timeout_seconds: float,
    *,
    gains: list[float] | None = None,
    track_outputs: list[Path] | None = None,
) -> None:
    """Render saved linear gains, or a legacy export when gains are omitted."""
    try:
        await _mix_steps(
            inputs,
            reference_path,
            output_path,
            timeout_seconds,
            gains=gains,
            track_outputs=track_outputs,
        )
    except MixTimeoutError as exc:
        raise RVCConversionError(
            f"FFmpeg mixing timed out after {timeout_seconds:g} seconds"
        ) from exc


async def _mix_steps(
    inputs: list[Path],
    reference_path: Path,
    output_path: Path,
    timeout_seconds: float,
    *,
    gains: list[float] | None = None,
    track_outputs: list[Path] | None = None,
) -> None:
    environment = prepare_ffmpeg_environment()
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout_seconds

    def remaining() -> float:
        value = deadline - loop.time()
        if value <= 0:
            raise MixTimeoutError("mix budget exhausted")
        return value

    sample_rate, channels, codec, duration = await _master_audio_format(
        reference_path, environment, remaining()
    )
    premix_path = output_path.with_name("premix.wav")
    command = ["ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error", "-y"]
    for input_path in inputs:
        command.extend(("-i", str(input_path)))
    if gains is not None:
        if (
            not inputs
            or len(gains) != len(inputs)
            or any(not math.isfinite(gain) or gain < 0 for gain in gains)
        ):
            raise RVCConversionError("Invalid mix inputs or gains")
        if track_outputs is not None:
            if len(track_outputs) != len(inputs):
                raise RVCConversionError("Invalid retained track outputs")
            for source, target, gain in zip(inputs, track_outputs, gains, strict=True):
                await _run_ffmpeg(
                    [
                        "ffmpeg",
                        "-nostdin",
                        "-hide_banner",
                        "-loglevel",
                        "error",
                        "-y",
                        "-i",
                        str(source),
                        "-af",
                        f"aresample={sample_rate},aformat=channel_layouts={channels}c,"
                        f"volume={gain:.12g},apad=whole_dur={duration:.9f},"
                        f"atrim=duration={duration:.9f},{LIMIT_FILTER.format(gain_db=0)}",
                        "-ar",
                        str(sample_rate),
                        "-ac",
                        str(channels),
                        "-c:a",
                        codec,
                        str(target),
                    ],
                    environment,
                    remaining(),
                )
            inputs = track_outputs
            gains = [1.0] * len(inputs)
            command = ["ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error", "-y"]
            for input_path in inputs:
                command.extend(("-i", str(input_path)))
        # Retained files already contain their levels; never compensate them back up.
        filters = [
            f"[{index}:a]aresample={sample_rate},"
            f"aformat=channel_layouts={channels}c,volume={gain:.12g}[track{index}]"
            for index, gain in enumerate(gains)
        ]
        labels = "".join(f"[track{index}]" for index in range(len(inputs)))
        filters.append(
            f"{labels}amix=inputs={len(inputs)}:duration=longest:"
            f"dropout_transition=0:normalize=0,apad=whole_dur={duration:.9f},"
            f"atrim=duration={duration:.9f},{LIMIT_FILTER.format(gain_db=0)}[premix]"
        )
        command.extend(
            (
                "-filter_complex",
                ";".join(filters),
                "-map",
                "[premix]",
                "-ar",
                str(sample_rate),
                "-ac",
                str(channels),
                "-c:a",
                codec,
                str(output_path),
            )
        )
        await _run_ffmpeg(command, environment, remaining())
        if not output_path.is_file() or output_path.stat().st_size == 0:
            raise RVCConversionError("FFmpeg mixing did not produce an output file")
        return
    command.extend(
        (
            "-filter_complex",
            build_mix_filter(len(inputs)),
            "-map",
            "[premix]",
            "-ar",
            str(sample_rate),
            "-ac",
            str(channels),
            "-c:a",
            "pcm_f32le",
            str(premix_path),
        )
    )
    await _run_ffmpeg(command, environment, remaining())
    reference_loudness, premix_loudness = await asyncio.gather(
        _integrated_loudness(reference_path, environment, remaining()),
        _integrated_loudness(premix_path, environment, remaining()),
    )
    # ponytail: cap malformed/silent-input compensation; widen only if real mixes need it.
    gain_db = max(-12.0, min(12.0, reference_loudness - premix_loudness))
    await _run_ffmpeg(
        [
            "ffmpeg",
            "-nostdin",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-i",
            str(premix_path),
            "-af",
            LIMIT_FILTER.format(gain_db=gain_db),
            "-ar",
            str(sample_rate),
            "-ac",
            str(channels),
            "-c:a",
            codec,
            str(output_path),
        ],
        environment,
        remaining(),
    )
    if not output_path.is_file() or output_path.stat().st_size == 0:
        raise RVCConversionError("FFmpeg mixing did not produce an output file")


async def _master_audio_format(
    input_path: Path, environment: dict[str, str], timeout_seconds: float
) -> tuple[int, int, str, float]:
    output = await _run_ffmpeg(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "a:0",
            "-show_entries",
            "stream=sample_rate,channels,codec_name,bits_per_sample,bits_per_raw_sample,duration:format=duration",
            "-of",
            "json",
            str(input_path),
        ],
        environment,
        timeout_seconds,
        capture_stdout=True,
    )
    try:
        metadata = json.loads(output)
        stream = metadata["streams"][0]
        sample_rate = int(stream["sample_rate"])
        channels = int(stream["channels"])
        bits = int(stream.get("bits_per_raw_sample") or stream.get("bits_per_sample") or 0)
        duration = float(stream.get("duration") or metadata["format"]["duration"])
    except (IndexError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise RVCConversionError("FFprobe could not read the master audio format") from exc
    if not 8_000 <= sample_rate <= 384_000 or not 1 <= channels <= 32:
        raise RVCConversionError("Master audio format is unsupported")
    if not math.isfinite(duration) or duration <= 0:
        raise RVCConversionError("Master audio duration is unsupported")
    codec = stream.get("codec_name")
    supported_pcm = {"pcm_u8", "pcm_s16le", "pcm_s24le", "pcm_s32le", "pcm_f32le", "pcm_f64le"}
    if codec not in supported_pcm:
        codec = {8: "pcm_u8", 24: "pcm_s24le", 32: "pcm_s32le"}.get(bits, "pcm_s16le")
    return sample_rate, channels, codec, duration


async def _integrated_loudness(
    input_path: Path, environment: dict[str, str], timeout_seconds: float
) -> float:
    stderr = await _run_ffmpeg(
        [
            "ffmpeg",
            "-nostdin",
            "-hide_banner",
            "-loglevel",
            "info",
            "-i",
            str(input_path),
            "-af",
            "loudnorm=I=-14:LRA=20:TP=-1:print_format=json",
            "-f",
            "null",
            "-",
        ],
        environment,
        timeout_seconds,
    )
    matches = re.findall(r'\{\s*"input_i".*?\}', stderr, flags=re.DOTALL)
    try:
        value = float(json.loads(matches[-1])["input_i"])
    except (IndexError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise RVCConversionError("FFmpeg could not measure audio loudness") from exc
    if not math.isfinite(value):
        raise RVCConversionError("Audio loudness is not finite")
    return value


async def _run_ffmpeg(
    command: list[str],
    environment: dict[str, str],
    timeout_seconds: float,
    *,
    capture_stdout: bool = False,
) -> str:
    """跑一个 ffmpeg/ffprobe 步骤；用法与 stems.py 的 BS-RoFormer 执行器保持一致。

    与旧实现（Popen + 50ms 轮询 + 在事件循环上阻塞 wait）相比，命令、参数与超时预算语义
    都没有变化，但不再阻塞事件循环，并显式关闭子进程 stdin（ffprobe 命令没带 `-nostdin`）。
    输出仍写临时文件而不是管道：既不占内存，也不会因为调用方取消而留下未读的管道。
    """
    with tempfile.TemporaryFile() as output_stream, tempfile.TemporaryFile() as error_stream:
        try:
            process = await asyncio.create_subprocess_exec(
                *command,
                env=environment,
                stdin=asyncio.subprocess.DEVNULL,
                stdout=output_stream if capture_stdout else asyncio.subprocess.DEVNULL,
                stderr=error_stream,
            )
        except OSError as exc:
            raise RVCConversionError(f"Unable to start FFmpeg: {exc}") from exc
        try:
            await asyncio.wait_for(process.wait(), timeout=timeout_seconds)
        except (asyncio.TimeoutError, asyncio.CancelledError) as exc:
            with suppress(ProcessLookupError):
                process.kill()
            await process.wait()
            if isinstance(exc, asyncio.CancelledError):
                raise
            raise MixTimeoutError("ffmpeg step timed out") from exc
        error_stream.seek(0)
        detail = error_stream.read().decode("utf-8", errors="replace")
        output_stream.seek(0)
        output = output_stream.read().decode("utf-8", errors="replace")
    if process.returncode != 0:
        raise RVCConversionError(f"FFmpeg mixing failed: {detail.strip() or process.returncode}")
    return output if capture_stdout else detail


def _resolves_to(settings: Settings, value: object, target: Path) -> bool:
    """记录里的路径是否正好指向 `target`（同一个歌曲目录里的同一个文件）。

    统一走 `output_path_from_url`：记录里可能是裸路径、站内路径或绝对 URL，直接拼路径会
    静默判错。解析不了或读不到都算"不是它"。
    """
    if not isinstance(value, str):
        return False
    try:
        return output_path_from_url(value, settings.output_dir) == target
    except (OSError, ValueError):
        return False
