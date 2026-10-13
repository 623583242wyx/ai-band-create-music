from __future__ import annotations

import asyncio
import sys
import wave
from importlib import resources
from pathlib import Path
from unittest.mock import patch

import pytest
from pydantic import ValidationError

from app.core.errors import GenerationError
from app.services.stems import (
    BS_ROFORMER_MODEL,
    STEM_NAMES,
    BSRoFormerStemSeparator,
    build_bs_roformer_command,
    publish_stems,
    require_bs_roformer_runtime,
    stem_output_files,
)
from tests.helpers import make_settings


def wav(path: Path, rate=8000, channels=1):
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as audio:
        audio.setparams((channels, 2, rate, 0, "NONE", "not compressed"))
        audio.writeframes(b"\x01\x00" * (rate * channels // 10))
    return path


def model_outputs(directory):
    directory.mkdir(parents=True, exist_ok=True)
    for name in STEM_NAMES:
        wav(directory / f"source_{'vocals' if name == 'vocal' else name}.wav", 44100, 2)


def test_command_selects_the_six_stem_model_and_project_cache(tmp_path):
    settings = make_settings(tmp_path)
    command = build_bs_roformer_command(settings, tmp_path / "input", tmp_path / "work")
    assert command[:3] == [sys.executable, "-m", "bs_roformer.inference"]
    assert command[command.index("--model") + 1] == BS_ROFORMER_MODEL
    assert command[command.index("--models_dir") + 1] == str(settings.bs_roformer_models_dir)
    assert command[command.index("--input_folder") + 1] == str(tmp_path / "input")
    assert command[command.index("--store_dir") + 1] == str(tmp_path / "work")
    assert "--device" not in command


@pytest.mark.parametrize("device", ["cpu", "cuda", "cuda:1"])
def test_explicit_device_is_forwarded(tmp_path, device):
    command = build_bs_roformer_command(
        make_settings(tmp_path, bs_roformer_device=device), tmp_path / "input", tmp_path / "work"
    )
    assert command[command.index("--device") + 1] == device


@pytest.mark.parametrize("device", ["", "mps", "gpu", "cuda:-1", "cuda:any"])
def test_invalid_device_is_rejected(tmp_path, device):
    with pytest.raises(ValidationError):
        make_settings(tmp_path, bs_roformer_device=device)


@pytest.mark.parametrize("timeout", [0, -1, float("nan"), float("inf")])
def test_timeout_is_positive_and_finite(tmp_path, timeout):
    with pytest.raises(ValidationError):
        make_settings(tmp_path, bs_roformer_timeout_seconds=timeout)


def test_runtime_reports_missing_package():
    with patch(
        "app.services.stems.importlib.util.find_spec",
        side_effect=[object(), object(), None, object()],
    ):
        with pytest.raises(GenerationError, match="依赖缺失：bs_roformer.*uv sync --frozen"):
            require_bs_roformer_runtime("auto")


@pytest.mark.parametrize("available,count,device", [(False, 0, "cuda"), (True, 1, "cuda:1")])
def test_explicit_cuda_does_not_silently_fall_back(available, count, device):
    with (
        patch("torch.cuda.is_available", return_value=available),
        patch("torch.cuda.device_count", return_value=count),
        patch("app.services.stems.importlib.util.find_spec", return_value=object()),
        pytest.raises(GenerationError, match="BS_ROFORMER_DEVICE=.*不可用"),
    ):
        require_bs_roformer_runtime(device)


def test_stem_outputs_include_original_music_name():
    assert stem_output_files(Path("普通话摇滚.mp3")) == {
        name: f"普通话摇滚_{name}.wav" for name in STEM_NAMES
    }


def test_installed_bs_roformer_architecture_produces_six_stereo_outputs():
    import torch
    import yaml
    from bs_roformer import get_model_from_config
    from bs_roformer.inference import SafeLoaderWithTuple
    from ml_collections import ConfigDict

    path = resources.files("bs_roformer") / "configs" / "BS-Rofo-SW-Fixed.yaml"
    config = ConfigDict(yaml.load(path.read_text(), Loader=SafeLoaderWithTuple))
    assert config.model.num_stems == 6
    assert {"vocal" if name == "vocals" else name for name in config.training.instruments} == set(
        STEM_NAMES
    )
    # Exercise the installed architecture without downloading the large pretrained weights.
    config.model.dim, config.model.depth = 16, 1
    config.model.heads, config.model.dim_head = 2, 8
    model = get_model_from_config("bs_roformer", config).eval()
    with torch.no_grad():
        output = model(torch.zeros(1, 2, 4096))
    assert output.shape == (1, 6, 2, 4096) and torch.isfinite(output).all()


def test_publish_maps_vocals_and_keeps_all_six_stems(tmp_path):
    work, output = tmp_path / "work", tmp_path / "output"
    model_outputs(work)
    output.mkdir()
    (work / "source_instrumental.wav").write_bytes(b"ignore extra derived output")
    files = stem_output_files(Path("demo.wav"))
    publish_stems(work, output, files, 4410)
    assert {path.name for path in output.iterdir()} == set(files.values())
    with wave.open(str(output / files["vocal"])) as audio:
        assert audio.getnchannels() == 2 and audio.getframerate() == 44100


@pytest.mark.parametrize(
    "bad_output",
    ["missing", "empty", "invalid", "wrong_rate", "wrong_channels", "wrong_length", "no_frames"],
)
def test_incomplete_model_output_cannot_publish_partial_stems(tmp_path, bad_output):
    work, output = tmp_path / "work", tmp_path / "output"
    model_outputs(work)
    output.mkdir()
    piano = work / "source_piano.wav"
    if bad_output == "missing":
        piano.unlink()
    elif bad_output in {"wrong_rate", "wrong_channels", "wrong_length", "no_frames"}:
        wav(
            piano,
            8000 if bad_output == "wrong_rate" else 44100,
            1 if bad_output == "wrong_channels" else 2,
        )
        if bad_output in {"wrong_length", "no_frames"}:
            with wave.open(str(piano), "wb") as audio:
                audio.setparams((2, 2, 44100, 0, "NONE", "not compressed"))
                audio.writeframes(b"\x00\x00" * (100 if bad_output == "wrong_length" else 0))
    else:
        piano.write_bytes(b"" if bad_output == "empty" else b"not audio")
    files = stem_output_files(Path("demo.wav"))
    (output / files["vocal"]).write_bytes(b"previous vocal")
    with pytest.raises(GenerationError):
        publish_stems(work, output, files, 4410)
    assert {p.name: p.read_bytes() for p in output.iterdir()} == {files["vocal"]: b"previous vocal"}


def test_publication_failure_rolls_back_the_whole_set(tmp_path, monkeypatch):
    work, output = tmp_path / "work", tmp_path / "output"
    model_outputs(work)
    output.mkdir()
    files = stem_output_files(Path("demo.wav"))
    replace = Path.replace

    def fail_guitar(path, target):
        if path.name == "source_guitar.wav":
            raise OSError("disk error")
        return replace(path, target)

    monkeypatch.setattr(Path, "replace", fail_guitar)
    with pytest.raises(OSError):
        publish_stems(work, output, files, 4410)
    assert list(output.iterdir()) == []


async def test_input_is_resampled_and_model_workdir_is_isolated(tmp_path, monkeypatch):
    settings = make_settings(tmp_path)
    separator = BSRoFormerStemSeparator(settings)
    source = wav(tmp_path / "原始录音.wav")
    output = tmp_path / "output"
    run = separator._run
    monkeypatch.setattr("app.services.stems.require_bs_roformer_runtime", lambda _: None)

    async def inference(command, environment, deadline, label):
        if command[0] == "ffmpeg":
            return await run(command, environment, deadline, label)
        input_dir = Path(command[command.index("--input_folder") + 1])
        work_dir = Path(command[command.index("--store_dir") + 1])
        assert input_dir.parent == work_dir and work_dir.parent == output
        with wave.open(str(input_dir / "source.wav")) as audio:
            assert audio.getframerate() == 44100 and audio.getnchannels() == 2
        model_outputs(work_dir)
        return "six stems generated", ""

    monkeypatch.setattr(separator, "_run", inference)
    first = await separator.split(source, output)
    second = await separator.split(source, output)
    assert set(first.files) == set(second.files) == set(STEM_NAMES)
    assert not set(first.files.values()) & set(second.files.values())
    assert first.stdout == "six stems generated" and first.duration_ms > 0
    assert all((output / name).is_file() for name in first.files.values())
    assert all(name.startswith("原始录音_") for name in first.files.values())
    assert not list(output.glob(".bs-roformer-*"))


async def test_inference_failure_cleans_staging_and_preserves_current_audio(tmp_path, monkeypatch):
    separator = BSRoFormerStemSeparator(make_settings(tmp_path))
    source = wav(tmp_path / "source.wav")
    output = tmp_path / "output"
    output.mkdir()
    old = output / "old_vocal.wav"
    old.write_bytes(b"old vocal")
    monkeypatch.setattr("app.services.stems.require_bs_roformer_runtime", lambda _: None)
    run = separator._run

    async def fail_model(command, environment, deadline, label):
        if command[0] == "ffmpeg":
            return await run(command, environment, deadline, label)
        raise GenerationError("model failed")

    monkeypatch.setattr(separator, "_run", fail_model)
    with pytest.raises(GenerationError, match="model failed"):
        await separator.split(source, output)
    assert list(output.iterdir()) == [old] and old.read_bytes() == b"old vocal"


async def test_failed_subprocess_reports_stderr(tmp_path):
    separator = BSRoFormerStemSeparator(make_settings(tmp_path))
    with pytest.raises(GenerationError, match="退出码：3.*model failure"):
        await separator._run(
            [
                sys.executable,
                "-c",
                "import sys; print('model failure', file=sys.stderr); sys.exit(3)",
            ],
            {},
            asyncio.get_running_loop().time() + 5,
            "BS-RoFormer 六轨分离",
        )


@pytest.mark.parametrize("cancel", [False, True])
async def test_timeout_and_cancellation_reap_the_model_process(tmp_path, monkeypatch, cancel):
    separator = BSRoFormerStemSeparator(make_settings(tmp_path, bs_roformer_timeout_seconds=0.2))
    launch = asyncio.create_subprocess_exec
    started = asyncio.Event()
    processes = []

    async def spawn(*args, **kwargs):
        process = await launch(*args, **kwargs)
        processes.append(process)
        started.set()
        return process

    monkeypatch.setattr("app.services.stems.asyncio.create_subprocess_exec", spawn)
    task = asyncio.create_task(
        separator._run(
            [sys.executable, "-c", "import time; time.sleep(30)"],
            {},
            asyncio.get_running_loop().time() + (5 if cancel else 0.2),
            "模型推理",
        )
    )
    await asyncio.wait_for(started.wait(), 2)
    if cancel:
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
    else:
        with pytest.raises(GenerationError, match="BS-RoFormer 拆轨超时"):
            await task
    assert processes[0].returncode is not None
