"""Real-FFmpeg checks for the restored four-track mix.

These run the actual ffmpeg/ffprobe binaries: the mix filter graph, the loudness
match and the limiter are exactly the parts a stub cannot verify.
"""

from __future__ import annotations

import asyncio
import json
import math
import re
import shutil
import struct
import subprocess
import wave
from pathlib import Path

import pytest

from app.services.voice import MIX_FILTER, RVCConversionError, build_mix_filter, mix_tracks

RATE = 44_100
SECONDS = 2.0


def write_wav(
    path: Path, frequency: float, *, rate: int = RATE, channels: int = 2, amplitude: int = 8000
) -> Path:
    frames = bytearray()
    for index in range(int(rate * SECONDS)):
        value = int(amplitude * math.sin(2 * math.pi * frequency * index / rate))
        for _ in range(channels):
            frames += struct.pack("<h", value)
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(channels)
        handle.setsampwidth(2)
        handle.setframerate(rate)
        handle.writeframes(bytes(frames))
    return path


def probe(path: Path) -> dict:
    output = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "a:0",
            "-show_entries",
            "stream=sample_rate,channels,duration",
            "-of",
            "json",
            str(path),
        ],
        capture_output=True,
        check=True,
        text=True,
    ).stdout
    return json.loads(output)["streams"][0]


def peak(path: Path, *, channels: int = 2) -> float:
    """Highest absolute sample of the file, per channel.

    Measured per channel on purpose: `alimiter` bounds each channel's samples to the
    ceiling, whereas ffmpeg's stereo->mono downmix adds up to +3 dB for correlated
    channels (real stems in output/ show ratios up to 1.37), so a mono measurement
    would falsely look like the limiter failed.
    """
    output = subprocess.run(
        ["ffmpeg", "-v", "error", "-i", str(path), "-f", "f32le", "-ac", str(channels), "-"],
        capture_output=True,
        check=True,
    ).stdout
    samples = struct.unpack(f"<{len(output) // 4}f", output)
    assert all(math.isfinite(value) for value in samples)
    return max(abs(value) for value in samples)


def build_premix(inputs: list[Path], output_path: Path) -> Path:
    """Run only the mixing stage of the pipeline, without compensation or limiting."""
    command = ["ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error", "-y"]
    for path in inputs:
        command.extend(("-i", str(path)))
    command.extend(
        ("-filter_complex", MIX_FILTER, "-map", "[premix]", "-c:a", "pcm_f32le", str(output_path))
    )
    subprocess.run(command, check=True)
    return output_path


@pytest.fixture
def tracks(tmp_path: Path) -> tuple[list[Path], Path]:
    inputs = [
        write_wav(tmp_path / f"{name}.wav", frequency)
        for name, frequency in (("vocal", 220), ("drums", 330), ("bass", 110), ("other", 550))
    ]
    return inputs, write_wav(tmp_path / "master.wav", 440)


async def test_mix_matches_master_format_and_duration(tmp_path, tracks):
    inputs, master = tracks
    output = tmp_path / "mix.wav"

    await mix_tracks(inputs, master, output, 120)

    assert output.is_file() and output.stat().st_size > 0
    mixed, reference = probe(output), probe(master)
    assert mixed["sample_rate"] == reference["sample_rate"]
    assert mixed["channels"] == reference["channels"]
    assert float(mixed["duration"]) == pytest.approx(SECONDS, abs=0.1)
    assert peak(output) <= 0.891251 + 1e-3


async def test_mix_keeps_all_four_tracks_audible(tmp_path, tracks):
    inputs, master = tracks
    output = tmp_path / "mix.wav"

    await mix_tracks(inputs, master, output, 120)

    # Each input carries one tone; every band must still be present in the mix, so a
    # filter graph that silently drops inputs cannot pass.
    for frequency in (110.0, 220.0, 330.0, 550.0):
        report = subprocess.run(
            [
                "ffmpeg",
                "-hide_banner",
                "-nostdin",
                "-i",
                str(output),
                "-af",
                f"bandpass=f={frequency}:width_type=h:w=20,volumedetect",
                "-f",
                "null",
                "-",
            ],
            capture_output=True,
            text=True,
        ).stderr
        mean = re.search(r"mean_volume:\s*(-?[\d.]+) dB", report)
        assert mean is not None, (frequency, report[-400:])
        assert float(mean.group(1)) > -60.0, (frequency, mean.group(1))


async def test_mix_accepts_mono_master_and_silent_inputs(tmp_path):
    inputs = [write_wav(tmp_path / f"{index}.wav", 200 + index * 50) for index in range(4)]
    master = write_wav(tmp_path / "master.wav", 300, channels=1)
    silent = tmp_path / "silent.wav"
    with wave.open(str(silent), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(RATE)
        handle.writeframes(b"\x00\x00" * int(RATE * SECONDS))
    output = tmp_path / "mix.wav"

    await mix_tracks([silent, *inputs[1:]], master, output, 120)

    mixed = probe(output)
    assert int(mixed["channels"]) == 1
    assert int(mixed["sample_rate"]) == RATE
    assert math.isfinite(peak(output))


async def test_mix_timeout_kills_ffmpeg_and_reports_failure(tmp_path, tracks):
    """原版超时是 RVCConversionError（由路由映射成任务失败），并且必须回收子进程。"""
    inputs, master = tracks
    output = tmp_path / "mix.wav"

    with pytest.raises(RVCConversionError, match="timed out"):
        await mix_tracks(inputs, master, output, 0.001)

    assert not output.exists()
    # No stray ffmpeg/ffprobe process may outlive the timed-out mix. pgrep 不是到处都有
    # （精简镜像、macOS 默认没有），缺了就跳过这项检查，而不是让断言崩掉。
    pgrep = shutil.which("pgrep")
    if pgrep is not None:
        await asyncio.sleep(0.2)
        leftover = subprocess.run([pgrep, "-fa", "ffmpeg"], capture_output=True, text=True).stdout
        assert "mix.wav" not in leftover


def loudness(path: Path) -> float:
    report = subprocess.run(
        [
            "ffmpeg",
            "-hide_banner",
            "-nostdin",
            "-i",
            str(path),
            "-af",
            "loudnorm=I=-14:LRA=20:TP=-1:print_format=json",
            "-f",
            "null",
            "-",
        ],
        capture_output=True,
        text=True,
    ).stderr
    block = re.findall(r'\{\s*"input_i".*?\}', report, flags=re.DOTALL)[-1]
    return float(json.loads(block)["input_i"])


async def test_hot_stems_are_limited_to_the_true_peak_ceiling(tmp_path):
    """The limiter must bound the mix even when the raw sum overshoots full scale.

    The premix is built here with the same filter graph minus the gain/limiter stage, so
    the test proves the limiter did something instead of assuming it.
    """
    inputs = [
        write_wav(tmp_path / f"hot{index}.wav", 200 + index * 90, amplitude=25_000)
        for index in range(4)
    ]
    master = write_wav(tmp_path / "hot_master.wav", 440, amplitude=30_000)
    output = tmp_path / "hot_mix.wav"

    premix = build_premix(inputs, tmp_path / "raw_premix.wav")
    assert peak(premix) > 0.891251 + 1e-3, "fixture must overshoot, otherwise it proves nothing"

    await mix_tracks(inputs, master, output, 120)

    assert peak(output) <= 0.891251 + 1e-3
    assert peak(output) > 0.5  # not squashed to silence either


async def test_quiet_stems_are_compensated_towards_reference_but_capped_at_12_db(tmp_path):
    """Loudness matching is real but bounded: +12 dB is the documented ceiling.

    A quiet mix must be lifted towards the master, yet never by more than the cap, so a
    silent or malformed input cannot be amplified without limit.
    """
    inputs = [
        write_wav(tmp_path / f"quiet{index}.wav", 200 + index * 90, amplitude=300)
        for index in range(4)
    ]
    master = write_wav(tmp_path / "loud_master.wav", 440, amplitude=20_000)
    output = tmp_path / "quiet_mix.wav"

    premix = build_premix(inputs, tmp_path / "quiet_premix.wav")
    await mix_tracks(inputs, master, output, 120)

    gain = loudness(output) - loudness(premix)
    assert 11.0 <= gain <= 12.5, gain
    assert loudness(output) > loudness(premix)


async def test_all_silent_inputs_fail_instead_of_mixing_at_zero_gain(tmp_path):
    """原版严格性：响度测不出来（静音）就直接失败，不静默按 0 dB 补偿出成品。"""
    inputs = [write_wav(tmp_path / f"silent{index}.wav", 220, amplitude=0) for index in range(4)]
    master = write_wav(tmp_path / "silent_master.wav", 440, amplitude=20_000)
    output = tmp_path / "silent_mix.wav"

    with pytest.raises(RVCConversionError):
        await mix_tracks(inputs, master, output, 120)

    assert not output.exists()


def test_mix_filter_input_count_matches_the_stem_list():
    """六轨不能静默漏掉新增伴奏；旧四轨滤镜仍可兼容。"""
    from app.main import MIX_BACKING_STEMS
    from app.services.stems import STEM_NAMES

    mix_filter = build_mix_filter(len(STEM_NAMES))
    declared = int(re.search(r"amix=inputs=(\d+)", mix_filter).group(1))
    labels = {int(match) for match in re.findall(r"\[(\d+):a\]", mix_filter)}

    assert declared == len(labels) == 1 + len(MIX_BACKING_STEMS)
    assert len(STEM_NAMES) == 1 + len(MIX_BACKING_STEMS)


async def test_legacy_mix_keeps_guitar_and_piano_audible(tmp_path, tracks):
    inputs, master = tracks
    inputs += [write_wav(tmp_path / "guitar.wav", 770), write_wav(tmp_path / "piano.wav", 990)]
    output = tmp_path / "six-track-mix.wav"
    await mix_tracks(inputs, master, output, 120)
    # Removing either extra input must change the actual rendered signal.
    import numpy as np

    decoded = subprocess.run(
        ["ffmpeg", "-v", "error", "-i", str(output), "-ac", "1", "-f", "f32le", "-"],
        capture_output=True,
        check=True,
    ).stdout
    samples = np.frombuffer(decoded, dtype="<f4")
    spectrum = abs(np.fft.rfft(samples))
    for frequency in (770, 990):
        bin_index = round(frequency * len(samples) / RATE)
        assert spectrum[bin_index] > 100


def test_mix_filter_keeps_stereo_when_rvc_vocal_is_mono(tmp_path):
    inputs = [write_wav(tmp_path / "vocal.wav", 220, channels=1)] + [
        write_wav(tmp_path / f"backing{index}.wav", 330 + index * 110) for index in range(3)
    ]

    premix = build_premix(inputs, tmp_path / "premix.wav")

    assert int(probe(premix)["channels"]) == 2


async def test_project_gains_apply_once_without_eq_or_loudness_compensation(tmp_path, tracks):
    inputs, master = tracks
    first = tmp_path / "first.wav"
    second = tmp_path / "second.wav"
    gain = 10 ** (-6 / 20)
    await mix_tracks([inputs[0]], master, first, 120, gains=[gain])
    await mix_tracks([inputs[0]], first, second, 120, gains=[gain])
    assert peak(first) == pytest.approx(peak(inputs[0]) * gain, abs=1e-4)
    assert peak(second) == pytest.approx(peak(first), abs=1e-4)
    assert float(probe(second)["duration"]) == pytest.approx(SECONDS, abs=1 / RATE)


async def test_project_mix_preserves_length_for_silence_and_short_tracks(tmp_path, tracks):
    inputs, master = tracks
    silent = tmp_path / "silent.wav"
    await mix_tracks(inputs[:2], master, silent, 120, gains=[0, 0])
    assert peak(silent) == 0
    assert float(probe(silent)["duration"]) == pytest.approx(SECONDS, abs=1 / RATE)
    with wave.open(str(inputs[0]), "rb") as source:
        params = source.getparams()
        frames = source.readframes(RATE // 2)
    short = tmp_path / "short.wav"
    with wave.open(str(short), "wb") as output:
        output.setparams(params)
        output.writeframes(frames)
    padded = tmp_path / "padded.wav"
    await mix_tracks([short], master, padded, 120, gains=[1])
    assert float(probe(padded)["duration"]) == pytest.approx(SECONDS, abs=1 / RATE)
    with wave.open(str(padded), "rb") as output:
        output.setpos(RATE)
        assert not any(output.readframes(RATE))


async def test_project_mix_limits_hot_inputs_and_converts_mono(tmp_path):
    inputs = [write_wav(tmp_path / f"hot{index}.wav", 220, amplitude=30_000) for index in range(3)]
    master = write_wav(tmp_path / "mono.wav", 440, channels=1)
    output = tmp_path / "hot.wav"
    await mix_tracks(inputs, master, output, 120, gains=[2, 2, 2])
    assert int(probe(output)["channels"]) == 1
    assert 0.5 < peak(output, channels=1) <= 0.891251 + 1e-3


async def test_project_mix_writes_each_gain_into_its_retained_track(tmp_path, tracks):
    """保留音轨是交付物本身：长度补齐、格式统一，请求的增益必须只落一次。

    成品再由这些**已带增益**的文件求和 —— 实现若把增益又按一遍（或按原曲响度整体抬回），
    最后一条断言会明显偏轻。
    """
    inputs, master = tracks
    output = tmp_path / "mix.wav"
    retained = [tmp_path / f"kept{index}.wav" for index in range(2)]
    gains = [10 ** (-6 / 20), 1.0]

    await mix_tracks(inputs[:2], master, output, 120, gains=gains, track_outputs=retained)

    reference = probe(master)
    for source, kept, gain in zip(inputs[:2], retained, gains, strict=True):
        assert kept.is_file() and kept.stat().st_size > 0
        meta = probe(kept)
        assert int(meta["sample_rate"]) == int(reference["sample_rate"])
        assert int(meta["channels"]) == int(reference["channels"])
        assert float(meta["duration"]) == pytest.approx(SECONDS, abs=1 / RATE)
        assert peak(kept) == pytest.approx(peak(source) * gain, abs=1e-4)

    assert float(probe(output)["duration"]) == pytest.approx(SECONDS, abs=1 / RATE)
    assert peak(output) <= 0.891251 + 1e-3
    assert peak(output) > peak(retained[1])

    summed = tmp_path / "summed.wav"
    await mix_tracks(retained, master, summed, 120, gains=[1.0, 1.0])
    assert peak(output) == pytest.approx(peak(summed), abs=1e-3)


async def test_project_mix_pads_short_and_silent_retained_tracks(tmp_path, tracks):
    """短轨补静音、静音轨保留成同长度的 0：保留音轨与成品都不能比歌曲短。"""
    inputs, master = tracks
    with wave.open(str(inputs[0]), "rb") as source:
        params = source.getparams()
        half = source.readframes(RATE // 2)
    short = tmp_path / "short.wav"
    with wave.open(str(short), "wb") as output:
        output.setparams(params)
        output.writeframes(half)
    silent = write_wav(tmp_path / "silent.wav", 220, amplitude=0)
    retained = [tmp_path / "kept_short.wav", tmp_path / "kept_silent.wav"]
    output = tmp_path / "mix.wav"

    await mix_tracks([short, silent], master, output, 120, gains=[1.0, 1.0], track_outputs=retained)

    for kept in retained:
        assert float(probe(kept)["duration"]) == pytest.approx(SECONDS, abs=1 / RATE)
    assert peak(retained[1]) == 0
    with wave.open(str(retained[0]), "rb") as handle:
        assert abs(handle.getnframes() - int(RATE * SECONDS)) <= 2
        handle.setpos(int(RATE * 0.75))
        assert not any(handle.readframes(int(RATE * 0.25)))
    assert peak(output) > 0
