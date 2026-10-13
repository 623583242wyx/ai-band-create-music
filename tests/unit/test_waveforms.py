import asyncio
import inspect
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from app.services.waveforms import (
    WAVEFORM_BIN_COUNT,
    extract_waveform,
    extract_waveforms,
    summarize_waveform,
)


def test_summarize_waveform_preserves_relative_energy() -> None:
    samples = np.concatenate((np.full(100, 0.25), np.full(100, 1.0))).astype(np.float32)

    assert summarize_waveform(samples, 2) == [0.25, 1.0]
    assert summarize_waveform(np.zeros(8, dtype=np.float32), 4) == [0.0] * 4


@pytest.mark.parametrize("amplitude", [0.002, 0.01, 0.5])
def test_waveform_restores_per_track_display_amplitude(amplitude) -> None:
    assert summarize_waveform(np.full(8000, amplitude, dtype=np.float32), 4) == [1.0] * 4


def test_near_silent_separation_residue_is_not_normalized_into_a_waveform() -> None:
    # The reported piano has a peak RMS of about 0.000049, below -80 dBFS.
    samples = np.concatenate((np.full(100, 0.00001), np.full(100, 0.000049)))
    assert summarize_waveform(samples, 2) == [0.0, 0.0]


def test_low_level_residue_above_the_old_threshold_does_not_make_a_waveform() -> None:
    samples = np.concatenate((np.full(100, 0.0002), np.full(100, 0.0005)))
    assert summarize_waveform(samples, 2) == [0.0, 0.0]


async def test_silence_filter_applies_to_all_track_types(tmp_path):
    samples = np.concatenate((np.full(22_050, 0.0005), np.full(22_050, 0.1)))
    path = tmp_path / "quiet-then-audible.wav"
    sf.write(path, np.column_stack((samples, samples)), 44_100, subtype="FLOAT")
    names = (
        "full",
        "vocal",
        "drums",
        "bass",
        "guitar",
        "piano",
        "other",
        "replaced",
        "mix",
        "recording_test",
    )
    waveforms = await extract_waveforms({name: path for name in names})
    assert set(waveforms) == set(names)
    for waveform in waveforms.values():
        assert waveform[:320] == [0.0] * 320
        assert waveform[320:] == [1.0] * 320


def test_quiet_sections_remain_relative_to_the_audible_track_peak() -> None:
    samples = np.concatenate((np.full(100, 0.00002), np.full(100, 0.25), np.full(100, 0.5)))
    assert summarize_waveform(samples, 3) == [0.0, 0.5, 1.0]


def test_stereo_energy_does_not_cancel_opposite_phase_channels() -> None:
    samples = np.column_stack((np.full(100, 0.25), np.full(100, -0.25)))
    assert summarize_waveform(samples, 4) == [1.0] * 4


def test_waveform_amplitude_is_normalized_and_keeps_relative_levels() -> None:
    assert summarize_waveform(np.array([0.1, 2.0]), 2) == [0.05, 1.0]


@pytest.mark.parametrize("frequency,opposite_phase", [(440, True), (6000, False)])
async def test_extraction_preserves_stereo_and_high_frequency_energy(
    tmp_path, frequency, opposite_phase
):
    samples = 0.4 * np.sin(2 * np.pi * frequency * np.arange(44_100) / 44_100)
    stereo = np.column_stack((samples, -samples if opposite_phase else samples))
    source = tmp_path / "audible.wav"
    sf.write(source, stereo, 44_100, subtype="FLOAT")

    waveform = await extract_waveform(source)

    assert len(waveform) == WAVEFORM_BIN_COUNT
    assert max(waveform) == 1.0
    assert np.mean(waveform) > 0.9


def test_summarize_waveform_defaults_to_the_editor_bin_count() -> None:
    samples = np.ones(200_000, dtype=np.float32)

    assert WAVEFORM_BIN_COUNT == 640
    assert len(summarize_waveform(samples)) == WAVEFORM_BIN_COUNT


def test_both_entry_points_share_one_default_bin_count() -> None:
    # 同一模块内的两个默认值必须一致，否则 "full" 与分轨的 x 轴长度不同。
    assert (
        inspect.signature(summarize_waveform).parameters["bin_count"].default
        == inspect.signature(extract_waveform).parameters["bin_count"].default
        == WAVEFORM_BIN_COUNT
    )


async def test_waveform_progress_counts_processed_files_including_skipped_files(monkeypatch):
    counts = []

    async def extract(path):
        if path.name == "bad":
            raise RuntimeError("invalid audio")
        return [0.5]

    monkeypatch.setattr("app.services.waveforms.extract_waveform", extract)
    result = await extract_waveforms(
        {name: Path(name) for name in ("vocal", "bad", "bass")},
        progress=lambda finished, total: counts.append((finished, total)),
    )
    assert result == {"vocal": [0.5], "bass": [0.5]}
    assert counts == [(1, 3), (2, 3), (3, 3)]


async def test_extraction_bounds_concurrent_decodes_and_keeps_all_tracks(monkeypatch):
    active = peak = 0

    async def extract(_path):
        nonlocal active, peak
        active += 1
        peak = max(peak, active)
        await asyncio.sleep(0)
        active -= 1
        return [0.25]

    monkeypatch.setattr("app.services.waveforms.extract_waveform", extract)
    paths = {str(index): Path(str(index)) for index in range(7)}
    assert await extract_waveforms(paths) == {name: [0.25] for name in paths}
    assert peak <= 2


async def test_waveform_cancellation_does_not_count_unfinished_files(monkeypatch):
    counts = []
    started = asyncio.Event()

    async def extract(_path):
        started.set()
        await asyncio.Event().wait()

    monkeypatch.setattr("app.services.waveforms.extract_waveform", extract)
    task = asyncio.create_task(
        extract_waveforms(
            {"vocal": Path("vocal")},
            progress=lambda *count: counts.append(count),
        )
    )
    await started.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert counts == []
