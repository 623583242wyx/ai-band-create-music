from __future__ import annotations

import os
import wave
from pathlib import Path

import pytest
import soundfile as sf

from app.services.stems import STEM_NAMES, BSRoFormerStemSeparator
from tests.helpers import make_settings


@pytest.mark.slow
@pytest.mark.skipif(
    os.getenv("RUN_REAL_BS_ROFORMER_TEST") != "1",
    reason="set RUN_REAL_BS_ROFORMER_TEST=1 to download/load the six-stem model",
)
async def test_real_bs_roformer_smoke(tmp_path: Path) -> None:
    source = tmp_path / "short.wav"
    with wave.open(str(source), "wb") as audio:
        audio.setparams((2, 2, 44_100, 0, "NONE", "not compressed"))
        audio.writeframes(b"\x00\x00" * 2 * 44_100)
    settings = make_settings(tmp_path, bs_roformer_timeout_seconds=1800)
    output = tmp_path / "stems"
    result = await BSRoFormerStemSeparator(settings).split(source, output)
    assert set(result.files) == set(STEM_NAMES)
    for filename in result.files.values():
        samples, rate = sf.read(output / filename)
        assert rate == 44_100 and samples.shape == (44_100, 2)
