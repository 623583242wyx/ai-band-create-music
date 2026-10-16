from __future__ import annotations

import asyncio
import multiprocessing
from pathlib import Path

import httpx
import pytest

from app.core.errors import GenerationError
from app.services.providers import AudioReference, ElevenLabsMusicProvider, MurekaMusicProvider
from app.services.references import (
    load_references,
    new_reference_entry,
    references_dir,
    save_references,
)
from tests.helpers import make_settings
from tests.integration.test_reference_api import reference_app, sine_wav


def local_reference(tmp_path: Path, provider: str, *, seconds: float = 30):
    settings = make_settings(
        tmp_path, music_api_mode="real", elevenlabs_api_key="test", mureka_api_key="test"
    )
    directory = references_dir(settings.output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "clip.mp3").write_bytes(b"prepared audio")
    entry = new_reference_entry("local_ref", "reference.mp3", seconds, "clip.mp3")
    entry.update(provider=provider, localOnly=True)
    save_references(settings.output_dir, [entry])
    return settings


def resolver(provider: str, settings, client):
    if provider == "elevenlabs_music":
        return ElevenLabsMusicProvider(settings, client)._reference_song_id
    return MurekaMusicProvider(settings, client)._reference_file_id


@pytest.mark.parametrize("provider", ["elevenlabs_music", "mureka_music"])
async def test_reference_remote_id_survives_provider_restart(tmp_path: Path, provider: str) -> None:
    settings = local_reference(tmp_path, provider)
    uploads = []

    def handler(request):
        uploads.append(request.url.path)
        return httpx.Response(200, json={"song_id": "remote", "id": "remote"})

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        assert await resolver(provider, settings, client)("local_ref") == "remote"
        assert await resolver(provider, settings, client)("local_ref") == "remote"
    assert len(uploads) == 1, "recreating a provider must not bill the same reference again"


@pytest.mark.parametrize("provider", ["elevenlabs_music", "mureka_music"])
async def test_concurrent_reference_resolution_uploads_once(tmp_path: Path, provider: str) -> None:
    settings = local_reference(tmp_path, provider)
    uploads = []

    async def handler(request):
        uploads.append(request.url.path)
        await asyncio.sleep(0.05)
        return httpx.Response(200, json={"song_id": "remote", "id": "remote"})

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        results = await asyncio.gather(
            resolver(provider, settings, client)("local_ref"),
            resolver(provider, settings, client)("local_ref"),
        )
    assert results == ["remote", "remote"]
    assert len(uploads) == 1, "parallel first use must share a single billed upload"


async def test_elevenlabs_7_second_reference_is_padded_before_remote_upload(tmp_path: Path) -> None:
    from app.services.references import prepare_reference_audio

    settings = local_reference(tmp_path, "elevenlabs_music", seconds=7.4)
    source = tmp_path / "short.wav"
    sine_wav(source, 7.4)
    clip = references_dir(settings.output_dir) / "clip.mp3"
    await prepare_reference_audio(source, clip)
    upload_duration = []

    async def handler(request):
        # Probe the actual MP3 bytes sent in the multipart request, not the indexed duration.
        boundary = request.headers["content-type"].split("boundary=")[1].encode()
        part = request.content.split(b"--" + boundary)[1]
        audio = part.split(b"\r\n\r\n", 1)[1].rsplit(b"\r\n", 1)[0]
        uploaded = tmp_path / "sent.mp3"
        uploaded.write_bytes(audio)
        duration = await prepare_reference_audio(uploaded, tmp_path / "probed.mp3")
        upload_duration.append(duration)
        return httpx.Response(200, json={"song_id": "remote"})

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        await resolver("elevenlabs_music", settings, client)("local_ref")
    assert 10 <= upload_duration[0] <= 30
    assert not list(clip.parent.glob(".elevenlabs-*.mp3"))


@pytest.mark.parametrize("provider", ["elevenlabs_music", "mureka_music"])
async def test_reference_stage_precedes_remote_upload(tmp_path: Path, provider: str) -> None:
    settings = local_reference(tmp_path, provider)
    events = []

    async def progress(stage, *args, **kwargs):
        events.append(stage)

    def handler(request):
        events.append("remote upload")
        return httpx.Response(401, json={"detail": "test stops before paid generation"})

    cls = ElevenLabsMusicProvider if provider == "elevenlabs_music" else MurekaMusicProvider
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        with pytest.raises(GenerationError):
            await cls(settings, client).generate(
                "pop", None, "[歌词与创作内容]\n测试歌词",
                reference=AudioReference("local_ref", 30_000, "medium"), progress=progress,
            )
    assert events[0] == "preparing_reference"
    assert "remote upload" in events


@pytest.mark.parametrize("concurrent", [False, True])
async def test_identical_uploads_reuse_one_local_reference(
    tmp_path: Path, concurrent: bool
) -> None:
    from tests.helpers import client

    app, settings, fake = reference_app(tmp_path)
    audio = sine_wav(tmp_path / "same.wav", 12)
    async with client(app) as http:
        async def upload(name):
            return await http.post(
                "/api/references?provider=elevenlabs_music", files={"file": (name, audio)}
            )

        if concurrent:
            first, second = await asyncio.gather(upload("first.wav"), upload("renamed.wav"))
        else:
            first, second = await upload("first.wav"), await upload("renamed.wav")
        assert first.is_success and second.is_success
        assert first.json()["id"] == second.json()["id"]
        listed = (await http.get("/api/references?provider=elevenlabs_music")).json()
        assert len(listed["references"]) == 1
    assert len(list(references_dir(settings.output_dir).glob("*.mp3"))) == 1
    assert fake.uploads == []


def resolve_in_process(settings, provider, start, log_path):
    async def run():
        async def handler(request):
            with log_path.open("a") as log:
                log.write("upload\n")
            await asyncio.sleep(0.1)
            return httpx.Response(200, json={"song_id": "remote", "id": "remote"})

        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            assert await resolver(provider, settings, client)("local_ref") == "remote"

    assert start.wait(5)
    asyncio.run(run())


@pytest.mark.parametrize("provider", ["elevenlabs_music", "mureka_music"])
def test_reference_upload_is_single_flight_across_processes(tmp_path: Path, provider: str) -> None:
    settings = local_reference(tmp_path, provider)
    context = multiprocessing.get_context("spawn")
    start = context.Event()
    log_path = tmp_path / "uploads.log"
    processes = [
        context.Process(target=resolve_in_process, args=(settings, provider, start, log_path))
        for _ in range(2)
    ]
    try:
        for process in processes:
            process.start()
        start.set()
        for process in processes:
            process.join(10)
            assert process.exitcode == 0
        assert log_path.read_text().splitlines() == ["upload"]
    finally:
        for process in processes:
            if process.is_alive():
                process.terminate()
                process.join(5)


@pytest.mark.parametrize("provider", ["elevenlabs_music", "mureka_music"])
async def test_failed_upload_releases_lock_without_caching(tmp_path: Path, provider: str) -> None:
    settings = local_reference(tmp_path, provider)
    attempts = []

    def handler(request):
        attempts.append(request.url.path)
        if len(attempts) == 1:
            return httpx.Response(401, json={"detail": "bad key"})
        return httpx.Response(200, json={"song_id": "remote", "id": "remote"})

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        with pytest.raises(GenerationError):
            await resolver(provider, settings, client)("local_ref")
        assert "remoteId" not in load_references(settings.output_dir)[0]
        assert await resolver(provider, settings, client)("local_ref") == "remote"
    assert len(attempts) == 2


async def test_deduplication_handles_existing_records_without_hash(tmp_path: Path) -> None:
    from tests.helpers import client

    app, settings, _ = reference_app(tmp_path)
    audio = sine_wav(tmp_path / "same.wav", 12)
    async with client(app) as http:
        first = await http.post("/api/references", files={"file": ("first.wav", audio)})
        entries = load_references(settings.output_dir)
        del entries[0]["sha256"]
        save_references(settings.output_dir, entries)
        second = await http.post("/api/references", files={"file": ("renamed.wav", audio)})
    assert second.status_code == 200
    assert second.json()["id"] == first.json()["id"]
    assert len(list(references_dir(settings.output_dir).glob("*.mp3"))) == 1


async def test_same_filename_different_content_is_not_deduplicated(tmp_path: Path) -> None:
    from tests.helpers import client

    app, _, _ = reference_app(tmp_path)
    async with client(app) as http:
        ids = []
        for seconds in (12, 15):
            response = await http.post(
                "/api/references",
                files={"file": ("same.wav", sine_wav(tmp_path / "a.wav", seconds))},
            )
            assert response.status_code == 201
            ids.append(response.json()["id"])
    assert ids[0] != ids[1]


@pytest.mark.parametrize("provider", ["elevenlabs_music", "mureka_music"])
async def test_remote_reference_cache_is_scoped_to_credentials(
    tmp_path: Path, provider: str
) -> None:
    settings = local_reference(tmp_path, provider)
    attempts = []

    def handler(request):
        attempts.append(request.url.path)
        return httpx.Response(200, json={"song_id": "remote", "id": "remote"})

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        await resolver(provider, settings, client)("local_ref")
        key = "elevenlabs_api_key" if provider == "elevenlabs_music" else "mureka_api_key"
        from pydantic import SecretStr

        settings = settings.model_copy(update={key: SecretStr("another-account")})
        await resolver(provider, settings, client)("local_ref")
    assert len(attempts) == 2
    assert "another-account" not in (references_dir(settings.output_dir) / "index.json").read_text()


async def test_orchestrator_reports_reference_preparation_message(tmp_path: Path) -> None:
    app, _, fake = reference_app(tmp_path)
    original_generate = fake.generate
    reports = []

    async def generate(*args, **kwargs):
        await kwargs["progress"]("preparing_reference", None, None)
        return await original_generate(*args, **kwargs)

    async def progress(stage, value, message, **kwargs):
        reports.append((stage, message))

    fake.generate = generate
    await app.state.orchestrator.generate(
        "pop", None, "review-test", provider="elevenlabs_music", progress=progress,
        reference=AudioReference("local_ref", 30_000, "medium"),
    )
    assert any(stage == "preparing_reference" and "计费" in message for stage, message in reports)
