from __future__ import annotations

from typing import Any, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_serializer,
    model_validator,
)

from app.services.references import REFERENCE_ID_PATTERN
from app.services.stems import is_vocal_stem


class MixTrack(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    id: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9_:-]{0,99}$")
    source: Literal["stem", "full", "replaced"]
    stemId: str | None = Field(default=None, exclude_if=lambda value: value is None)
    gainDb: float = Field(ge=-66, le=6, allow_inf_nan=False)


def original_vocal_lane_id(stem_id: str) -> str:
    """替换编辑器里原人声车道的 id。

    未带前缀的人声键加一次 `original:`；已经带前缀的键（重新分轨可能落下 `original:vocal`）
    保持原样，不能再叠加一次。`replaced` 是替换产物本身，从来不是"原人声"那条车道。
    """
    if stem_id == "replaced" or stem_id.startswith("original:"):
        return stem_id
    return f"original:{stem_id}"


class TrackMixRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    commit: bool
    audioRevision: int = Field(ge=0)
    editor: Literal["tracks", "replace"]
    tracks: list[MixTrack] = Field(min_length=1, max_length=32)

    @model_validator(mode="after")
    def valid_tracks(self):
        ids = [track.id for track in self.tracks]
        if not self.commit or len(ids) != len(set(ids)):
            raise ValueError("A committed mix requires unique tracks")
        for track in self.tracks:
            expected = track.source
            if track.source == "stem":
                if not track.stemId:
                    raise ValueError("Stem ID is required")
                expected = track.stemId
                if self.editor == "replace" and is_vocal_stem(track.stemId):
                    expected = original_vocal_lane_id(track.stemId)
            elif "stemId" in track.model_fields_set:
                raise ValueError("Only stems can have a stem ID")
            if track.id != expected:
                raise ValueError("Track ID does not match its source")
            if track.source == "full" and len(self.tracks) != 1:
                raise ValueError("A full track must be the only input")
        return self


class GenerateRequest(BaseModel):
    model_config = ConfigDict(extra="ignore")

    prompt: str
    durationMinutes: int | float | str | None = None
    # Exact short length (e.g. 30 s previews); takes precedence over durationMinutes.
    durationSeconds: int | None = Field(default=None, ge=10, le=360)
    # Display name chosen in the group chat; shown by the history page instead of the lyrics.
    title: str | None = Field(default=None, max_length=100)
    provider: Literal["minimax_music", "elevenlabs_music", "mureka_music"] | None = None
    count: Literal[1, 2] = 1
    # ID returned by POST /api/references; ElevenLabs and Mureka only.
    referenceId: str | None = Field(default=None, pattern=REFERENCE_ID_PATTERN)

    @field_validator("title")
    @classmethod
    def normalize_title(cls, value: str | None) -> str | None:
        return value.strip() or None if value is not None else None


class ErrorResponse(BaseModel):
    success: bool = False
    message: str


class RecordingUploadResponse(BaseModel):
    id: str
    name: str
    audioUrl: str
    playbackUrl: str | None = None
    durationSeconds: float
    waveform: list[float]
    audioRevision: int


class PlaybackUrls(BaseModel):
    fullTrack: str | None = Field(default=None, exclude_if=lambda value: value is None)
    replacedVocal: str | None = Field(default=None, exclude_if=lambda value: value is None)
    mixedTrack: str | None = Field(default=None, exclude_if=lambda value: value is None)
    stems: dict[str, str] | None = Field(default=None, exclude_if=lambda value: value is None)


class MusicOutput(BaseModel):
    audioRevision: int = Field(default=0, ge=0)
    mixConfig: TrackMixRequest | None = Field(default=None, exclude_if=lambda value: value is None)
    # 最近一次完成创作（覆盖）的时间；从未覆盖过时缺省。
    updatedAt: str | None = Field(default=None, exclude_if=lambda value: value is None)
    songNumber: int | None = Field(default=None, ge=1, le=2, exclude_if=lambda value: value is None)
    fullTrack: str
    playback: PlaybackUrls | None = Field(default=None, exclude_if=lambda value: value is None)
    mixedTrack: str | None = Field(default=None, exclude_if=lambda value: value is None)
    replacedVocal: str | None = Field(default=None, exclude_if=lambda value: value is None)
    durationSeconds: float | None = None
    stems: dict[str, str]
    stemNames: dict[str, str] | None = Field(default=None, exclude_if=lambda value: value is None)
    stemUrls: list[str]
    waveforms: dict[str, list[float]]
    splitEnabled: bool
    debug: dict[str, Any]
    # 「替换后」那首歌：与原曲分开编辑、分开覆盖，字段与原曲同构。
    mixed: MusicOutput | None = Field(default=None, exclude_if=lambda value: value is None)


class GenerateResponse(MusicOutput):
    success: bool = True
    jobId: str
    createdAt: str | None = None
    provider: str | None = None
    requestedCount: int | None = None
    requestedDurationSeconds: float | None = Field(
        default=None, exclude_if=lambda value: value is None
    )
    prompt: str
    title: str | None = None
    durationMinutes: int | float | Literal["auto"]
    structuredPrompt: str
    lyrics: str
    count: Literal[1, 2] = 1
    alternatives: list[MusicOutput] = Field(default_factory=list)
    # Backward-compatible response field used by released clients.
    warning: str | None = None


class CreateGenerationJobResponse(BaseModel):
    jobId: str
    status: Literal["pending", "running"]


class UpdateGenerationJobRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    status: Literal["cancelled"]


class SongGenerationState(BaseModel):
    songNumber: int = Field(ge=1, le=2)
    status: Literal["pending", "running", "succeeded", "failed", "cancelled"]
    stage: str
    message: str = ""
    progress: int | None = Field(default=None, ge=0, le=100)
    step: int | None = None
    totalSteps: int | None = None
    receivedAudioSeconds: float | None = Field(default=None, ge=0, allow_inf_nan=False)
    expectedAudioSeconds: float | None = Field(default=None, gt=0, allow_inf_nan=False)
    error: str | None = None


class GenerationJobResponse(BaseModel):
    @model_serializer(mode="wrap")
    def serialize_retention(self, handler):
        response = handler(self)
        if self.retentionState == "disabled":
            for name in ("expiresAt", "retentionState", "audioAvailable"):
                response.pop(name, None)
        return response

    jobId: str
    createdAt: str
    expiresAt: str | None = None
    retentionState: Literal[
        "disabled", "dry_run", "unmanaged", "retained", "expired_active", "expired"
    ] = "disabled"
    audioAvailable: bool = False
    title: str | None = None
    prompt: str
    structuredPrompt: str | None = None
    lyrics: str | None = None
    status: Literal["pending", "running", "succeeded", "failed", "cancelled"]
    stage: str
    progress: int | None = None
    step: int | None = None
    totalSteps: int | None = None
    receivedAudioSeconds: float | None = Field(default=None, ge=0, allow_inf_nan=False)
    expectedAudioSeconds: float | None = Field(default=None, gt=0, allow_inf_nan=False)
    currentSong: int | None = Field(default=None, ge=1, le=2)
    songStates: list[SongGenerationState] = Field(default_factory=list)
    message: str
    # Backward-compatible response field used by released clients.
    warning: str | None = None
    result: GenerateResponse | None = None
    error: str | None = None
    # Operation progress is independent of the music generation task above.
    operation: Literal["split", "replace", "mix"] | None = None
    operationStatus: Literal["pending", "running", "succeeded", "failed", "cancelled"] | None = None
    operationSong: int | None = Field(default=None, ge=0)
    operationStage: str | None = None
    operationProgress: int | None = None
    operationMessage: str | None = None
    splitStatus: Literal["pending", "running", "succeeded", "failed", "cancelled"] | None = None
    splitSong: int | None = None
    splitError: str | None = None
    replaceStatus: Literal["pending", "running", "succeeded", "failed", "cancelled"] | None = None
    replaceSong: int | None = None
    replaceError: str | None = None
    mixStatus: Literal["pending", "running", "succeeded", "failed", "cancelled"] | None = None
    mixSong: int | None = None
    mixError: str | None = None
