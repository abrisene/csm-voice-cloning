from enum import Enum
from typing import Dict, List, Optional, Union

from pydantic import BaseModel, Field


class AudioFormat(str, Enum):
    """Audio format enum."""

    WAV = "wav"
    MP3 = "mp3"


class VoiceCloneRequest(BaseModel):
    """Request model for voice cloning."""

    text: str = Field(..., description="Text to synthesize")
    context_text: Optional[str] = Field(
        "", description="Transcription of the context audio (optional)"
    )
    speaker_id: int = Field(
        999, description="Speaker ID (default: 999)"
    )
    max_audio_length_ms: float = Field(
        15_000, description="Maximum audio length in milliseconds"
    )
    temperature: float = Field(
        0.6, description="Temperature for sampling (lower = more deterministic)"
    )
    topk: int = Field(
        20, description="Top-k for sampling (lower = more focused)"
    )
    output_format: AudioFormat = Field(
        AudioFormat.WAV, description="Output audio format"
    )
    stream: bool = Field(
        False, description="Whether to stream the audio response"
    )


class VoiceCloneResponse(BaseModel):
    """Response model for voice cloning."""

    audio_url: str = Field(..., description="URL to the generated audio")
    duration_seconds: float = Field(..., description="Duration of the generated audio in seconds")


class StreamingTextRequest(BaseModel):
    """Request model for streaming text input."""

    text_chunk: str = Field(..., description="Text chunk to synthesize")
    is_final: bool = Field(False, description="Whether this is the final chunk")
    session_id: Optional[str] = Field(None, description="Session ID for continuing a stream")


class StreamingSessionRequest(BaseModel):
    """Request model for creating a streaming session."""

    context_text: Optional[str] = Field(
        "", description="Transcription of the context audio (optional)"
    )
    speaker_id: int = Field(
        999, description="Speaker ID (default: 999)"
    )
    temperature: float = Field(
        0.6, description="Temperature for sampling (lower = more deterministic)"
    )
    topk: int = Field(
        20, description="Top-k for sampling (lower = more focused)"
    )
    output_format: AudioFormat = Field(
        AudioFormat.WAV, description="Output audio format"
    )


class StreamingSessionResponse(BaseModel):
    """Response model for creating a streaming session."""

    session_id: str = Field(..., description="Session ID for the streaming session")
    message: str = Field("Streaming session created", description="Status message")


# OpenAI API compatibility models

class OpenAIVoice(str, Enum):
    """Voice enum for OpenAI compatibility."""

    CUSTOM = "custom"


class OpenAIModel(str, Enum):
    """Model enum for OpenAI compatibility."""

    CSM_1B = "csm-1b"


class OpenAITTSRequest(BaseModel):
    """Request model for OpenAI TTS API compatibility."""

    model: OpenAIModel = Field(OpenAIModel.CSM_1B, description="Model to use")
    input: str = Field(..., description="Text to synthesize")
    voice: OpenAIVoice = Field(OpenAIVoice.CUSTOM, description="Voice to use")
    response_format: AudioFormat = Field(
        AudioFormat.MP3, description="Output audio format"
    )
    speed: float = Field(
        1.0, description="Speed of the generated audio (not used)"
    )
    # Custom fields for voice cloning
    voice_sample: Optional[str] = Field(
        None, description="Base64-encoded audio sample for voice cloning"
    )
    voice_sample_text: Optional[str] = Field(
        None, description="Transcription of the voice sample"
    )
    temperature: Optional[float] = Field(
        0.6, description="Temperature for sampling"
    )
    topk: Optional[int] = Field(
        20, description="Top-k for sampling"
    )
    stream: bool = Field(
        False, description="Whether to stream the audio response"
    )


class OpenAITTSResponse(BaseModel):
    """Response model for OpenAI TTS API compatibility."""

    audio_url: str = Field(..., description="URL to the generated audio")
    duration_seconds: float = Field(..., description="Duration of the generated audio in seconds")


class OpenAIError(BaseModel):
    """Error model for OpenAI API compatibility."""

    message: str
    type: str
    param: Optional[str] = None
    code: Optional[str] = None


class OpenAIErrorResponse(BaseModel):
    """Error response model for OpenAI API compatibility."""

    error: OpenAIError
