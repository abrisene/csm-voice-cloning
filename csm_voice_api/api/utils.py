import base64
import os
import tempfile
import uuid
from pathlib import Path
from typing import Optional, Tuple

import torchaudio
from fastapi import HTTPException, UploadFile
from pydub import AudioSegment

from csm_voice_api.api.config import settings
from csm_voice_api.api.models import AudioFormat


def save_upload_file(upload_file: UploadFile) -> str:
    """Save an uploaded file to the upload directory.

    Args:
        upload_file: The uploaded file.

    Returns:
        The path to the saved file.
    """
    # Generate a unique filename
    filename = f"{uuid.uuid4()}{Path(upload_file.filename).suffix}"
    file_path = settings.UPLOAD_DIR / filename

    # Save the file
    with open(file_path, "wb") as f:
        f.write(upload_file.file.read())

    return str(file_path)


def save_base64_audio(base64_audio: str, suffix: str = ".wav") -> str:
    """Save a base64-encoded audio file to the upload directory.

    Args:
        base64_audio: The base64-encoded audio.
        suffix: The file suffix.

    Returns:
        The path to the saved file.
    """
    # Generate a unique filename
    filename = f"{uuid.uuid4()}{suffix}"
    file_path = settings.UPLOAD_DIR / filename

    # Decode and save the file
    try:
        audio_data = base64.b64decode(base64_audio)
        with open(file_path, "wb") as f:
            f.write(audio_data)
    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid base64-encoded audio: {str(e)}"
        )

    return str(file_path)


def get_audio_duration(file_path: str) -> float:
    """Get the duration of an audio file in seconds.

    Args:
        file_path: The path to the audio file.

    Returns:
        The duration of the audio file in seconds.
    """
    try:
        audio, sample_rate = torchaudio.load(file_path)
        duration = audio.shape[1] / sample_rate
        return duration
    except Exception:
        # Fallback to pydub if torchaudio fails
        try:
            audio = AudioSegment.from_file(file_path)
            return len(audio) / 1000.0
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"Failed to get audio duration: {str(e)}"
            )


def convert_audio_format(
    input_path: str,
    output_format: AudioFormat,
    output_path: Optional[str] = None
) -> str:
    """Convert an audio file to the specified format.

    Args:
        input_path: The path to the input audio file.
        output_format: The output audio format.
        output_path: The path to the output audio file.

    Returns:
        The path to the converted audio file.
    """
    if output_path is None:
        # Generate a unique filename
        filename = f"{uuid.uuid4()}.{output_format.value}"
        output_path = str(settings.OUTPUT_DIR / filename)

    try:
        if output_format == AudioFormat.WAV:
            # Convert to WAV
            audio = AudioSegment.from_file(input_path)
            audio.export(output_path, format="wav")
        elif output_format == AudioFormat.MP3:
            # Convert to MP3
            audio = AudioSegment.from_file(input_path)
            audio.export(output_path, format="mp3", bitrate="192k")
        else:
            raise ValueError(f"Unsupported audio format: {output_format}")
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to convert audio format: {str(e)}"
        )

    return output_path
