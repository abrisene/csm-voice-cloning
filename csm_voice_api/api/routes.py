import os
from typing import Dict, List, Optional

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from starlette.status import HTTP_400_BAD_REQUEST

from csm_voice_api.api.config import settings
from csm_voice_api.api.models import (
    AudioFormat,
    OpenAIErrorResponse,
    OpenAITTSRequest,
    OpenAITTSResponse,
    StreamingSessionRequest,
    StreamingSessionResponse,
    StreamingTextRequest,
    VoiceCloneRequest,
    VoiceCloneResponse,
)
from csm_voice_api.api.utils import (
    convert_audio_format,
    get_audio_duration,
    save_base64_audio,
    save_upload_file,
)
from csm_voice_api.core.voice_cloning import VoiceCloner

# Create router
router = APIRouter()

# Create voice cloner
voice_cloner = VoiceCloner(device=settings.DEVICE, hf_token=settings.HF_TOKEN)


@router.post(
    "/v1/voice-clone",
    response_model=VoiceCloneResponse,
    summary="Clone a voice and generate speech",
    description="Clone a voice from an audio sample and generate speech with the cloned voice.",
    tags=["Voice Cloning"],
)
async def voice_clone(
    text: str = Form(...),
    audio_file: UploadFile = File(...),
    context_text: Optional[str] = Form(""),
    speaker_id: Optional[int] = Form(999),
    max_audio_length_ms: Optional[float] = Form(15_000),
    temperature: Optional[float] = Form(0.6),
    topk: Optional[int] = Form(20),
    output_format: Optional[AudioFormat] = Form(AudioFormat.WAV),
    stream: Optional[bool] = Form(False),
) -> VoiceCloneResponse:
    """Clone a voice and generate speech."""
    # Save the uploaded audio file
    audio_path = save_upload_file(audio_file)

    # Generate a unique output filename
    output_filename = f"{os.path.basename(audio_path).split('.')[0]}_output.{output_format.value}"
    output_path = str(settings.OUTPUT_DIR / output_filename)

    try:
        # If streaming is requested, return a streaming response
        if stream:
            async def stream_generator():
                async for chunk in voice_cloner.stream_audio(
                    text=text,
                    context_audio_path=audio_path,
                    context_text=context_text,
                    speaker_id=speaker_id,
                    max_audio_length_ms=max_audio_length_ms,
                    temperature=temperature,
                    topk=topk,
                ):
                    yield chunk

            content_type = "audio/wav" if output_format == AudioFormat.WAV else "audio/mpeg"
            return StreamingResponse(
                stream_generator(),
                media_type=content_type,
                headers={"Content-Disposition": f"attachment; filename={output_filename}"}
            )

        # Otherwise, generate the audio normally
        audio_bytes, temp_path = voice_cloner.clone_voice(
            text=text,
            context_audio_path=audio_path,
            context_text=context_text,
            speaker_id=speaker_id,
            max_audio_length_ms=max_audio_length_ms,
            temperature=temperature,
            topk=topk,
        )

        # Convert to the requested format if needed
        if output_format != AudioFormat.WAV:
            output_path = convert_audio_format(temp_path, output_format, output_path)
        else:
            # Copy the file
            with open(output_path, "wb") as f:
                f.write(audio_bytes)

        # Get the audio duration
        duration = get_audio_duration(output_path)

        # Create the response
        audio_url = f"/v1/audio/{os.path.basename(output_path)}"

        return VoiceCloneResponse(
            audio_url=audio_url,
            duration_seconds=duration,
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to clone voice: {str(e)}",
        )


@router.get(
    "/v1/audio/{filename}",
    summary="Get audio file",
    description="Get an audio file by filename.",
    tags=["Voice Cloning"],
)
async def get_audio(filename: str) -> FileResponse:
    """Get an audio file by filename."""
    file_path = settings.OUTPUT_DIR / filename
    if not file_path.exists():
        raise HTTPException(
            status_code=404,
            detail=f"Audio file not found: {filename}",
        )

    return FileResponse(
        path=str(file_path),
        media_type="audio/wav" if filename.endswith(".wav") else "audio/mpeg",
        filename=filename,
    )


# Streaming API endpoints

@router.post(
    "/v1/streaming-session",
    response_model=StreamingSessionResponse,
    summary="Create a streaming session",
    description="Create a streaming session for voice cloning with text streaming.",
    tags=["Streaming"],
)
async def create_streaming_session(
    audio_file: UploadFile = File(...),
    context_text: Optional[str] = Form(""),
    speaker_id: Optional[int] = Form(999),
    temperature: Optional[float] = Form(0.6),
    topk: Optional[int] = Form(20),
    output_format: Optional[AudioFormat] = Form(AudioFormat.WAV),
) -> StreamingSessionResponse:
    """Create a streaming session for voice cloning."""
    # Save the uploaded audio file
    audio_path = save_upload_file(audio_file)

    try:
        # Create a streaming session
        session_id = voice_cloner.create_streaming_session(
            context_audio_path=audio_path,
            context_text=context_text,
            speaker_id=speaker_id,
            temperature=temperature,
            topk=topk,
        )

        return StreamingSessionResponse(
            session_id=session_id,
            message="Streaming session created successfully",
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to create streaming session: {str(e)}",
        )


@router.post(
    "/v1/stream-text",
    summary="Stream text for voice cloning",
    description="Stream text for voice cloning and get audio chunks in response.",
    tags=["Streaming"],
)
async def stream_text(request: StreamingTextRequest):
    """Stream text for voice cloning."""
    try:
        # Process the text chunk
        async def stream_generator():
            async for chunk in voice_cloner.process_text_chunk(
                session_id=request.session_id,
                text_chunk=request.text_chunk,
                is_final=request.is_final,
            ):
                yield chunk

        return StreamingResponse(
            stream_generator(),
            media_type="audio/wav",
        )
    except ValueError as e:
        raise HTTPException(
            status_code=404,
            detail=str(e),
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to process text chunk: {str(e)}",
        )


@router.websocket("/v1/ws/stream-text/{session_id}")
async def websocket_stream_text(websocket: WebSocket, session_id: str):
    """WebSocket endpoint for streaming text and receiving audio chunks."""
    await websocket.accept()

    try:
        # Check if session exists
        if session_id not in voice_cloner.streaming_sessions:
            await websocket.send_json({"error": f"Session {session_id} not found"})
            await websocket.close()
            return

        while True:
            # Receive text chunk from client
            data = await websocket.receive_json()
            text_chunk = data.get("text_chunk", "")
            is_final = data.get("is_final", False)

            # Process the text chunk
            audio_chunks = []
            async for chunk in voice_cloner.process_text_chunk(
                session_id=session_id,
                text_chunk=text_chunk,
                is_final=is_final,
            ):
                audio_chunks.append(chunk)

            # Send audio chunks to client
            if audio_chunks:
                for chunk in audio_chunks:
                    await websocket.send_bytes(chunk)

            # If this is the final chunk, close the connection
            if is_final:
                await websocket.close()
                break

    except WebSocketDisconnect:
        # Clean up the session if the client disconnects
        if session_id in voice_cloner.streaming_sessions:
            del voice_cloner.streaming_sessions[session_id]
    except Exception as e:
        await websocket.send_json({"error": str(e)})
        await websocket.close()


# OpenAI API compatibility routes

@router.post(
    "/v1/audio/speech",
    response_model=OpenAITTSResponse,
    responses={400: {"model": OpenAIErrorResponse}},
    summary="Generate speech with OpenAI API compatibility",
    description="Generate speech with a cloned voice using OpenAI API compatibility.",
    tags=["OpenAI Compatibility"],
)
async def openai_tts(request: OpenAITTSRequest) -> OpenAITTSResponse:
    """Generate speech with OpenAI API compatibility."""
    # Check if voice sample is provided
    if not request.voice_sample:
        return JSONResponse(
            status_code=HTTP_400_BAD_REQUEST,
            content={
                "error": {
                    "message": "Voice sample is required for custom voice",
                    "type": "invalid_request_error",
                    "param": "voice_sample",
                    "code": "parameter_missing",
                }
            },
        )

    try:
        # Save the voice sample
        audio_path = save_base64_audio(request.voice_sample)

        # Generate a unique output filename
        output_format = request.response_format
        output_filename = f"{os.path.basename(audio_path).split('.')[0]}_output.{output_format.value}"
        output_path = str(settings.OUTPUT_DIR / output_filename)

        # If streaming is requested, return a streaming response
        if request.stream:
            async def stream_generator():
                async for chunk in voice_cloner.stream_audio(
                    text=request.input,
                    context_audio_path=audio_path,
                    context_text=request.voice_sample_text or "",
                    speaker_id=999,  # Default speaker ID
                    max_audio_length_ms=15_000,  # Default max audio length
                    temperature=request.temperature or 0.6,
                    topk=request.topk or 20,
                ):
                    yield chunk

            content_type = "audio/wav" if output_format == AudioFormat.WAV else "audio/mpeg"
            return StreamingResponse(
                stream_generator(),
                media_type=content_type,
                headers={"Content-Disposition": f"attachment; filename={output_filename}"}
            )

        # Clone the voice
        audio_bytes, temp_path = voice_cloner.clone_voice(
            text=request.input,
            context_audio_path=audio_path,
            context_text=request.voice_sample_text or "",
            speaker_id=999,  # Default speaker ID
            max_audio_length_ms=15_000,  # Default max audio length
            temperature=request.temperature or 0.6,
            topk=request.topk or 20,
        )

        # Convert to the requested format if needed
        if output_format != AudioFormat.WAV:
            output_path = convert_audio_format(temp_path, output_format, output_path)
        else:
            # Copy the file
            with open(output_path, "wb") as f:
                f.write(audio_bytes)

        # Get the audio duration
        duration = get_audio_duration(output_path)

        # Create the response
        audio_url = f"/v1/audio/{os.path.basename(output_path)}"

        return OpenAITTSResponse(
            audio_url=audio_url,
            duration_seconds=duration,
        )
    except Exception as e:
        return JSONResponse(
            status_code=HTTP_400_BAD_REQUEST,
            content={
                "error": {
                    "message": f"Failed to generate speech: {str(e)}",
                    "type": "invalid_request_error",
                    "param": None,
                    "code": "processing_error",
                }
            },
        )
