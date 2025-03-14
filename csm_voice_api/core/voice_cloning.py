import os
import tempfile
import uuid
from pathlib import Path
from typing import AsyncGenerator, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torchaudio
from huggingface_hub import hf_hub_download

from csm_voice_api.core.generator import Segment, load_csm_1b


class StreamingSession:
    """Streaming session for voice cloning."""

    def __init__(
        self,
        context_audio: torch.Tensor,
        context_text: str,
        speaker_id: int,
        temperature: float,
        topk: int,
        sample_rate: int,
    ):
        """Initialize a streaming session.

        Args:
            context_audio: Context audio tensor.
            context_text: Transcription of the context audio.
            speaker_id: Speaker ID.
            temperature: Temperature for sampling.
            topk: Top-k for sampling.
            sample_rate: Sample rate of the audio.
        """
        self.context_audio = context_audio
        self.context_text = context_text
        self.speaker_id = speaker_id
        self.temperature = temperature
        self.topk = topk
        self.sample_rate = sample_rate

        # Buffer for accumulating text
        self.text_buffer = ""

        # Buffer for accumulating audio
        self.audio_buffer = None


class VoiceCloner:
    """Voice cloning service using CSM-1B model."""

    def __init__(self, device: str = "cuda", hf_token: Optional[str] = None):
        """Initialize the voice cloner.

        Args:
            device: Device to run the model on.
            hf_token: Hugging Face token for downloading the model.
        """
        self.device = device
        self.hf_token = hf_token

        # Set HF token if provided
        if hf_token:
            os.environ["HF_TOKEN"] = hf_token

        # Load the model
        self.generator = None

        # Store streaming sessions
        self.streaming_sessions: Dict[str, StreamingSession] = {}

    def load_model(self) -> None:
        """Load the CSM-1B model."""
        if self.generator is not None:
            return

        # Download the model
        model_path = hf_hub_download(
            repo_id="sesame/csm-1b",
            filename="ckpt.pt",
            token=self.hf_token
        )

        # Load the model
        self.generator = load_csm_1b(model_path, self.device)

    def remove_silence(
        self,
        audio: torch.Tensor,
        threshold: float = 0.01,
        min_silence_duration: float = 0.2,
        sample_rate: int = 24000
    ) -> torch.Tensor:
        """Remove silence from audio.

        Args:
            audio: Audio tensor.
            threshold: Threshold for silence detection.
            min_silence_duration: Minimum silence duration in seconds.
            sample_rate: Sample rate of the audio.

        Returns:
            Audio tensor with silence removed.
        """
        # Convert to numpy for easier processing
        audio_np = audio.cpu().numpy()

        # Calculate energy
        energy = np.abs(audio_np)

        # Find regions above threshold (speech)
        is_speech = energy > threshold

        # Convert min_silence_duration to samples
        min_silence_samples = int(min_silence_duration * sample_rate)

        # Find speech segments
        speech_segments = []
        in_speech = False
        speech_start = 0

        for i in range(len(is_speech)):
            if is_speech[i] and not in_speech:
                # Start of speech segment
                in_speech = True
                speech_start = i
            elif not is_speech[i] and in_speech:
                # Potential end of speech segment
                # Only end if silence is long enough
                silence_count = 0
                for j in range(i, min(len(is_speech), i + min_silence_samples)):
                    if not is_speech[j]:
                        silence_count += 1
                    else:
                        break

                if silence_count >= min_silence_samples:
                    # End of speech segment
                    in_speech = False
                    speech_segments.append((speech_start, i))

        # Handle case where audio ends during speech
        if in_speech:
            speech_segments.append((speech_start, len(is_speech)))

        # Concatenate speech segments
        if not speech_segments:
            return audio  # Return original if no speech found

        # Add small buffer around segments
        buffer_samples = int(0.05 * sample_rate)  # 50ms buffer
        processed_segments = []

        for start, end in speech_segments:
            buffered_start = max(0, start - buffer_samples)
            buffered_end = min(len(audio_np), end + buffer_samples)
            processed_segments.append(audio_np[buffered_start:buffered_end])

        # Concatenate all segments
        processed_audio = np.concatenate(processed_segments)

        return torch.tensor(processed_audio, device=audio.device)

    def preprocess_audio(self, audio_path: str) -> Tuple[torch.Tensor, int]:
        """Preprocess audio for voice cloning.

        Args:
            audio_path: Path to the audio file.

        Returns:
            Preprocessed audio tensor and sample rate.
        """
        # Load model if not already loaded
        self.load_model()

        # Load audio
        context_audio, sr = torchaudio.load(audio_path)
        context_audio = context_audio.mean(dim=0)  # Convert to mono

        # Resample if needed
        if sr != self.generator.sample_rate:
            context_audio = torchaudio.functional.resample(
                context_audio, orig_freq=sr, new_freq=self.generator.sample_rate
            )

        # Normalize audio volume for better consistency
        context_audio = context_audio / (torch.max(torch.abs(context_audio)) + 1e-8)

        # Apply silence removal
        audio_duration_sec = len(context_audio) / self.generator.sample_rate

        # Adjust threshold based on audio length
        silence_threshold = 0.015
        if audio_duration_sec > 10:
            # For longer files, be more aggressive with silence removal
            silence_threshold = 0.02

        context_audio = self.remove_silence(
            context_audio,
            threshold=silence_threshold,
            min_silence_duration=0.15,
            sample_rate=self.generator.sample_rate
        )

        return context_audio, self.generator.sample_rate

    def clone_voice(
        self,
        text: str,
        context_audio_path: str,
        context_text: str = "",
        speaker_id: int = 999,
        max_audio_length_ms: float = 15_000,
        temperature: float = 0.6,
        topk: int = 20,
        output_path: Optional[str] = None,
    ) -> Tuple[bytes, str]:
        """Clone a voice and generate speech.

        Args:
            text: Text to synthesize.
            context_audio_path: Path to the context audio file.
            context_text: Transcription of the context audio.
            speaker_id: Speaker ID.
            max_audio_length_ms: Maximum audio length in milliseconds.
            temperature: Temperature for sampling.
            topk: Top-k for sampling.
            output_path: Path to save the output audio.

        Returns:
            Audio bytes and temporary file path.
        """
        # Load model if not already loaded
        self.load_model()

        # Preprocess audio
        context_audio, _ = self.preprocess_audio(context_audio_path)

        # Create context segment
        context_segment = Segment(
            text=context_text,
            speaker=speaker_id,
            audio=context_audio
        )

        # Preprocess text for better pronunciation
        # Add punctuation if missing to help with phrasing
        if not any(p in text for p in ['.', ',', '!', '?']):
            text = text + '.'

        # Generate audio with context
        audio = self.generator.generate(
            text=text,
            speaker=speaker_id,
            context=[context_segment],
            max_audio_length_ms=max_audio_length_ms,
            temperature=temperature,
            topk=topk,
        )

        # Save the audio
        if output_path:
            torchaudio.save(output_path, audio.unsqueeze(0).cpu(), self.generator.sample_rate)
            temp_path = output_path
        else:
            # Create a temporary file
            temp_file = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
            temp_path = temp_file.name
            temp_file.close()

            torchaudio.save(temp_path, audio.unsqueeze(0).cpu(), self.generator.sample_rate)

        # Read the file and return the bytes
        with open(temp_path, "rb") as f:
            audio_bytes = f.read()

        return audio_bytes, temp_path

    async def stream_audio(
        self,
        text: str,
        context_audio_path: str,
        context_text: str = "",
        speaker_id: int = 999,
        max_audio_length_ms: float = 15_000,
        temperature: float = 0.6,
        topk: int = 20,
        chunk_size: int = 4096,  # Bytes per chunk
    ) -> AsyncGenerator[bytes, None]:
        """Stream audio for voice cloning.

        Args:
            text: Text to synthesize.
            context_audio_path: Path to the context audio file.
            context_text: Transcription of the context audio.
            speaker_id: Speaker ID.
            max_audio_length_ms: Maximum audio length in milliseconds.
            temperature: Temperature for sampling.
            topk: Top-k for sampling.
            chunk_size: Size of each audio chunk in bytes.

        Yields:
            Audio chunks.
        """
        # Generate the audio
        audio_bytes, temp_path = self.clone_voice(
            text=text,
            context_audio_path=context_audio_path,
            context_text=context_text,
            speaker_id=speaker_id,
            max_audio_length_ms=max_audio_length_ms,
            temperature=temperature,
            topk=topk,
        )

        # Stream the audio in chunks
        for i in range(0, len(audio_bytes), chunk_size):
            yield audio_bytes[i:i+chunk_size]

    def create_streaming_session(
        self,
        context_audio_path: str,
        context_text: str = "",
        speaker_id: int = 999,
        temperature: float = 0.6,
        topk: int = 20,
    ) -> str:
        """Create a streaming session for voice cloning.

        Args:
            context_audio_path: Path to the context audio file.
            context_text: Transcription of the context audio.
            speaker_id: Speaker ID.
            temperature: Temperature for sampling.
            topk: Top-k for sampling.

        Returns:
            Session ID.
        """
        # Load model if not already loaded
        self.load_model()

        # Preprocess audio
        context_audio, sample_rate = self.preprocess_audio(context_audio_path)

        # Create a session ID
        session_id = str(uuid.uuid4())

        # Create a streaming session
        self.streaming_sessions[session_id] = StreamingSession(
            context_audio=context_audio,
            context_text=context_text,
            speaker_id=speaker_id,
            temperature=temperature,
            topk=topk,
            sample_rate=sample_rate,
        )

        return session_id

    async def process_text_chunk(
        self,
        session_id: str,
        text_chunk: str,
        is_final: bool = False,
    ) -> AsyncGenerator[bytes, None]:
        """Process a text chunk for streaming voice cloning.

        Args:
            session_id: Session ID.
            text_chunk: Text chunk to synthesize.
            is_final: Whether this is the final chunk.

        Yields:
            Audio chunks.
        """
        # Check if session exists
        if session_id not in self.streaming_sessions:
            raise ValueError(f"Session {session_id} not found")

        # Get the session
        session = self.streaming_sessions[session_id]

        # Add the text chunk to the buffer
        session.text_buffer += text_chunk

        # If this is the final chunk or the buffer contains a complete sentence,
        # generate audio for the buffer
        if is_final or any(p in session.text_buffer for p in ['.', '!', '?']):
            # Create context segment
            context_segment = Segment(
                text=session.context_text,
                speaker=session.speaker_id,
                audio=session.context_audio
            )

            # Generate audio with context
            audio = self.generator.generate(
                text=session.text_buffer,
                speaker=session.speaker_id,
                context=[context_segment],
                max_audio_length_ms=5_000,  # Shorter for streaming
                temperature=session.temperature,
                topk=session.topk,
            )

            # Convert to bytes
            temp_file = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
            temp_path = temp_file.name
            temp_file.close()

            torchaudio.save(temp_path, audio.unsqueeze(0).cpu(), session.sample_rate)

            with open(temp_path, "rb") as f:
                audio_bytes = f.read()

            # Clean up
            os.unlink(temp_path)

            # Reset the buffer if this is the final chunk
            if is_final:
                session.text_buffer = ""
                # Clean up the session
                del self.streaming_sessions[session_id]
            else:
                # Reset the buffer but keep the session
                session.text_buffer = ""

            # Stream the audio in chunks
            chunk_size = 4096  # Bytes per chunk
            for i in range(0, len(audio_bytes), chunk_size):
                yield audio_bytes[i:i+chunk_size]
