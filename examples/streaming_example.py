#!/usr/bin/env python
"""
Example script to demonstrate how to use the CSM Voice Cloning API with streaming.
"""

import argparse
import asyncio
import json
import os
import time
from pathlib import Path

import requests
import websockets


async def stream_text_websocket(
    api_url: str,
    session_id: str,
    text: str,
    chunk_size: int = 10,
    delay: float = 0.5,
):
    """Stream text to the API using WebSocket and receive audio chunks."""
    # Connect to the WebSocket
    ws_url = f"ws://{api_url.replace('http://', '')}/v1/ws/stream-text/{session_id}"
    async with websockets.connect(ws_url) as websocket:
        print(f"Connected to WebSocket at {ws_url}")

        # Split the text into chunks
        chunks = [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]

        # Create output directory
        output_dir = Path("streaming_output")
        output_dir.mkdir(parents=True, exist_ok=True)

        # Send each chunk with a delay
        for i, chunk in enumerate(chunks):
            is_final = i == len(chunks) - 1

            # Send the chunk
            await websocket.send(json.dumps({
                "text_chunk": chunk,
                "is_final": is_final,
            }))
            print(f"Sent chunk {i+1}/{len(chunks)}: '{chunk}'")

            # Receive audio chunks
            try:
                audio_data = await asyncio.wait_for(websocket.recv(), timeout=10.0)

                # Save the audio chunk
                chunk_file = output_dir / f"chunk_{i+1}.wav"
                if isinstance(audio_data, bytes):
                    with open(chunk_file, "wb") as f:
                        f.write(audio_data)
                    print(f"Received and saved audio chunk to {chunk_file}")
            except asyncio.TimeoutError:
                print(f"No audio received for chunk {i+1}")

            # Wait before sending the next chunk
            if not is_final:
                await asyncio.sleep(delay)


def create_streaming_session(
    api_url: str,
    voice_sample: str,
    context_text: str = "",
    speaker_id: int = 999,
    temperature: float = 0.6,
    topk: int = 20,
) -> str:
    """Create a streaming session."""
    # Prepare the API URL
    session_url = f"{api_url}/v1/streaming-session"

    # Prepare the form data
    with open(voice_sample, "rb") as f:
        files = {"audio_file": (os.path.basename(voice_sample), f)}
        data = {
            "context_text": context_text,
            "speaker_id": speaker_id,
            "temperature": temperature,
            "topk": topk,
        }

        # Make the API request
        response = requests.post(session_url, files=files, data=data)

    # Check if the request was successful
    if response.status_code != 200:
        raise Exception(f"Error creating streaming session: {response.text}")

    # Parse the response
    response_data = response.json()
    session_id = response_data["session_id"]

    return session_id


async def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Stream text to the CSM Voice Cloning API.")
    parser.add_argument("--api-url", type=str, default="http://localhost:8000", help="API URL")
    parser.add_argument("--voice-sample", type=str, required=True, help="Path to voice sample")
    parser.add_argument("--text", type=str, required=True, help="Text to synthesize")
    parser.add_argument("--context-text", type=str, default="", help="Transcription of the voice sample")
    parser.add_argument("--chunk-size", type=int, default=10, help="Size of each text chunk")
    parser.add_argument("--delay", type=float, default=0.5, help="Delay between chunks in seconds")
    args = parser.parse_args()

    print(f"Creating streaming session with voice sample: {args.voice_sample}")
    session_id = create_streaming_session(
        api_url=args.api_url,
        voice_sample=args.voice_sample,
        context_text=args.context_text,
    )
    print(f"Streaming session created with ID: {session_id}")

    print(f"Streaming text: {args.text}")
    await stream_text_websocket(
        api_url=args.api_url,
        session_id=session_id,
        text=args.text,
        chunk_size=args.chunk_size,
        delay=args.delay,
    )
    print("Streaming completed")


if __name__ == "__main__":
    asyncio.run(main())
