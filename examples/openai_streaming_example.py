#!/usr/bin/env python
"""
Example script to demonstrate how to use the CSM Voice Cloning API with streaming via OpenAI client.
"""

import argparse
import base64
import os
from pathlib import Path

from openai import OpenAI


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Stream audio from the CSM Voice Cloning API using OpenAI client.")
    parser.add_argument("--api-url", type=str, default="http://localhost:8000/v1", help="API URL")
    parser.add_argument("--voice-sample", type=str, required=True, help="Path to voice sample")
    parser.add_argument("--text", type=str, required=True, help="Text to synthesize")
    parser.add_argument("--voice-sample-text", type=str, default="", help="Transcription of the voice sample")
    parser.add_argument("--output", type=str, default="output_stream.mp3", help="Output file path")
    parser.add_argument("--temperature", type=float, default=0.6, help="Temperature for sampling")
    parser.add_argument("--topk", type=int, default=20, help="Top-k for sampling")
    args = parser.parse_args()

    # Read the voice sample
    with open(args.voice_sample, "rb") as f:
        voice_sample_bytes = f.read()

    # Encode the voice sample as base64
    voice_sample_base64 = base64.b64encode(voice_sample_bytes).decode("utf-8")

    # Configure the OpenAI client
    client = OpenAI(
        api_key="not-needed",  # Not used but required
        base_url=args.api_url
    )

    print(f"Streaming audio for text: {args.text}")

    # Create output directory if it doesn't exist
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Generate speech with streaming
    response = client.audio.speech.create(
        model="csm-1b",
        voice="custom",
        input=args.text,
        voice_sample=voice_sample_base64,
        voice_sample_text=args.voice_sample_text,
        temperature=args.temperature,
        topk=args.topk,
        response_format="mp3",
        stream=True  # Enable streaming
    )

    # The response is a streaming response
    # We need to read the chunks and write them to a file
    with open(output_path, "wb") as f:
        # In a real streaming scenario, you would process each chunk as it arrives
        # Here we're just writing them to a file
        for chunk in response.iter_bytes():
            f.write(chunk)
            print(f"Received chunk of size {len(chunk)} bytes")

    print(f"Streaming completed. Output saved to {output_path}")


if __name__ == "__main__":
    main()
