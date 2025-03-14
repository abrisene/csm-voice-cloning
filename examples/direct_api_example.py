#!/usr/bin/env python
"""
Example script to demonstrate how to use the CSM Voice Cloning API directly with requests.
"""

import argparse
import os
from pathlib import Path

import requests


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Clone a voice using the CSM Voice Cloning API.")
    parser.add_argument("--api-url", type=str, default="http://localhost:8000", help="API URL")
    parser.add_argument("--voice-sample", type=str, required=True, help="Path to voice sample")
    parser.add_argument("--text", type=str, required=True, help="Text to synthesize")
    parser.add_argument("--context-text", type=str, default="", help="Transcription of the voice sample")
    parser.add_argument("--output", type=str, default="output.wav", help="Output file path")
    parser.add_argument("--temperature", type=float, default=0.6, help="Temperature for sampling")
    parser.add_argument("--topk", type=int, default=20, help="Top-k for sampling")
    parser.add_argument("--output-format", type=str, default="wav", choices=["wav", "mp3"], help="Output format")
    args = parser.parse_args()

    # Prepare the API URL
    api_url = f"{args.api_url}/v1/voice-clone"

    print(f"Cloning voice from {args.voice_sample}...")
    print(f"Text to synthesize: {args.text}")

    # Prepare the form data
    with open(args.voice_sample, "rb") as f:
        files = {"audio_file": (os.path.basename(args.voice_sample), f)}
        data = {
            "text": args.text,
            "context_text": args.context_text,
            "temperature": args.temperature,
            "topk": args.topk,
            "output_format": args.output_format,
        }

        # Make the API request
        response = requests.post(api_url, files=files, data=data)

    # Check if the request was successful
    if response.status_code != 200:
        print(f"Error: {response.status_code}")
        print(response.text)
        return

    # Parse the response
    response_data = response.json()
    audio_url = response_data["audio_url"]
    duration_seconds = response_data["duration_seconds"]

    # Download the audio
    audio_response = requests.get(f"{args.api_url}{audio_url}")

    # Save the output
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "wb") as f:
        f.write(audio_response.content)

    print(f"Voice cloned successfully! Output saved to {output_path}")
    print(f"Duration: {duration_seconds:.2f} seconds")


if __name__ == "__main__":
    main()
