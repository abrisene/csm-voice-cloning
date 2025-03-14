# CSM Voice Cloning API

An OpenAI-compatible API for voice cloning using the Sesame CSM-1B model.

## Features

- Voice cloning from audio samples
- OpenAI API compatibility
- FastAPI with automatic OpenAPI documentation
- Support for multiple audio formats (WAV, MP3)
- Poetry for dependency management
- **Streaming audio responses**
- **Real-time text-to-speech streaming**

## Prerequisites

- Python 3.10+
- CUDA-compatible GPU (for optimal performance)
- Hugging Face account with access to the CSM-1B model
- Hugging Face API token

## Installation

### Using Poetry (Recommended)

1. Install Poetry if you don't have it already:

```bash
curl -sSL https://install.python-poetry.org | python3 -
```

2. Clone this repository:

```bash
git clone https://github.com/yourusername/csm-voice-api.git
cd csm-voice-api
```

3. Install dependencies:

```bash
poetry install
```

### Using pip

1. Clone this repository:

```bash
git clone https://github.com/yourusername/csm-voice-api.git
cd csm-voice-api
```

2. Install dependencies:

```bash
pip install -e .
```

## Setting Up Your Hugging Face Token

You need to set your Hugging Face token to download the model. You can do this in two ways:

1. Set it as an environment variable:

```bash
export HF_TOKEN="your_hugging_face_token"
```

2. Create a `.env` file in the project root:

```
HF_TOKEN=your_hugging_face_token
```

## Accepting the Model on Hugging Face

Before using the model, you need to accept the terms on Hugging Face:

1. Visit the [Sesame CSM-1B model page](https://huggingface.co/sesame/csm-1b)
2. Click on "Access repository" and accept the terms
3. Make sure you're logged in with the same account that your HF_TOKEN belongs to

## Running the API

### Using Poetry

```bash
poetry run start
```

### Using Python

```bash
python -m csm_voice_api.api.main
```

The API will be available at http://localhost:8000.

## API Documentation

Once the API is running, you can access the documentation at:

- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

## API Endpoints

### Native Endpoints

#### POST /v1/voice-clone

Clone a voice and generate speech.

**Form Parameters:**

- `text` (string, required): Text to synthesize
- `audio_file` (file, required): Audio sample for voice cloning
- `context_text` (string, optional): Transcription of the audio sample
- `speaker_id` (integer, optional, default: 999): Speaker ID
- `max_audio_length_ms` (number, optional, default: 15000): Maximum audio length in milliseconds
- `temperature` (number, optional, default: 0.6): Temperature for sampling
- `topk` (integer, optional, default: 20): Top-k for sampling
- `output_format` (string, optional, default: "wav"): Output audio format (wav or mp3)
- `stream` (boolean, optional, default: false): Whether to stream the audio response

**Response:**

```json
{
  "audio_url": "/v1/audio/filename.wav",
  "duration_seconds": 3.5
}
```

#### GET /v1/audio/{filename}

Get an audio file by filename.

**Path Parameters:**

- `filename` (string, required): Filename of the audio file

**Response:**

- Audio file (WAV or MP3)

### Streaming Endpoints

#### POST /v1/streaming-session

Create a streaming session for voice cloning with text streaming.

**Form Parameters:**

- `audio_file` (file, required): Audio sample for voice cloning
- `context_text` (string, optional): Transcription of the audio sample
- `speaker_id` (integer, optional, default: 999): Speaker ID
- `temperature` (number, optional, default: 0.6): Temperature for sampling
- `topk` (integer, optional, default: 20): Top-k for sampling
- `output_format` (string, optional, default: "wav"): Output audio format (wav or mp3)

**Response:**

```json
{
  "session_id": "12345678-1234-5678-1234-567812345678",
  "message": "Streaming session created successfully"
}
```

#### POST /v1/stream-text

Stream text for voice cloning and get audio chunks in response.

**Request Body:**

```json
{
  "session_id": "12345678-1234-5678-1234-567812345678",
  "text_chunk": "Hello, ",
  "is_final": false
}
```

**Response:**

- Audio chunk (WAV)

#### WebSocket /v1/ws/stream-text/{session_id}

WebSocket endpoint for streaming text and receiving audio chunks.

**Path Parameters:**

- `session_id` (string, required): Session ID from the streaming session

**WebSocket Messages:**

- Send:
```json
{
  "text_chunk": "Hello, ",
  "is_final": false
}
```

- Receive: Audio chunk (bytes)

### OpenAI-Compatible Endpoints

#### POST /v1/audio/speech

Generate speech with OpenAI API compatibility.

**Request Body:**

```json
{
  "model": "csm-1b",
  "input": "Text to synthesize",
  "voice": "custom",
  "response_format": "mp3",
  "voice_sample": "base64-encoded audio",
  "voice_sample_text": "Transcription of the voice sample",
  "temperature": 0.6,
  "topk": 20,
  "stream": false
}
```

**Response:**

```json
{
  "audio_url": "/v1/audio/filename.mp3",
  "duration_seconds": 3.5
}
```

## Using with OpenAI Python Client

You can use the API with the OpenAI Python client:

```python
import openai

# Configure the client to use your API
client = openai.OpenAI(
    api_key="not-needed",  # Not used but required
    base_url="http://localhost:8000/v1"  # Your API URL
)

# Generate speech with a custom voice
response = client.audio.speech.create(
    model="csm-1b",
    voice="custom",
    input="Hello, this is my cloned voice speaking.",
    voice_sample="base64-encoded audio",  # Base64-encoded audio sample
    voice_sample_text="This is a sample of my voice for context.",  # Optional
    response_format="mp3"
)

# Save the audio to a file
with open("output.mp3", "wb") as f:
    # Assuming response.content contains the audio bytes
    f.write(response.content)
```

### Streaming with OpenAI Client

You can also use streaming with the OpenAI client:

```python
import openai

# Configure the client to use your API
client = openai.OpenAI(
    api_key="not-needed",  # Not used but required
    base_url="http://localhost:8000/v1"  # Your API URL
)

# Generate speech with streaming
response = client.audio.speech.create(
    model="csm-1b",
    voice="custom",
    input="Hello, this is my cloned voice speaking.",
    voice_sample="base64-encoded audio",  # Base64-encoded audio sample
    voice_sample_text="This is a sample of my voice for context.",  # Optional
    response_format="mp3",
    stream=True  # Enable streaming
)

# Process the streaming response
with open("output_stream.mp3", "wb") as f:
    for chunk in response.iter_bytes():
        f.write(chunk)
        # Process each chunk as it arrives
```

## Streaming Text Input and Audio Output

For more advanced use cases, you can stream text input and receive audio output in real-time:

```python
import asyncio
import json
import websockets
import requests

# Create a streaming session
def create_session(api_url, voice_sample_path):
    with open(voice_sample_path, "rb") as f:
        files = {"audio_file": f}
        response = requests.post(f"{api_url}/v1/streaming-session", files=files)
    return response.json()["session_id"]

# Stream text and receive audio
async def stream_text(api_url, session_id, text):
    ws_url = f"ws://{api_url.replace('http://', '')}/v1/ws/stream-text/{session_id}"
    async with websockets.connect(ws_url) as websocket:
        # Split text into chunks (in a real app, this would be from a live source)
        chunks = ["Hello, ", "this is ", "streaming ", "text input."]

        for i, chunk in enumerate(chunks):
            is_final = i == len(chunks) - 1
            await websocket.send(json.dumps({
                "text_chunk": chunk,
                "is_final": is_final
            }))

            # Receive audio chunk
            audio_data = await websocket.recv()
            # Process audio chunk (play it, save it, etc.)

# Run the example
async def main():
    api_url = "http://localhost:8000"
    voice_sample = "path/to/voice_sample.mp3"

    session_id = create_session(api_url, voice_sample)
    await stream_text(api_url, session_id, "Hello, this is streaming text input.")

asyncio.run(main())
```

## Examples

Check out the example scripts in the `examples/` directory:

- `direct_api_example.py`: Example of using the API directly with requests
- `openai_client_example.py`: Example of using the API with the OpenAI client
- `streaming_example.py`: Example of using the streaming API with WebSockets
- `openai_streaming_example.py`: Example of using the streaming API with the OpenAI client

## License

This project uses the Sesame CSM-1B model, which is subject to its own license terms. Please refer to the [model page](https://huggingface.co/sesame/csm-1b) for details.
