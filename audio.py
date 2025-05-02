import asyncio
import websockets
import whisper
import ffmpeg

buffer = b""
model = whisper.load_model("base")

def decode_webm_to_wav_bytes(webm_bytes):
    """Decode webm bytes to wav bytes using ffmpeg in-memory"""
    try:
        out, err = (
            ffmpeg
            .input('pipe:0', format='webm')  # explicitly declare input format
            .output('pipe:1', format='wav', acodec='pcm_s16le', ac=1, ar='16000')
            .run(input=webm_bytes, capture_stdout=True, capture_stderr=True)
        )
        return out
    except ffmpeg.Error as e:
        print("❌ ffmpeg error:\n", e.stderr.decode())
        raise

import io
import tempfile

async def handler(websocket):
    global buffer
    print("Client connected")
    try:
        async for message in websocket:
            buffer += message

            if len(buffer) > 100000:  # ~1-2 seconds of audio
                try:
                    wav_bytes = decode_webm_to_wav_bytes(buffer)

                    if len(wav_bytes) < 1000:
                        print("⚠️ Skipped short/empty chunk")
                        buffer = b""
                        continue

                    # Save the wav_bytes temporarily for whisper (requires a file)
                    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                        f.write(wav_bytes)
                        temp_wav_path = f.name

                    result = model.transcribe(temp_wav_path, fp16=False)
                    print("🗣️", result["text"])

                except Exception as e:
                    print("❌ Decode/transcribe error:", str(e))

                buffer = b""
    except websockets.ConnectionClosed:
        print("Client disconnected")

async def main():
    async with websockets.serve(handler, "0.0.0.0", 8000):
        await asyncio.Future()  # Run forever

if __name__ == "__main__":
    asyncio.run(main())
send 