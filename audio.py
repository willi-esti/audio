import asyncio
import websockets
import whisper
import numpy as np
import ffmpeg
import logging
import aiohttp # For async HTTP requests to Ollama
import json
import os
import io
from TTS.api import TTS # Coqui TTS wrapper

# --- Configuration ---
WEBSOCKET_HOST = "0.0.0.0"
WEBSOCKET_PORT = 8000
WHISPER_MODEL_NAME = "base"
TARGET_SAMPLE_RATE = 16000
TARGET_CHANNELS = 1
TARGET_FORMAT = "s16le"
BYTES_PER_SAMPLE = 2
TRANSCRIPTION_THRESHOLD_SECONDS = 2 # Process audio roughly every N seconds (shorter for conversation)
AUDIO_BUFFER_MAX_SIZE = TARGET_SAMPLE_RATE * BYTES_PER_SAMPLE * TRANSCRIPTION_THRESHOLD_SECONDS * 3

OLLAMA_API_URL = "http://ollama:11434/api/generate" # Assuming ollama service name in Docker
OLLAMA_MODEL = "tinyllama" #"llama3" # Or mistral, phi3, etc. Make sure it's pulled in Ollama
TTS_MODEL_NAME = "tts_models/en/ljspeech/tacotron2-DDC" # Example Coqui TTS model
# TTS_MODEL_NAME = "tts_models/en/vctk/vits" # Alternative VITS model (often faster)
# You might need to download models the first time: python -m TTS.server.cli --list_models, python -m TTS.utils.download --model_name 'tts_models/en/ljspeech/tacotron2-DDC'
TTS_SPEAKER = None # Use default speaker for the model
TTS_STREAM_CHUNK_SIZE = TARGET_SAMPLE_RATE * BYTES_PER_SAMPLE // 4 # Send ~250ms chunks of TTS audio

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
log = logging.getLogger(__name__)

# --- Global State ---
try:
    log.info(f"Loading Whisper model: {WHISPER_MODEL_NAME}...")
    whisper_model = whisper.load_model(WHISPER_MODEL_NAME)
    log.info("Whisper model loaded.")
except Exception as e:
    log.error(f"Failed to load Whisper model: {e}")
    exit(1)

try:
    log.info(f"Loading TTS model: {TTS_MODEL_NAME}...")
    # Check if CUDA is available for TTS (optional but recommended)
    # device = "cuda" if torch.cuda.is_available() else "cpu"
    # tts = TTS(model_name=TTS_MODEL_NAME, progress_bar=False).to(device)
    tts = TTS(model_name=TTS_MODEL_NAME, progress_bar=False) # Use CPU by default for simplicity
    log.info("TTS model loaded.")
except Exception as e:
    log.error(f"Failed to load TTS model: {e}")
    log.warning("Make sure you have downloaded the model first.")
    exit(1)

audio_buffer = bytearray()
buffer_lock = asyncio.Lock()
is_speaking = asyncio.Event() # Flag to indicate if TTS is currently playing
interrupt_flag = asyncio.Event() # Flag to signal interruption

# --- Ollama Interaction ---
async def get_ollama_response(text):
    payload = {
        "model": OLLAMA_MODEL,
        "prompt": text,
        "stream": False # Get full response at once for now
    }
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(OLLAMA_API_URL, json=payload) as response:
                if response.status == 200:
                    data = await response.json()
                    return data.get("response", "").strip()
                else:
                    log.error(f"Ollama API error: {response.status} - {await response.text()}")
                    return None
    except aiohttp.ClientConnectorError as e:
        log.error(f"Could not connect to Ollama API at {OLLAMA_API_URL}: {e}")
        return None
    except Exception as e:
        log.error(f"Error during Ollama request: {e}")
        return None

# --- TTS Generation ---
async def generate_tts_audio(text, websocket):
    """Generates TTS audio and streams it to the websocket."""
    log.info(f"Generating TTS for: '{text}'")
    try:
        # Use asyncio.to_thread to run blocking TTS generation
        wav_chunks = await asyncio.to_thread(
            tts.tts,
            text=text,
            speaker=TTS_SPEAKER,
            # The 'tts' function itself doesn't directly yield bytes like older versions.
            # We'll generate the whole thing and chunk it for sending.
            # For true streaming *generation*, you'd need TTS library's streaming API if available,
            # or run it chunk-by-chunk which is complex.
        )

        # wav_chunks is likely a list of numpy int16 samples. Convert to bytes.
        # Ensure it's float32 first, then scale to int16
        if isinstance(wav_chunks, np.ndarray):
             if wav_chunks.dtype != np.float32:
                 wav_chunks = wav_chunks.astype(np.float32)
             # Scale to int16 range if it's float (often -1 to 1)
             if np.max(np.abs(wav_chunks)) <= 1.0:
                 wav_chunks = (wav_chunks * 32767).astype(np.int16)
             else: # Assume already scaled if values are large
                 wav_chunks = wav_chunks.astype(np.int16)

             audio_bytes = wav_chunks.tobytes()
        else:
            # Handle cases where it might return raw bytes or other formats if library changes
            log.error(f"Unexpected TTS output type: {type(wav_chunks)}")
            return

        log.info(f"Generated {len(audio_bytes)} bytes of TTS audio.")
        is_speaking.set() # Indicate that we are now speaking
        interrupt_flag.clear() # Clear any previous interrupt signal

        # Send audio in chunks
        bytes_sent = 0
        while bytes_sent < len(audio_bytes):
             if interrupt_flag.is_set():
                 log.info("TTS sending interrupted by user.")
                 interrupt_flag.clear() # Reset flag
                 break # Stop sending

             chunk_end = min(bytes_sent + TTS_STREAM_CHUNK_SIZE, len(audio_bytes))
             chunk = audio_bytes[bytes_sent:chunk_end]
             if not chunk:
                 break

             try:
                 # Send raw audio bytes directly
                 await websocket.send(chunk)
                 # log.debug(f"Sent TTS chunk: {len(chunk)} bytes")
                 bytes_sent += len(chunk)
                 await asyncio.sleep(0.05) # Small sleep to prevent overwhelming network/client buffer
             except websockets.ConnectionClosed:
                 log.warning("WebSocket closed while sending TTS audio.")
                 break
             except Exception as e:
                 log.error(f"Error sending TTS chunk: {e}")
                 break

    except RuntimeError as e:
        if "CUDA" in str(e):
            log.error(f"TTS CUDA error: {e}. Ensure GPU is available and memory is sufficient.")
        else:
            log.error(f"TTS generation error: {e}")
    except Exception as e:
        log.error(f"Error in TTS generation/streaming: {e}", exc_info=True)
    finally:
        is_speaking.clear() # Finished speaking (or interrupted)
        log.info("TTS sending finished or stopped.")


# --- Transcription & Conversation Handling ---
async def transcribe_and_respond(audio_data_bytes, websocket):
    """Transcribes audio, gets LLM response, generates TTS."""
    try:
        # Convert byte data to numpy array
        audio_np = np.frombuffer(audio_data_bytes, dtype=np.int16).astype(np.float32) / 32768.0

        if len(audio_np) < TARGET_SAMPLE_RATE * 0.5: # Ignore very short chunks
            log.info("Skipping very short audio chunk for transcription.")
            return

        log.info(f"Transcribing audio chunk of duration: {len(audio_np)/TARGET_SAMPLE_RATE:.2f}s")
        result = await asyncio.to_thread(
            whisper_model.transcribe,
            audio_np,
            fp16=False, # Set based on your hardware
            language=None # Auto-detect
        )
        user_text = result["text"].strip()

        if user_text and user_text.lower() not in ["...", ".", "thanks for watching"]: # Filter out common Whisper noise
            log.info(f"🗣️ User: {user_text}")

            # --- Get LLM Response ---
            log.info("Getting response from Ollama...")
            ai_response = await get_ollama_response(user_text)

            if ai_response:
                log.info(f"🤖 AI: {ai_response}")
                # --- Generate and Send TTS ---
                # Run TTS generation concurrently, don't block transcription loop
                asyncio.create_task(generate_tts_audio(ai_response, websocket))
            else:
                log.warning("Received no response from Ollama.")
        else:
            log.info("Transcription resulted in empty or filtered text.")

    except Exception as e:
        log.error(f"❌ Transcription/Response error: {e}", exc_info=True)


# --- WebSocket Handler & FFmpeg Pipeline ---
# (Similar to previous version, but with interruption logic)

async def ffmpeg_stderr_reader(stderr):
    """Reads and logs ffmpeg stderr."""
    while True:
        line = await stderr.readline()
        if not line:
            break
        log.warning(f"ffmpeg stderr: {line.decode().strip()}")
    log.info("ffmpeg stderr stream ended.")


async def audio_processor():
    """Periodically checks the buffer and triggers transcription."""
    global audio_buffer
    threshold_bytes = TARGET_SAMPLE_RATE * BYTES_PER_SAMPLE * TRANSCRIPTION_THRESHOLD_SECONDS
    last_processed_size = 0
    silence_counter = 0

    while True:
        await asyncio.sleep(TRANSCRIPTION_THRESHOLD_SECONDS / 3) # Check more frequently

        async with buffer_lock:
            current_buffer_size = len(audio_buffer)

            # Simple silence detection: if buffer hasn't grown much, maybe process remaining
            if current_buffer_size > TARGET_SAMPLE_RATE * BYTES_PER_SAMPLE * 0.5 and \
               (current_buffer_size >= threshold_bytes or (current_buffer_size == last_processed_size and current_buffer_size > 0)):

                if current_buffer_size == last_processed_size:
                    silence_counter += 1
                else:
                    silence_counter = 0

                # Process if threshold reached OR if buffer hasn't changed for a couple cycles (likely end of speech)
                if current_buffer_size >= threshold_bytes or silence_counter >= 2:
                    chunk_to_process = bytes(audio_buffer) # Process the whole buffer on likely end of speech
                    audio_buffer = bytearray() # Clear buffer
                    last_processed_size = 0
                    silence_counter = 0
                    log.debug(f"Processing buffer chunk of {len(chunk_to_process)} bytes (threshold or silence).")
                    if chunk_to_process:
                         # Run transcription in background
                        asyncio.create_task(transcribe_and_respond(chunk_to_process, current_websocket)) # Need websocket ref
                else:
                    # Update last processed size if buffer grew but didn't meet threshold yet
                    last_processed_size = current_buffer_size
                    log.debug(f"Buffer size {current_buffer_size}, waiting for threshold or silence.")

            elif current_buffer_size == 0:
                last_processed_size = 0 # Reset if buffer becomes empty
                silence_counter = 0

            # Safety net to prevent runaway buffer growth
            if current_buffer_size > AUDIO_BUFFER_MAX_SIZE:
                 log.warning(f"Audio buffer exceeded max size ({AUDIO_BUFFER_MAX_SIZE}), clearing...")
                 audio_buffer = bytearray()
                 last_processed_size = 0


# Global variable to hold the current websocket connection for the processor
current_websocket = None

async def handler(websocket):
    """Handles WebSocket connections and audio streaming."""
    global audio_buffer, current_websocket
    client_address = websocket.remote_address
    log.info(f"Client connected: {client_address}")
    current_websocket = websocket # Store current connection (simplistic, assumes one client)

    ffmpeg_process = None
    stderr_reader_task = None
    stdout_reader_task = None

    try:
        ffmpeg_command = [
            'ffmpeg', '-loglevel', 'warning', '-f', 'webm', '-i', 'pipe:0',
            '-f', TARGET_FORMAT, '-acodec', f'pcm_{TARGET_FORMAT}',
            '-ar', str(TARGET_SAMPLE_RATE), '-ac', str(TARGET_CHANNELS), 'pipe:1'
        ]
        log.info(f"Starting ffmpeg process: {' '.join(ffmpeg_command)}")
        ffmpeg_process = await asyncio.create_subprocess_exec(
            *ffmpeg_command, stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
        )
        log.info(f"ffmpeg process started (PID: {ffmpeg_process.pid})")

        stderr_reader_task = asyncio.create_task(ffmpeg_stderr_reader(ffmpeg_process.stderr))

        async def stdout_reader():
            """Reads decoded WAV data from ffmpeg stdout and adds to buffer."""
            global audio_buffer
            while True:
                try:
                    wav_chunk = await ffmpeg_process.stdout.read(TARGET_SAMPLE_RATE * BYTES_PER_SAMPLE // 2) # Read smaller chunks
                    if not wav_chunk:
                        log.info("ffmpeg stdout EOF reached.")
                        break
                    async with buffer_lock:
                        audio_buffer.extend(wav_chunk)
                    # log.debug(f"Read {len(wav_chunk)} bytes from ffmpeg stdout. Buffer size: {len(audio_buffer)}")
                except asyncio.CancelledError:
                    log.info("stdout_reader task cancelled.")
                    break
                except Exception as e:
                    log.error(f"Error reading ffmpeg stdout: {e}")
                    break
            log.info("stdout_reader task finished.")

        stdout_reader_task = asyncio.create_task(stdout_reader())

        # Receive messages (WebM chunks) from WebSocket
        async for message in websocket:
            if isinstance(message, bytes) and ffmpeg_process.stdin and not ffmpeg_process.stdin.is_closing():
                # --- Interruption Logic ---
                if is_speaking.is_set():
                    log.info("User started speaking while AI was speaking. Setting interrupt flag.")
                    interrupt_flag.set() # Signal TTS loop to stop sending
                    # Note: TTS generation thread might still be running, but sending stops.

                # log.debug(f"Received {len(message)} bytes from websocket. Writing to ffmpeg stdin.")
                try:
                    ffmpeg_process.stdin.write(message)
                    await ffmpeg_process.stdin.drain()
                except (BrokenPipeError, ConnectionResetError):
                    log.warning("ffmpeg stdin pipe closed unexpectedly.")
                    break
                except Exception as e:
                    log.error(f"Error writing to ffmpeg stdin: {e}")
                    break
            # Add handling for potential text messages if needed (e.g., config)
            # elif isinstance(message, str):
            #     log.info(f"Received text message: {message}")

    except websockets.ConnectionClosed as e:
        log.info(f"Client {client_address} disconnected: {e.code} {e.reason}")
    except Exception as e:
        log.error(f"Handler error: {e}", exc_info=True)
    finally:
        log.info(f"Cleaning up for client {client_address}...")
        current_websocket = None # Clear global reference
        is_speaking.clear()
        interrupt_flag.clear()

        # Close ffmpeg stdin
        if ffmpeg_process and ffmpeg_process.stdin and not ffmpeg_process.stdin.is_closing():
            try:
                ffmpeg_process.stdin.close()
                await ffmpeg_process.stdin.wait_closed()
            except Exception as e: log.warning(f"Error closing ffmpeg stdin: {e}")

        # Cancel and wait for tasks
        for task in [stdout_reader_task, stderr_reader_task]:
            if task and not task.done():
                task.cancel()
                try: await asyncio.wait_for(task, timeout=2.0)
                except asyncio.TimeoutError: log.warning(f"Timeout waiting for task {task.get_name()} to finish.")
                except asyncio.CancelledError: pass

        # Wait for ffmpeg process
        if ffmpeg_process and ffmpeg_process.returncode is None:
            log.info(f"Waiting for ffmpeg process (PID: {ffmpeg_process.pid}) to terminate...")
            try:
                await asyncio.wait_for(ffmpeg_process.wait(), timeout=5.0)
                log.info(f"ffmpeg process terminated with code: {ffmpeg_process.returncode}")
            except asyncio.TimeoutError:
                log.warning(f"Timeout waiting for ffmpeg process. Killing.")
                try: ffmpeg_process.kill(); await ffmpeg_process.wait()
                except Exception as e: log.error(f"Error killing ffmpeg process: {e}")
            except Exception as e: log.error(f"Error waiting for ffmpeg process: {e}")

        # Clear buffer
        async with buffer_lock:
            global audio_buffer
            audio_buffer = bytearray()
            log.info("Global audio buffer cleared.")

        log.info(f"Cleanup complete for client {client_address}.")

async def main():
    # Start the background audio processor task
    processor_task = asyncio.create_task(audio_processor())
    log.info("Audio processor task started.")

    async with websockets.serve(handler, WEBSOCKET_HOST, WEBSOCKET_PORT, max_size=10*1024*1024): # Increased max msg size
        log.info(f"WebSocket server started on ws://{WEBSOCKET_HOST}:{WEBSOCKET_PORT}")
        await asyncio.Future()  # Run forever

if __name__ == "__main__":
    # Make sure TTS models are available (optional check)
    # Example: Check if model path exists, or try a quick TTS test
    # if not os.path.exists(os.path.expanduser(f"~/.local/share/tts/{TTS_MODEL_NAME}")):
    #      print(f"Warning: TTS model {TTS_MODEL_NAME} might not be downloaded.")
    #      print("Run: python -m TTS.utils.download --model_name '{TTS_MODEL_NAME}'")

    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        log.info("Server stopped by user.")