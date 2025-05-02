import asyncio
import websockets
import whisper
import numpy as np
import ffmpeg # Use ffmpeg-python
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Configuration ---
WEBSOCKET_HOST = "0.0.0.0"
WEBSOCKET_PORT = 8000
MODEL_NAME = "base" # Or "tiny", "small", "medium", "large"
TARGET_SAMPLE_RATE = 16000
TARGET_CHANNELS = 1
TARGET_FORMAT = "s16le" # Signed 16-bit little-endian PCM
BYTES_PER_SAMPLE = 2 # 16-bit = 2 bytes
TRANSCRIPTION_THRESHOLD_SECONDS = 2 # Process audio roughly every N seconds
AUDIO_BUFFER_MAX_SIZE = TARGET_SAMPLE_RATE * BYTES_PER_SAMPLE * TRANSCRIPTION_THRESHOLD_SECONDS * 2 # Keep a bit more buffer

# --- Global Whisper Model (Load once) ---
try:
    logging.info(f"Loading Whisper model: {MODEL_NAME}...")
    model = whisper.load_model(MODEL_NAME)
    logging.info("Whisper model loaded.")
except Exception as e:
    logging.error(f"Failed to load Whisper model: {e}")
    exit(1)

# --- Global Audio Buffer ---
# This will now store decoded WAV data
audio_buffer = bytearray()
buffer_lock = asyncio.Lock()

async def transcribe_audio_chunk(audio_data_bytes):
    """Transcribes a chunk of raw PCM audio bytes using Whisper."""
    try:
        # Convert byte data to numpy array
        audio_np = np.frombuffer(audio_data_bytes, dtype=np.int16).astype(np.float32) / 32768.0

        if len(audio_np) == 0:
            logging.warning("Empty audio chunk received for transcription.")
            return

        logging.info(f"Transcribing audio chunk of duration: {len(audio_np)/TARGET_SAMPLE_RATE:.2f}s")

        # Run transcription in a separate thread (as Whisper might block)
        result = await asyncio.to_thread(
            model.transcribe,
            audio_np,
            fp16=False, # Set to True if using CUDA and compatible GPU
            language=None # Auto-detect language
        )
        if result and result["text"]:
             # Strip leading/trailing spaces and dots common in streaming
            text = result["text"].strip()
            if text and text != '.':
                 print(f"🗣️ {text}") # Use print for immediate feedback
            else:
                logging.info("Transcription resulted in empty or insignificant text.")
        else:
            logging.info("Transcription result was empty.")

    except Exception as e:
        logging.error(f"❌ Transcription error: {e}")

async def ffmpeg_stderr_reader(stderr):
    """Reads and logs ffmpeg stderr."""
    while True:
        line = await stderr.readline()
        if not line:
            break
        logging.warning(f"ffmpeg stderr: {line.decode().strip()}")
    logging.info("ffmpeg stderr stream ended.")


async def audio_processor():
    """Periodically checks the buffer and triggers transcription."""
    global audio_buffer
    threshold_bytes = TARGET_SAMPLE_RATE * BYTES_PER_SAMPLE * TRANSCRIPTION_THRESHOLD_SECONDS
    while True:
        await asyncio.sleep(TRANSCRIPTION_THRESHOLD_SECONDS / 2) # Check periodically

        async with buffer_lock:
            if len(audio_buffer) >= threshold_bytes:
                # Take the chunk to process
                chunk_to_process = bytes(audio_buffer[:len(audio_buffer)]) # Make a copy
                # Keep remaining data (could be improved for less fragmentation)
                audio_buffer = audio_buffer[len(audio_buffer):]
                # audio_buffer = bytearray() # Simpler: just clear for now

                logging.debug(f"Buffer size before taking chunk: {len(chunk_to_process)}")
                logging.debug(f"Buffer size after taking chunk: {len(audio_buffer)}")

                if chunk_to_process:
                     # Don't block the processor loop, run transcription concurrently
                    asyncio.create_task(transcribe_audio_chunk(chunk_to_process))
            elif len(audio_buffer) > AUDIO_BUFFER_MAX_SIZE:
                 logging.warning(f"Audio buffer exceeded max size ({AUDIO_BUFFER_MAX_SIZE}), clearing...")
                 audio_buffer = bytearray() # Prevent runaway buffer growth


async def handler(websocket):
    """Handles WebSocket connections and audio streaming."""
    global audio_buffer
    client_address = websocket.remote_address
    logging.info(f"Client connected: {client_address}")

    ffmpeg_process = None
    stderr_reader_task = None

    try:
        # Start ffmpeg process to decode WebM stream and output raw PCM
        ffmpeg_command = [
            'ffmpeg',
            '-loglevel', 'warning', # Reduce ffmpeg chattiness, use stderr_reader for warnings/errors
            '-f', 'webm',      # Input format
            '-i', 'pipe:0',    # Read from stdin
            '-f', TARGET_FORMAT, # Output format (raw PCM)
            '-acodec', f'pcm_{TARGET_FORMAT}', # Output codec
            '-ar', str(TARGET_SAMPLE_RATE), # Output sample rate
            '-ac', str(TARGET_CHANNELS),    # Output channels (mono)
            'pipe:1'           # Output to stdout
        ]
        logging.info(f"Starting ffmpeg process: {' '.join(ffmpeg_command)}")

        ffmpeg_process = await asyncio.create_subprocess_exec(
            *ffmpeg_command,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )

        logging.info(f"ffmpeg process started (PID: {ffmpeg_process.pid})")

        # Create tasks to read stdout and stderr concurrently
        stderr_reader_task = asyncio.create_task(ffmpeg_stderr_reader(ffmpeg_process.stderr))

        async def stdout_reader():
            """Reads decoded WAV data from ffmpeg stdout and adds to buffer."""
            global audio_buffer
            while True:
                try:
                    wav_chunk = await ffmpeg_process.stdout.read(TARGET_SAMPLE_RATE * BYTES_PER_SAMPLE) # Read up to 1 second
                    if not wav_chunk:
                        logging.info("ffmpeg stdout EOF reached.")
                        break
                    async with buffer_lock:
                        audio_buffer.extend(wav_chunk)
                        # Optional: Add check here to prevent buffer exceeding an absolute max size
                        # if len(audio_buffer) > SOME_ABSOLUTE_MAX: handle overflow
                    logging.debug(f"Read {len(wav_chunk)} bytes from ffmpeg stdout. Buffer size: {len(audio_buffer)}")
                except asyncio.CancelledError:
                    logging.info("stdout_reader task cancelled.")
                    break
                except Exception as e:
                    logging.error(f"Error reading ffmpeg stdout: {e}")
                    break
            logging.info("stdout_reader task finished.")

        stdout_reader_task = asyncio.create_task(stdout_reader())

        # Receive messages (WebM chunks) from WebSocket and feed to ffmpeg's stdin
        async for message in websocket:
            if isinstance(message, bytes) and ffmpeg_process.stdin.is_closing() is False:
                logging.debug(f"Received {len(message)} bytes from websocket. Writing to ffmpeg stdin.")
                try:
                    ffmpeg_process.stdin.write(message)
                    await ffmpeg_process.stdin.drain()
                except (BrokenPipeError, ConnectionResetError):
                    logging.warning("ffmpeg stdin pipe closed unexpectedly.")
                    break
                except Exception as e:
                    logging.error(f"Error writing to ffmpeg stdin: {e}")
                    break
            else:
                logging.warning(f"Received non-bytes message or stdin closed: {type(message)}")

    except websockets.ConnectionClosed as e:
        logging.info(f"Client {client_address} disconnected: {e}")
    except Exception as e:
        logging.error(f"Handler error: {e}", exc_info=True)
    finally:
        logging.info(f"Cleaning up for client {client_address}...")
        # Close ffmpeg stdin gracefully
        if ffmpeg_process and ffmpeg_process.stdin and not ffmpeg_process.stdin.is_closing():
            try:
                logging.info("Closing ffmpeg stdin...")
                ffmpeg_process.stdin.close()
                await ffmpeg_process.stdin.wait_closed()
                logging.info("ffmpeg stdin closed.")
            except Exception as e:
                logging.warning(f"Error closing ffmpeg stdin: {e}")

        # Wait for reader tasks to finish
        if 'stdout_reader_task' in locals() and stdout_reader_task and not stdout_reader_task.done():
             stdout_reader_task.cancel()
             try:
                 await asyncio.wait_for(stdout_reader_task, timeout=2.0)
             except asyncio.TimeoutError:
                 logging.warning("Timeout waiting for stdout_reader task to finish.")
             except asyncio.CancelledError:
                 pass # Expected
        if stderr_reader_task and not stderr_reader_task.done():
            stderr_reader_task.cancel()
            try:
                await asyncio.wait_for(stderr_reader_task, timeout=2.0)
            except asyncio.TimeoutError:
                 logging.warning("Timeout waiting for stderr_reader task to finish.")
            except asyncio.CancelledError:
                pass # Expected

        # Wait for ffmpeg process to terminate
        if ffmpeg_process and ffmpeg_process.returncode is None:
            logging.info(f"Waiting for ffmpeg process (PID: {ffmpeg_process.pid}) to terminate...")
            try:
                await asyncio.wait_for(ffmpeg_process.wait(), timeout=5.0)
                logging.info(f"ffmpeg process terminated with code: {ffmpeg_process.returncode}")
            except asyncio.TimeoutError:
                logging.warning(f"Timeout waiting for ffmpeg process to terminate. Killing.")
                try:
                    ffmpeg_process.kill()
                    await ffmpeg_process.wait() # Ensure it's killed
                except ProcessLookupError:
                    logging.warning("Process already terminated.") # Race condition ok
                except Exception as e:
                    logging.error(f"Error killing ffmpeg process: {e}")
            except Exception as e:
                 logging.error(f"Error waiting for ffmpeg process: {e}")

        # Clear buffer for the next connection (or manage per connection if needed)
        async with buffer_lock:
            global audio_buffer
            audio_buffer = bytearray()
            logging.info("Global audio buffer cleared.")

        logging.info(f"Cleanup complete for client {client_address}.")


async def main():
    # Start the background audio processor task
    processor_task = asyncio.create_task(audio_processor())
    logging.info("Audio processor task started.")

    async with websockets.serve(handler, WEBSOCKET_HOST, WEBSOCKET_PORT):
        logging.info(f"WebSocket server started on ws://{WEBSOCKET_HOST}:{WEBSOCKET_PORT}")
        await asyncio.Future()  # Run forever

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logging.info("Server stopped by user.")