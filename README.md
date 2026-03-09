# Meeting Recorder — Option A (100% Local)

Meeting recording with real-time transcription and AI analysis, entirely offline.

## What does this program do?

This script simultaneously captures sound from your microphone and system audio (what you hear through your speakers/headphones), mixes them into a single stream, transcribes speech to text in real time using Whisper on GPU, and periodically generates meeting analyses via a local LLM (Mistral through Ollama).

When you end the meeting (Ctrl+C), it produces a complete report in Markdown format containing:
- A final AI-generated summary
- The full timestamped transcript
- All intermediate analyses

**No data leaves your machine.** Everything runs locally: transcription (faster-whisper) and analysis (Ollama/Mistral).

## System requirements

| Component | Required |
|---|---|
| OS | Linux with PipeWire or PulseAudio |
| GPU | NVIDIA with CUDA (RTX 3060 6GB minimum recommended) |
| Python | 3.12+ (with `python3-venv`) |
| FFmpeg | Installed and in PATH |
| Ollama | Installed and running |
| pactl | Available (`pulseaudio-utils` package or included with PipeWire) |

## Installation

```bash
# 1. Create the venv and install Python dependencies (PyTorch + requirements.txt)
cd option_a_local
chmod +x setup.sh
./setup.sh

# 2. Install Ollama (if not already done)
curl -fsSL https://ollama.com/install.sh | sh

# 3. Download the Mistral model
ollama pull mistral
```

## Usage

### Basic launch

```bash
source .venv/bin/activate
python meeting.py
```

The script will:
1. Automatically detect your audio sources (system default mic + monitor)
2. Load the Whisper model on the GPU (with fallback to CPU if CUDA is unavailable)
3. Start recording and transcribing
4. Display a real-time dashboard in the terminal

### Transcription language

The `--language` option sets the transcription and analysis language (default: French):

```bash
python meeting.py                    # French (default)
python meeting.py --language en      # English
python meeting.py --language de      # German
python meeting.py --language es      # Spanish
```

Whisper transcription and AI analysis prompts automatically adapt to the chosen language.

### Real-time translation

The `--translate` option enables real-time translation of the transcript into another language:

```bash
# Meeting in German, translated to French
python meeting.py --language de --translate fr

# Meeting in Spanish, translated to English
python meeting.py --language es --translate en
```

Translation uses the same LLM (Ollama/Mistral) that performs the analysis. Every 5 seconds, new transcript lines are sent in batch for translation. The translated text appears in a dedicated panel alongside the original transcript. The Markdown export also includes a "Translation" section.

Note: `--translate` must differ from `--language` (translating to the same language is an error).

### Speaker identification (diarization)

The `--diarization` option enables speaker labeling in the transcript:

```bash
# No speaker identification (default)
python meeting.py --diarization off

# Simple mode: labels "Moi" (your mic) vs "Interlocuteur" (system audio)
python meeting.py --diarization simple

# Advanced mode: AI-based speaker identification via pyannote
python meeting.py --diarization advanced
```

**Simple mode** — Captures mic and system audio separately (two FFmpeg processes). Each line is prefixed with `Moi:` or `Interlocuteur:`. No extra dependency. Ideal for video calls where you need to distinguish yourself from remote participants.

**Advanced mode (pyannote)** — Uses the `pyannote/speaker-diarization-3.1` AI model to identify individual speakers (SPEAKER_00, SPEAKER_01, etc.), even when multiple people speak on the same audio channel. Extra setup required:

1. **Create a Hugging Face account** at https://huggingface.co/join, then generate a **Fine-grained** token at https://huggingface.co/settings/tokens with at least the permission "Read access to contents of all public gated repos you can access"
2. **Accept the model licenses** (mandatory, otherwise download fails):
   - https://huggingface.co/pyannote/speaker-diarization-3.1 → click "Agree and access repository"
   - https://huggingface.co/pyannote/segmentation-3.0 → click "Agree and access repository"
   - https://huggingface.co/pyannote/speaker-diarization-community-1 → click "Agree and access repository"
3. **Install the dependency**:
   ```bash
   pip install pyannote.audio
   ```
   This also installs `torch` and `torchaudio` (~2 GB). If PyTorch with CUDA is already in the venv, it will be reused.
4. **Set the token** via a `.env` file (recommended) or environment variable:
   ```bash
   # Option 1: .env file (recommended — loaded automatically)
   echo 'HF_TOKEN=hf_ABCDxxxxxxxx' > .env

   # Option 2: environment variable
   export HF_TOKEN="hf_ABCDxxxxxxxx"
   ```
   The `.env` file is automatically loaded at startup via `python-dotenv`. Make sure `.env` is in your `.gitignore` to avoid leaking your token.

The model (~300 MB) is downloaded on first use, then cached locally.

### Stopping the meeting

- **1st Ctrl+C**: graceful shutdown — stops recording, finishes transcribing remaining chunks, generates a final summary with Mistral, exports the Markdown file
- **2nd Ctrl+C**: forced shutdown — immediate partial save and exit

### Output file

The report is saved to:

```
option_a_local/outputs/meeting_YYYY-MM-DD_HHhMM.md
```

## Configuration

Constants are defined at the top of `meeting.py`:

| Constant | Default value | Description |
|---|---|---|
| `SAMPLE_RATE` | `16000` | Audio sample rate (Hz) |
| `CHUNK_SECONDS` | `30` | Duration of each audio chunk sent to Whisper (seconds) |
| `OVERLAP_SECONDS` | `2` | Overlap between chunks to avoid cutting words |
| `ANALYSIS_INTERVAL` | `150` | Interval between AI analyses (seconds, ~2.5 min) |
| `WHISPER_MODEL` | `"large-v3"` | Primary Whisper model (~3 GB VRAM in int8_float16) |
| `WHISPER_FALLBACK` | `"medium"` | Fallback model if large-v3 doesn't fit in VRAM |
| `WHISPER_COMPUTE` | `"int8_float16"` | Quantization type for Whisper |
| `OLLAMA_MODEL` | `"mistral"` | LLM model used through Ollama |
| `TRANSLATION_INTERVAL` | `5` | Interval between translation batches (seconds) |
| `LANGUAGE_DEFAULT` | `"fr"` | Default transcription language (overridden by `--language`) |

To change a parameter, edit the constant directly in the file. The language can also be changed at launch time with `--language`.

## Terminal interface

The display uses Rich and consists of four areas.

**Without `--translate`** (default):

```
+----------------------------------+------------------+
|                                  |                  |
|         Transcription            |   AI Analysis    |
|    (last 20 timestamped         |   (latest        |
|     lines)                       |    analysis)     |
|                                  +------------------+
|                                  |                  |
|                                  |   Suggestions    |
|                                  |   (actionable    |
|                                  |    advice)       |
+----------------------------------+------------------+
|  Duration: 0:05:23 | Chunks: 10 | Next: 42s        |
+----------------------------------------------------+
```

**With `--translate`**:

```
+------------------+------------------+
|  Transcription   |   Translation    |
|  (original)      |   (target lang)  |
+------------------+------------------+
|   AI Analysis    |   Suggestions    |
+------------------+------------------+
|  Duration: 0:05:23 | Chunks: 10    |
+------------------------------------+
```

- **Transcription panel** (green): the last 20 transcribed lines with `[HH:MM:SS]` timestamps
- **Translation panel** (yellow, only with `--translate`): the last 20 translated lines
- **AI Analysis panel** (blue): the content of the latest intermediate or final analysis
- **Suggestions panel** (magenta): real-time actionable suggestions — questions to ask, points to clarify, consensus alerts, action reminders
- **Status bar**: elapsed time, number of processed chunks, countdown to next analysis

---

## How the components work together

### Overall architecture

The script relies on **3 worker threads** coordinated by a **main thread**:

```
Mic ────┐                  ┌──────────────┐     ┌──────────────┐
        ├─► FFmpeg (amix) ─┤ AudioCapture ├────►│ Transcriber  │
Monitor─┘    subprocess    │   Thread 1   │Queue│   Thread 2   │
                           └──────────────┘     └──────┬───────┘
                                                       │ transcript_log
                                                       ▼
                                                ┌──────────────┐
                                                │   Analyzer   │
                                                │   Thread 3   │
                                                └──────┬───────┘
                                                       │ analysis_log
                                                       ▼
                                                ┌──────────────┐
                                                │  Main Thread │
                                                │ Rich Display │
                                                │ + Signal     │
                                                │ + Export MD  │
                                                └──────────────┘
```

### Data flow

1. **AudioCapture** launches an FFmpeg subprocess that captures the mic and monitor, mixes them, and sends the raw PCM stream through a pipe. The thread reads this pipe one second at a time, accumulates into a buffer, and slices into 30-second chunks with 2-second overlap. Each chunk is placed into a thread-safe `Queue`.

2. **Transcriber** consumes chunks from the Queue. For each chunk, it calls `faster-whisper` which returns text segments. Each segment is timestamped and appended to `transcript_log` (a shared list protected by a `threading.Lock`).

3. **Analyzer** wakes up every 2.5 minutes. It reads new lines from `transcript_log` (since its last index), sends them to Ollama/Mistral with a prompt requesting a structured analysis, and stores the result in `analysis_log`.

4. **Main thread** runs a Rich Live display at 2 FPS that reads `transcript_log` and `analysis_log` to update the panels. It also handles the Ctrl+C signal to orchestrate graceful shutdown.

### Synchronization

- **Queue** (max size 50): between AudioCapture and Transcriber. If Transcriber is too slow, the queue fills up and AudioCapture blocks (backpressure).
- **Lock**: protects `transcript_log` for writes (Transcriber) and reads (Analyzer, Main, export).
- **stop_event**: `threading.Event` shared by all threads. When set (Ctrl+C), each thread exits its loop cleanly.

### Shutdown sequence

```
Ctrl+C ─► signal_handler
           ├── stop_event.set()
           ├── audio.stop() (terminates FFmpeg)
           ▼
         Main loop exits
           ├── transcriber.join(30s) — drains the queue
           ├── analyzer.run_final() — summary of full transcript
           └── export_markdown() — writes the .md file
```

---

## How each tool works in detail

### FFmpeg — Audio capture

FFmpeg is launched as a subprocess with the following command:

```
ffmpeg -hide_banner -loglevel error \
  -f pulse -i <mic_source> \
  -f pulse -i <monitor_source> \
  -filter_complex "amix=inputs=2:duration=longest" \
  -ac 1 -ar 16000 \
  -f s16le -acodec pcm_s16le \
  pipe:1
```

- **`-f pulse`**: uses the PulseAudio backend (compatible with PipeWire via `pipewire-pulse`)
- **Two `-i` inputs**: the physical microphone and the sink "monitor" (= what comes out of the speakers). This captures both your voice and remote participants' voices
- **`amix=inputs=2:duration=longest`**: mixes both streams into one. `duration=longest` keeps the stream active as long as at least one source produces sound
- **`-ac 1 -ar 16000`**: converts to mono 16 kHz, the format expected by Whisper
- **`-f s16le -acodec pcm_s16le`**: outputs raw PCM, signed 16-bit little-endian integers (no WAV header, no compression)
- **`pipe:1`**: writes to stdout, which Python reads via `subprocess.PIPE`

**Source detection**: the `detect_sources()` function uses `pactl get-default-source` and `pactl get-default-sink` to find the system's current default input and output. The monitor source is derived by appending `.monitor` to the default sink name. This automatically follows system audio settings — if you switch from speakers to a Bluetooth headset, the script will use the headset's output monitor without any manual configuration.

### faster-whisper — Transcription

faster-whisper is a reimplementation of OpenAI's Whisper using CTranslate2 for inference. It's 4x faster than the original implementation with equivalent quality.

**Model loading** (done in the main thread before starting workers):
- First attempts `large-v3` on CUDA in `int8_float16` (~3 GB VRAM). This mode quantizes weights to int8 but keeps computations in float16, offering the best quality/speed ratio
- If it fails, tries `medium` on CUDA (~2 GB VRAM)
- If CUDA is unavailable, falls back to `medium` on CPU in `int8`
- If all attempts fail, exits with a clear error message

**Transcription**:
- Each 30-second chunk (numpy float32 array, normalized between -1 and 1) is passed to `model.transcribe()`
- Parameters: `language` set via `--language` flag (default `"fr"`, no automatic detection), `beam_size=5` (beam search for better quality)
- **VAD filter** (Silero Voice Activity Detection) is enabled to ignore silence and prevent hallucinated output (e.g., Whisper generating "Merci" or "Sous-titrage FR" on silent audio)
- The method returns an iterator of segments, each with a `.text` attribute
- Empty segments are filtered out

**Overlap**: each chunk shares its last 2 seconds with the beginning of the next chunk. This prevents cutting a word at the boundary between two chunks, as Whisper can "see" the context.

### Ollama / Mistral — AI analysis

Ollama is a runtime for running LLMs locally. Mistral 7B is a performant open-source model with good French support.

**How it works**:
- The Python `ollama` client communicates with the Ollama server which must be running in the background
- At each analysis, the script sends new transcript lines (since the last analysis) with a prompt structuring the response
- The prompt requests: summary of discussed points, decisions made, action items (with responsible person if mentioned), open questions
- The final analysis takes the complete transcript for a comprehensive summary

**Intermediate analyses**:
- Triggered every 150 seconds (~2.5 min)
- Process only new lines since the last analysis (`last_analysis_index`)
- Allow following the meeting's evolution in real time

**Final analysis**:
- Triggered at shutdown (Ctrl+C), after transcription is complete
- Resends the entire transcript for a complete and coherent summary

### Rich — Terminal display

Rich is a Python library for creating rich terminal interfaces.

**Components used**:
- **`Live`**: updates the display in-place (no scrolling) at 2 refreshes per second
- **`Layout`**: organizes the screen as a grid. Here: two columns on top (transcription 2/3, right panel 1/3 split into analysis + suggestions), one bar at the bottom
- **`Panel`**: frames each section with a title and colored border (green for transcription, blue for analysis, magenta for suggestions)
- **`Text`**: styled text for the status bar (bold white on dark blue background)

**Displayed data**:
- The last 20 transcript lines (to avoid cluttering the screen)
- The latest analysis (intermediate or final)
- The latest suggestions (actionable advice for the participant)
- Elapsed time, number of processed chunks, seconds remaining until next analysis

### NumPy — Signal processing

NumPy is used to efficiently manipulate raw audio data:

- **Reading from FFmpeg pipe**: raw bytes are converted to an `int16` array via `np.frombuffer()`
- **Accumulation buffer**: samples arrive in 1-second blocks and are concatenated with `np.concatenate()`
- **Chunk slicing**: when the buffer reaches 480,000 samples (30s at 16 kHz), the first 480,000 are extracted, and the buffer is shortened keeping the last 32,000 samples (2s overlap)
- **Normalization**: `int16 → float32` conversion by dividing by 32768.0, producing values between -1.0 and 1.0, the format expected by faster-whisper

### Markdown export

At the end of the meeting, the script generates a structured Markdown file:

```markdown
# Meeting report — 2026-03-06 14:30

## Final summary
[Content generated by Mistral on the full transcript]

## Full transcript
[HH:MM:SS] First transcribed sentence...
[HH:MM:SS] Second transcribed sentence...
...

## Intermediate analyses
### Analysis at 14:32:30
[Intermediate analysis content]

### Analysis at 14:35:00
[Intermediate analysis content]
```

The file is created in `outputs/` with a name based on the start date and time: `meeting_2026-03-06_14h30.md`.
