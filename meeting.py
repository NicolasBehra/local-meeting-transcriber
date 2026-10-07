#!/usr/bin/env python3
"""Meeting recorder & analyzer — Option A (100% local: Parakeet/faster-whisper + Ollama)."""

import argparse
import os

from dotenv import load_dotenv
load_dotenv()
import signal
import subprocess
import sys
import threading
import time
from datetime import datetime, timedelta
from pathlib import Path
from queue import Empty, Queue

import numpy as np
import ollama


def _preload_nvidia_libs():
    """Preload CUDA 12 libs from pip wheels (nvidia-*-cu12) so ctranslate2 and
    onnxruntime-gpu find them without LD_LIBRARY_PATH."""
    import ctypes
    import glob
    import site
    for sp in site.getsitepackages():
        for pattern in ("nvidia/cuda_runtime/lib/libcudart.so.*",
                        "nvidia/cublas/lib/libcublasLt.so.*", "nvidia/cublas/lib/libcublas.so.*",
                        "nvidia/curand/lib/libcurand.so.*", "nvidia/cufft/lib/libcufft.so.*",
                        "nvidia/cudnn/lib/libcudnn*.so.*"):
            for lib in sorted(glob.glob(os.path.join(sp, pattern))):
                try:
                    ctypes.CDLL(lib, mode=ctypes.RTLD_GLOBAL)
                except OSError:
                    pass


_preload_nvidia_libs()
from faster_whisper import WhisperModel
from rich.console import Console
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.text import Text

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
SAMPLE_RATE = 16_000
CHUNK_SECONDS = 30
OVERLAP_SECONDS = 2
CHUNK_SAMPLES = SAMPLE_RATE * CHUNK_SECONDS
OVERLAP_SAMPLES = SAMPLE_RATE * OVERLAP_SECONDS
ANALYSIS_INTERVAL = 150  # seconds (~2.5 min)
TRANSLATION_INTERVAL = 5  # seconds between translation batches
WHISPER_MODEL = "large-v3"
WHISPER_FALLBACK = "medium"
WHISPER_COMPUTE = "int8_float16"
PARAKEET_MODEL = "nemo-parakeet-tdt-0.6b-v3"
ASR_BACKENDS = ("parakeet", "whisper")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "ministral-3:3b")
OLLAMA_NUM_CTX = int(os.environ.get("OLLAMA_NUM_CTX", "16384"))
OLLAMA_KEEP_ALIVE = "30m"
DIARIZATION_MODEL = "pyannote/speaker-diarization-community-1"
LANGUAGE_DEFAULT = "fr"
SUPPORTED_LANGUAGES = {
    "fr": "français",
    "en": "English",
    "de": "Deutsch",
    "es": "español",
}
DIARIZATION_MODES = ("off", "simple", "advanced")

console = Console()


# ---------------------------------------------------------------------------
# ASR model pre-loading
# ---------------------------------------------------------------------------
def ensure_whisper_model() -> WhisperModel:
    """Load Whisper model in the main thread with fallback chain."""
    attempts = [
        (WHISPER_MODEL, "cuda", WHISPER_COMPUTE),
        (WHISPER_FALLBACK, "cuda", WHISPER_COMPUTE),
        (WHISPER_FALLBACK, "cpu", "int8"),
    ]
    last_error = None
    for model_name, device, compute in attempts:
        with console.status(f"Chargement du modele Whisper '{model_name}' ({device}, {compute})..."):
            try:
                model = WhisperModel(model_name, device=device, compute_type=compute)
                console.print(f"[green]Whisper '{model_name}' charge ({device}, {compute}).[/green]")
                return model
            except Exception as e:
                last_error = e
                console.print(f"[yellow]Echec '{model_name}' ({device}, {compute}): {e}[/yellow]")
    console.print("[red bold]Impossible de charger un modele Whisper.[/red bold]")
    console.print(f"[red]Derniere erreur: {last_error}[/red]")
    console.print("[red]Verifiez que faster-whisper est installe et que CUDA/CPU est disponible.[/red]")
    sys.exit(1)


def ensure_parakeet_model():
    """Load Parakeet TDT v3 (onnx-asr) with Silero VAD, GPU first then CPU."""
    import onnx_asr
    # Grow the GPU memory arena only as needed (default doubles it) to leave VRAM for Ollama
    cuda = ("CUDAExecutionProvider", {"arena_extend_strategy": "kSameAsRequested",
                                      "cudnn_conv_algo_search": "HEURISTIC"})
    attempts = [
        ("cuda", [cuda, "CPUExecutionProvider"]),
        ("cpu", ["CPUExecutionProvider"]),
    ]
    last_error = None
    for device, providers in attempts:
        with console.status(f"Chargement du modele Parakeet '{PARAKEET_MODEL}' ({device})..."):
            try:
                model = onnx_asr.load_model(PARAKEET_MODEL, providers=providers)
                vad = onnx_asr.load_vad("silero", providers=providers)
                # onnxruntime silently falls back to CPU if CUDA libs fail to load
                encoder = getattr(getattr(model, "asr", None), "_encoder", None)
                if encoder is not None and "CUDAExecutionProvider" not in encoder.get_providers():
                    device = "cpu"
                console.print(f"[green]Parakeet '{PARAKEET_MODEL}' charge ({device}).[/green]")
                return model.with_vad(vad, min_silence_duration_ms=500)
            except Exception as e:
                last_error = e
                console.print(f"[yellow]Echec Parakeet ({device}): {e}[/yellow]")
    console.print("[red bold]Impossible de charger Parakeet.[/red bold]")
    console.print(f"[red]Derniere erreur: {last_error}[/red]")
    console.print("[red]Essayez --asr whisper ou relancez ./setup.sh.[/red]")
    sys.exit(1)


# ---------------------------------------------------------------------------
# Diarization pipeline pre-loading (advanced mode)
# ---------------------------------------------------------------------------
def ensure_diarization_pipeline():
    """Load pyannote speaker-diarization pipeline. Returns None on failure."""
    token = os.environ.get("HF_TOKEN")
    if not token:
        console.print("[red]HF_TOKEN non defini. Necessaire pour pyannote (mode advanced).[/red]")
        console.print("[red]Creez un token sur https://huggingface.co/settings/tokens[/red]")
        sys.exit(1)
    try:
        import torch
        from pyannote.audio import Pipeline
    except ImportError:
        console.print("[red]pyannote.audio non installe. Lancez:[/red]")
        console.print("[red]  ./setup.sh --advanced[/red]")
        sys.exit(1)
    with console.status(f"Chargement du modele pyannote '{DIARIZATION_MODEL}'..."):
        try:
            pipeline = Pipeline.from_pretrained(DIARIZATION_MODEL, token=token)
            if torch.cuda.is_available():
                pipeline.to(torch.device("cuda"))
            console.print("[green]Pyannote speaker-diarization charge.[/green]")
            return pipeline
        except Exception as e:
            console.print(f"[red]Echec chargement pyannote: {e}[/red]")
            sys.exit(1)


# ---------------------------------------------------------------------------
# Audio source detection
# ---------------------------------------------------------------------------
def detect_sources() -> tuple[str, str]:
    """Return (microphone_source, monitor_source) using system defaults."""
    try:
        mic_result = subprocess.run(
            ["pactl", "get-default-source"],
            capture_output=True, text=True, check=True,
        )
        sink_result = subprocess.run(
            ["pactl", "get-default-sink"],
            capture_output=True, text=True, check=True,
        )
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        console.print(f"[red]Impossible de detecter les sources audio: {e}[/red]")
        sys.exit(1)
    mic = mic_result.stdout.strip()
    monitor = sink_result.stdout.strip() + ".monitor"
    if not mic or not monitor:
        console.print("[red]Impossible de detecter les sources audio par defaut.[/red]")
        console.print("Verifiez vos parametres son systeme.")
        sys.exit(1)
    return mic, monitor


# ---------------------------------------------------------------------------
# AudioCapture thread
# ---------------------------------------------------------------------------
class AudioCapture(threading.Thread):
    """Capture audio from one or two PulseAudio sources.

    - 1 source: capture directly (used in simple diarization mode).
    - 2 sources: mix with amix (used in off/advanced modes).
    """

    def __init__(self, sources: list[str], queue: Queue, stop_event: threading.Event,
                 source_label: str = ""):
        super().__init__(daemon=True)
        self.sources = sources
        self.queue = queue
        self.stop_event = stop_event
        self.source_label = source_label
        self.process: subprocess.Popen | None = None

    def run(self):
        if len(self.sources) == 1:
            cmd = [
                "ffmpeg", "-hide_banner", "-loglevel", "error",
                "-f", "pulse", "-i", self.sources[0],
                "-ac", "1", "-ar", str(SAMPLE_RATE),
                "-f", "s16le", "-acodec", "pcm_s16le",
                "pipe:1",
            ]
        else:
            cmd = [
                "ffmpeg", "-hide_banner", "-loglevel", "error",
                "-f", "pulse", "-i", self.sources[0],
                "-f", "pulse", "-i", self.sources[1],
                "-filter_complex", "amix=inputs=2:duration=longest",
                "-ac", "1", "-ar", str(SAMPLE_RATE),
                "-f", "s16le", "-acodec", "pcm_s16le",
                "pipe:1",
            ]
        self.process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        buffer = np.array([], dtype=np.int16)
        bytes_per_read = SAMPLE_RATE * 2  # 1 second of audio

        while not self.stop_event.is_set():
            raw = self.process.stdout.read(bytes_per_read)
            if not raw:
                break
            samples = np.frombuffer(raw, dtype=np.int16)
            buffer = np.concatenate([buffer, samples])
            while len(buffer) >= CHUNK_SAMPLES:
                chunk = buffer[:CHUNK_SAMPLES].astype(np.float32) / 32768.0
                self.queue.put((chunk, self.source_label))
                buffer = buffer[CHUNK_SAMPLES - OVERLAP_SAMPLES:]

        # Flush remaining buffer
        if len(buffer) > SAMPLE_RATE:  # at least 1s
            chunk = buffer.astype(np.float32) / 32768.0
            self.queue.put((chunk, self.source_label))

    def stop(self):
        if self.process and self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.process.kill()


# ---------------------------------------------------------------------------
# Transcriber thread
# ---------------------------------------------------------------------------
class Transcriber(threading.Thread):
    def __init__(self, queue: Queue, transcript_log: list, lock: threading.Lock,
                 stop_event: threading.Event, chunks_counter: list,
                 thread_errors: list, model,
                 language: str = "fr", asr_backend: str = "parakeet",
                 diarization: str = "off", diarization_pipeline=None):
        super().__init__(daemon=True)
        self.queue = queue
        self.transcript_log = transcript_log
        self.lock = lock
        self.stop_event = stop_event
        self.chunks_counter = chunks_counter
        self.thread_errors = thread_errors
        self.model = model
        self.language = language
        self.asr_backend = asr_backend
        self.diarization = diarization
        self.diarization_pipeline = diarization_pipeline

    def run(self):
        try:
            self._run_loop()
        except BaseException as e:
            self.thread_errors.append(f"Transcriber crash: {e}")
            self.stop_event.set()

    def _run_loop(self):
        while not self.stop_event.is_set():
            try:
                audio, source_label = self.queue.get(timeout=1)
            except Empty:
                continue
            try:
                self._transcribe_chunk(audio, source_label)
            except Exception as e:
                self.thread_errors.append(f"Transcriber chunk error: {e}")
            self.chunks_counter[0] += 1

        # Drain remaining chunks
        while not self.queue.empty():
            try:
                audio, source_label = self.queue.get_nowait()
            except Empty:
                break
            try:
                self._transcribe_chunk(audio, source_label)
            except Exception as e:
                self.thread_errors.append(f"Transcriber chunk error: {e}")
            self.chunks_counter[0] += 1

    def _transcribe_chunk(self, audio: np.ndarray, source_label: str):
        if self.diarization == "advanced" and self.diarization_pipeline is not None:
            self._transcribe_advanced(audio)
        else:
            self._transcribe_standard(audio, source_label)

    def _segments(self, audio: np.ndarray) -> list[dict]:
        """Transcribe a chunk into [{"text", "start", "end"}] with the selected ASR backend."""
        if self.asr_backend == "parakeet":
            segments = self.model.recognize(audio, sample_rate=SAMPLE_RATE)
        else:
            segments, _ = self.model.transcribe(
                audio, language=self.language, beam_size=5,
                vad_filter=True, vad_parameters=dict(min_silence_duration_ms=500),
            )
        result = []
        for s in segments:
            text = s.text.strip()
            if text:
                result.append({"text": text, "start": s.start, "end": s.end})
        return result

    def _transcribe_standard(self, audio: np.ndarray, source_label: str):
        segments = self._segments(audio)
        timestamp = datetime.now().strftime("%H:%M:%S")
        prefix = f"{source_label}: " if source_label else ""
        for seg in segments:
            entry = f"[{timestamp}] {prefix}{seg['text']}"
            with self.lock:
                self.transcript_log.append(entry)

    def _transcribe_advanced(self, audio: np.ndarray):
        import torch

        # Transcription with timestamps
        segments = self._segments(audio)
        if not segments:
            return

        # Pyannote diarization
        try:
            waveform = torch.from_numpy(audio).unsqueeze(0).float()
            result = self.diarization_pipeline(
                {"waveform": waveform, "sample_rate": SAMPLE_RATE}
            )
            # pyannote 4.x returns DiarizeOutput: prefer the exclusive (non-overlapping)
            # diarization, which aligns better with transcription segments
            diarization = getattr(result, "exclusive_speaker_diarization", None) \
                or getattr(result, "speaker_diarization", result)
        except Exception as e:
            # Diarization failed — fallback to transcription without speaker labels
            self.thread_errors.append(f"Pyannote error: {e}")
            timestamp = datetime.now().strftime("%H:%M:%S")
            for seg in segments:
                entry = f"[{timestamp}] {seg['text']}"
                with self.lock:
                    self.transcript_log.append(entry)
            return

        timestamp = datetime.now().strftime("%H:%M:%S")
        for seg in segments:
            speaker = self._find_speaker(diarization, seg["start"], seg["end"])
            entry = f"[{timestamp}] {speaker}: {seg['text']}"
            with self.lock:
                self.transcript_log.append(entry)

    @staticmethod
    def _find_speaker(diarization, start: float, end: float) -> str:
        best_speaker = "?"
        best_overlap = 0.0
        for segment, _, speaker in diarization.itertracks(yield_label=True):
            overlap = min(end, segment.end) - max(start, segment.start)
            if overlap > best_overlap:
                best_overlap = overlap
                best_speaker = speaker
        return best_speaker


# ---------------------------------------------------------------------------
# Ollama helper
# ---------------------------------------------------------------------------
def llm_chat(prompt: str, temperature: float, thread_errors: list) -> str:
    """Send a prompt to Ollama with an explicit context size (Ollama's default
    truncates long transcripts silently) and keep the model loaded between calls."""
    if len(prompt) > OLLAMA_NUM_CTX * 3:  # ~3 chars/token: prompt will be truncated
        warning = (f"Prompt trop long pour num_ctx={OLLAMA_NUM_CTX}: debut de transcription "
                   f"tronque par Ollama. Augmentez OLLAMA_NUM_CTX.")
        if warning not in thread_errors:
            thread_errors.append(warning)
    response = ollama.chat(
        model=OLLAMA_MODEL,
        messages=[{"role": "user", "content": prompt}],
        options={"num_ctx": OLLAMA_NUM_CTX, "temperature": temperature},
        keep_alive=OLLAMA_KEEP_ALIVE,
    )
    return response["message"]["content"]


# ---------------------------------------------------------------------------
# Analyzer thread (Ollama)
# ---------------------------------------------------------------------------
class Analyzer(threading.Thread):
    PROMPTS = {
        "fr": {
            "analyze": (
                "Voici la transcription {completeness} d'une reunion.\n"
                "Fournis une analyse {kind} en francais avec:\n"
                "- Resume des points discutes\n"
                "- Decisions prises\n"
                "- Actions a mener (avec responsable si mentionne)\n"
                "- Questions ouvertes ou points en suspens\n\n"
                "Transcription:\n{text}"
            ),
            "suggest": (
                "Tu es un assistant de reunion en temps reel. Voici la transcription recente d'une reunion en cours.\n"
                "Propose des suggestions concretes et utiles pour le participant :\n"
                "- Questions a poser pour clarifier un point flou\n"
                "- Points importants non abordes qui meriteraient de l'etre\n"
                "- Reformulations ou syntheses a proposer aux participants\n"
                "- Alertes si une decision semble prise sans consensus\n"
                "- Rappels d'actions ou engagements pris plus tot\n\n"
                "Sois concis et actionnable. Reponds en francais.\n\n"
                "Transcription recente:\n{text}"
            ),
            "completeness": ("complete", "partielle"),
            "kind": ("final", "intermediaire"),
        },
        "en": {
            "analyze": (
                "Here is the {completeness} transcript of a meeting.\n"
                "Provide a {kind} analysis in English with:\n"
                "- Summary of discussed points\n"
                "- Decisions made\n"
                "- Action items (with responsible person if mentioned)\n"
                "- Open questions or pending items\n\n"
                "Transcript:\n{text}"
            ),
            "suggest": (
                "You are a real-time meeting assistant. Here is the recent transcript of an ongoing meeting.\n"
                "Suggest concrete and useful actions for the participant:\n"
                "- Questions to ask to clarify unclear points\n"
                "- Important topics not yet addressed\n"
                "- Reformulations or summaries to propose to participants\n"
                "- Alerts if a decision seems made without consensus\n"
                "- Reminders of actions or commitments made earlier\n\n"
                "Be concise and actionable. Answer in English.\n\n"
                "Recent transcript:\n{text}"
            ),
            "completeness": ("complete", "partial"),
            "kind": ("final", "intermediate"),
        },
        "de": {
            "analyze": (
                "Hier ist die {completeness} Transkription eines Meetings.\n"
                "Erstelle eine {kind} Analyse auf Deutsch mit:\n"
                "- Zusammenfassung der besprochenen Punkte\n"
                "- Getroffene Entscheidungen\n"
                "- Massnahmen (mit Verantwortlichem, falls erwaehnt)\n"
                "- Offene Fragen oder ausstehende Punkte\n\n"
                "Transkription:\n{text}"
            ),
            "suggest": (
                "Du bist ein Echtzeit-Meeting-Assistent. Hier ist die aktuelle Transkription eines laufenden Meetings.\n"
                "Schlage konkrete und nuetzliche Aktionen fuer den Teilnehmer vor:\n"
                "- Fragen zur Klaerung unklarer Punkte\n"
                "- Wichtige noch nicht angesprochene Themen\n"
                "- Umformulierungen oder Zusammenfassungen fuer die Teilnehmer\n"
                "- Warnungen, falls eine Entscheidung ohne Konsens getroffen wurde\n"
                "- Erinnerungen an fruehere Massnahmen oder Zusagen\n\n"
                "Sei praegnant und umsetzbar. Antworte auf Deutsch.\n\n"
                "Aktuelle Transkription:\n{text}"
            ),
            "completeness": ("vollstaendige", "teilweise"),
            "kind": ("finale", "Zwischen-"),
        },
        "es": {
            "analyze": (
                "Aqui esta la transcripcion {completeness} de una reunion.\n"
                "Proporciona un analisis {kind} en espanol con:\n"
                "- Resumen de los puntos discutidos\n"
                "- Decisiones tomadas\n"
                "- Acciones a realizar (con responsable si se menciona)\n"
                "- Preguntas abiertas o puntos pendientes\n\n"
                "Transcripcion:\n{text}"
            ),
            "suggest": (
                "Eres un asistente de reuniones en tiempo real. Aqui esta la transcripcion reciente de una reunion en curso.\n"
                "Sugiere acciones concretas y utiles para el participante:\n"
                "- Preguntas para aclarar puntos confusos\n"
                "- Temas importantes aun no abordados\n"
                "- Reformulaciones o sintesis para proponer a los participantes\n"
                "- Alertas si una decision parece tomada sin consenso\n"
                "- Recordatorios de acciones o compromisos previos\n\n"
                "Se conciso y accionable. Responde en espanol.\n\n"
                "Transcripcion reciente:\n{text}"
            ),
            "completeness": ("completa", "parcial"),
            "kind": ("final", "intermedio"),
        },
    }

    def __init__(self, transcript_log: list, lock: threading.Lock,
                 analysis_log: list, suggestions_log: list, stop_event: threading.Event,
                 thread_errors: list, language: str = "fr"):
        super().__init__(daemon=True)
        self.transcript_log = transcript_log
        self.lock = lock
        self.analysis_log = analysis_log
        self.suggestions_log = suggestions_log
        self.stop_event = stop_event
        self.thread_errors = thread_errors
        self.last_analysis_index = 0
        self.language = language
        self.prompts = self.PROMPTS.get(language, self.PROMPTS["en"])

    def _analyze(self, text: str, is_final: bool = False) -> str:
        p = self.prompts
        completeness = p["completeness"][0] if is_final else p["completeness"][1]
        kind = p["kind"][0] if is_final else p["kind"][1]
        prompt = p["analyze"].format(completeness=completeness, kind=kind, text=text)
        try:
            return llm_chat(prompt, 0.2, self.thread_errors)
        except Exception as e:
            return f"[Erreur analyse: {e}]"

    def _suggest(self, text: str) -> str:
        prompt = self.prompts["suggest"].format(text=text)
        try:
            return llm_chat(prompt, 0.5, self.thread_errors)
        except Exception as e:
            return f"[Erreur suggestions: {e}]"

    def run(self):
        try:
            while not self.stop_event.is_set():
                for _ in range(int(ANALYSIS_INTERVAL)):
                    if self.stop_event.is_set():
                        return
                    time.sleep(1)
                self._run_analysis()
        except BaseException as e:
            self.thread_errors.append(f"Analyzer crash: {e}")
            self.stop_event.set()

    def _run_analysis(self, is_final: bool = False):
        with self.lock:
            current_log = list(self.transcript_log)
        if len(current_log) <= self.last_analysis_index:
            return
        new_text = "\n".join(current_log[self.last_analysis_index:])
        self.last_analysis_index = len(current_log)
        result = self._analyze(new_text, is_final=is_final)
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.analysis_log.append({"time": timestamp, "content": result, "final": is_final})
        if not is_final:
            suggestions = self._suggest(new_text)
            self.suggestions_log.append({"time": timestamp, "content": suggestions})

    def run_final(self):
        with self.lock:
            full_text = "\n".join(self.transcript_log)
        if not full_text.strip():
            return
        result = self._analyze(full_text, is_final=True)
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.analysis_log.append({"time": timestamp, "content": result, "final": True})


# ---------------------------------------------------------------------------
# Translator thread (Ollama)
# ---------------------------------------------------------------------------
class Translator(threading.Thread):
    def __init__(self, transcript_log: list, lock: threading.Lock,
                 translation_log: list, stop_event: threading.Event,
                 thread_errors: list, source_lang: str, target_lang: str):
        super().__init__(daemon=True)
        self.transcript_log = transcript_log
        self.lock = lock
        self.translation_log = translation_log
        self.stop_event = stop_event
        self.thread_errors = thread_errors
        self.source_lang = source_lang
        self.target_lang = target_lang
        self.last_translation_index = 0

    def _translate_batch(self, lines: list[str]) -> list[str]:
        src = SUPPORTED_LANGUAGES.get(self.source_lang, self.source_lang)
        tgt = SUPPORTED_LANGUAGES.get(self.target_lang, self.target_lang)
        text = "\n".join(lines)
        prompt = (
            f"Translate the following text from {src} to {tgt}. "
            f"Only output the translation, one translated line per input line. "
            f"Keep the timestamps in square brackets as-is.\n\n{text}"
        )
        try:
            result = llm_chat(prompt, 0.2, self.thread_errors).strip()
            return result.split("\n")
        except Exception as e:
            return [f"[Erreur traduction: {e}]"]

    def run(self):
        try:
            while not self.stop_event.is_set():
                with self.lock:
                    current_log = list(self.transcript_log)
                if len(current_log) > self.last_translation_index:
                    new_lines = current_log[self.last_translation_index:]
                    self.last_translation_index = len(current_log)
                    translated = self._translate_batch(new_lines)
                    self.translation_log.extend(translated)
                for _ in range(TRANSLATION_INTERVAL):
                    if self.stop_event.is_set():
                        return
                    time.sleep(1)
        except BaseException as e:
            self.thread_errors.append(f"Translator crash: {e}")
            self.stop_event.set()


# ---------------------------------------------------------------------------
# Rich display
# ---------------------------------------------------------------------------
def build_display(transcript_log: list, lock: threading.Lock, analysis_log: list,
                  suggestions_log: list, start_time: datetime, chunks_counter: list,
                  next_analysis: float, thread_errors: list,
                  translation_log: list | None = None) -> Layout:
    layout = Layout()
    layout.split_column(
        Layout(name="upper", ratio=3),
        Layout(name="status", size=3),
    )

    if translation_log is not None:
        # Layout with translation: 2x2 grid
        layout["upper"].split_column(
            Layout(name="top_row", ratio=1),
            Layout(name="bottom_row", ratio=1),
        )
        layout["top_row"].split_row(
            Layout(name="transcript", ratio=1),
            Layout(name="translation", ratio=1),
        )
        layout["bottom_row"].split_row(
            Layout(name="analysis", ratio=1),
            Layout(name="suggestions", ratio=1),
        )
    else:
        # Default layout: transcript left, analysis+suggestions right
        layout["upper"].split_row(
            Layout(name="transcript", ratio=2),
            Layout(name="right", ratio=1),
        )
        layout["right"].split_column(
            Layout(name="analysis", ratio=1),
            Layout(name="suggestions", ratio=1),
        )

    # Transcript panel
    with lock:
        lines = transcript_log[-20:]
    if thread_errors and not lines:
        error_lines = "\n".join(f"[red bold]{e}[/red bold]" for e in thread_errors)
        transcript_text = f"[red]ERREUR(S) THREAD:[/red]\n{error_lines}"
    elif lines:
        transcript_text = "\n".join(lines)
    else:
        transcript_text = "[dim]En attente de transcription...[/dim]"
    layout["transcript"].update(Panel(transcript_text, title="Transcription", border_style="green" if not thread_errors else "red"))

    # Translation panel (conditional)
    if translation_log is not None:
        trans_lines = translation_log[-20:]
        if trans_lines:
            translation_text = "\n".join(trans_lines)
        else:
            translation_text = "[dim]En attente de traduction...[/dim]"
        layout["translation"].update(Panel(translation_text, title="Traduction", border_style="yellow"))

    # Analysis panel
    if analysis_log:
        last = analysis_log[-1]
        tag = "FINAL" if last["final"] else last["time"]
        analysis_text = f"[bold][{tag}][/bold]\n{last['content']}"
    else:
        analysis_text = "[dim]Premiere analyse dans ~2.5 min...[/dim]"
    layout["analysis"].update(Panel(analysis_text, title="Analyse IA", border_style="blue"))

    # Suggestions panel
    if suggestions_log:
        last_s = suggestions_log[-1]
        suggestions_text = f"[bold][{last_s['time']}][/bold]\n{last_s['content']}"
    else:
        suggestions_text = "[dim]Suggestions apres la premiere analyse...[/dim]"
    layout["suggestions"].update(Panel(suggestions_text, title="Suggestions", border_style="magenta"))

    # Status bar
    elapsed = datetime.now() - start_time
    elapsed_str = str(timedelta(seconds=int(elapsed.total_seconds())))
    remaining = max(0, int(next_analysis - time.time()))
    error_indicator = "  |  [red bold]ERREUR THREAD[/red bold]" if thread_errors else ""
    status_line = (
        f"  Duree: {elapsed_str}  |  Chunks: {chunks_counter[0]}  |  "
        f"Prochaine analyse: {remaining}s  |  Ctrl+C pour arreter{error_indicator}"
    )
    status_text = Text.from_markup(
        status_line,
        style="bold white on dark_blue",
    )
    layout["status"].update(Panel(status_text, style="dark_blue"))

    return layout


# ---------------------------------------------------------------------------
# Markdown export
# ---------------------------------------------------------------------------
def export_markdown(transcript_log: list, analysis_log: list, start_time: datetime,
                    translation_log: list | None = None, translate_lang: str | None = None,
                    thread_errors: list | None = None):
    output_dir = Path("outputs")
    output_dir.mkdir(exist_ok=True)
    filename = f"meeting_{start_time.strftime('%Y-%m-%d_%Hh%M')}.md"
    filepath = output_dir / filename

    lines = [f"# Compte-rendu de reunion — {start_time.strftime('%Y-%m-%d %H:%M')}\n"]

    # Final analysis
    final = [a for a in analysis_log if a["final"]]
    if final:
        lines.append("## Resume final\n")
        lines.append(final[-1]["content"] + "\n")

    # Translation section
    if translation_log and translate_lang:
        lang_name = SUPPORTED_LANGUAGES.get(translate_lang, translate_lang)
        lines.append(f"## Traduction ({lang_name})\n")
        for entry in translation_log:
            lines.append(entry)
        lines.append("")

    # Full transcript
    lines.append("## Transcription complete\n")
    for entry in transcript_log:
        lines.append(entry)
    lines.append("")

    # Intermediate analyses
    intermediates = [a for a in analysis_log if not a["final"]]
    if intermediates:
        lines.append("## Analyses intermediaires\n")
        for a in intermediates:
            lines.append(f"### Analyse a {a['time']}\n")
            lines.append(a["content"] + "\n")

    # Thread errors
    if thread_errors:
        lines.append("## Erreurs\n")
        for err in thread_errors:
            lines.append(f"- {err}")
        lines.append("")

    filepath.write_text("\n".join(lines), encoding="utf-8")
    console.print(f"\n[green]Rapport exporte: {filepath}[/green]")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Meeting Recorder — Option A (Local)")
    parser.add_argument(
        "--diarization", choices=DIARIZATION_MODES, default="off",
        help="Mode d'identification des locuteurs: "
             "off = pas de distinction, "
             "simple = Moi (micro) / Interlocuteur (systeme), "
             "advanced = pyannote IA (necessite HF_TOKEN)",
    )
    parser.add_argument(
        "--asr", choices=ASR_BACKENDS, default="parakeet",
        help="Moteur de transcription: "
             "parakeet = NVIDIA Parakeet TDT 0.6B v3 (defaut, plus precis et rapide), "
             "whisper = faster-whisper large-v3",
    )
    parser.add_argument(
        "--language", choices=SUPPORTED_LANGUAGES.keys(), default=LANGUAGE_DEFAULT,
        help=f"Langue de transcription (defaut: {LANGUAGE_DEFAULT}). "
             + ", ".join(f"{k}={v}" for k, v in SUPPORTED_LANGUAGES.items()),
    )
    parser.add_argument(
        "--translate", choices=SUPPORTED_LANGUAGES.keys(), default=None,
        help="Langue cible de traduction en temps reel (optionnel, desactive par defaut). "
             "Ex: --language de --translate fr pour traduire de l'allemand vers le francais.",
    )
    args = parser.parse_args()
    language = args.language

    if args.translate and args.translate == language:
        console.print(f"[red]Erreur: --translate ({args.translate}) identique a --language ({language}). "
                       f"La traduction n'a pas de sens vers la meme langue.[/red]")
        sys.exit(1)

    console.print("[bold cyan]Meeting Recorder — Option A (Local)[/bold cyan]")
    asr_name = PARAKEET_MODEL if args.asr == "parakeet" else f"faster-whisper {WHISPER_MODEL}"
    console.print(f"[dim]Transcription: {asr_name} | Analyse: Ollama {OLLAMA_MODEL}[/dim]")
    translate_info = f" | Traduction: {SUPPORTED_LANGUAGES[args.translate]}" if args.translate else ""
    console.print(f"[dim]Diarisation: {args.diarization} | Langue: {SUPPORTED_LANGUAGES[language]}{translate_info}[/dim]\n")

    # Detect audio sources
    mic, monitor = detect_sources()
    console.print(f"[green]Micro:[/green] {mic}")
    console.print(f"[green]Monitor:[/green] {monitor}\n")

    # Pre-load ASR model in main thread
    asr_model = ensure_parakeet_model() if args.asr == "parakeet" else ensure_whisper_model()

    # Pre-load diarization pipeline if advanced mode
    diarization_pipeline = None
    if args.diarization == "advanced":
        diarization_pipeline = ensure_diarization_pipeline()

    # Shared state
    audio_queue: Queue = Queue(maxsize=50)
    transcript_log: list[str] = []
    analysis_log: list[dict] = []
    suggestions_log: list[dict] = []
    translation_log: list[str] | None = [] if args.translate else None
    thread_errors: list[str] = []
    lock = threading.Lock()
    stop_event = threading.Event()
    chunks_counter = [0]
    start_time = datetime.now()
    force_quit = [False]

    # Audio captures
    audio_captures: list[AudioCapture] = []
    if args.diarization == "simple":
        audio_captures.append(AudioCapture([mic], audio_queue, stop_event, source_label="Moi"))
        audio_captures.append(AudioCapture([monitor], audio_queue, stop_event, source_label="Interlocuteur"))
    else:
        audio_captures.append(AudioCapture([mic, monitor], audio_queue, stop_event))

    transcriber = Transcriber(
        audio_queue, transcript_log, lock, stop_event, chunks_counter,
        thread_errors, asr_model,
        language=language, asr_backend=args.asr,
        diarization=args.diarization, diarization_pipeline=diarization_pipeline,
    )
    analyzer = Analyzer(transcript_log, lock, analysis_log, suggestions_log, stop_event,
                        thread_errors, language=language)

    translator = None
    if args.translate:
        translator = Translator(
            transcript_log, lock, translation_log, stop_event,
            thread_errors, source_lang=language, target_lang=args.translate,
        )

    def signal_handler(sig, frame):
        if stop_event.is_set():
            console.print("\n[red]Arret force.[/red]")
            force_quit[0] = True
            export_markdown(transcript_log, analysis_log, start_time,
                            translation_log=translation_log, translate_lang=args.translate,
                            thread_errors=thread_errors)
            sys.exit(1)
        console.print("\n[yellow]Arret en cours... (Ctrl+C a nouveau pour forcer)[/yellow]")
        stop_event.set()
        for ac in audio_captures:
            ac.stop()

    signal.signal(signal.SIGINT, signal_handler)

    # Start threads
    for ac in audio_captures:
        ac.start()
    transcriber.start()
    analyzer.start()
    if translator:
        translator.start()

    next_analysis_time = time.time() + ANALYSIS_INTERVAL

    try:
        with Live(console=console, refresh_per_second=2) as live:
            while not stop_event.is_set():
                now = time.time()
                if now >= next_analysis_time:
                    next_analysis_time = now + ANALYSIS_INTERVAL
                # Thread liveness check
                if not transcriber.is_alive() and not stop_event.is_set():
                    if not any("Transcriber" in e for e in thread_errors):
                        thread_errors.append("Transcriber: thread mort de maniere inattendue")
                if not analyzer.is_alive() and not stop_event.is_set():
                    if not any("Analyzer" in e for e in thread_errors):
                        thread_errors.append("Analyzer: thread mort de maniere inattendue")
                if translator and not translator.is_alive() and not stop_event.is_set():
                    if not any("Translator" in e for e in thread_errors):
                        thread_errors.append("Translator: thread mort de maniere inattendue")
                display = build_display(
                    transcript_log, lock, analysis_log, suggestions_log,
                    start_time, chunks_counter, next_analysis_time, thread_errors,
                    translation_log=translation_log,
                )
                live.update(display)
                time.sleep(0.5)
    except KeyboardInterrupt:
        pass

    # Graceful shutdown
    stop_event.set()
    for ac in audio_captures:
        ac.stop()
    console.print("[cyan]Attente fin de transcription...[/cyan]")
    transcriber.join(timeout=30)

    if not force_quit[0]:
        console.print("[cyan]Generation du resume final...[/cyan]")
        analyzer.run_final()

    export_markdown(transcript_log, analysis_log, start_time,
                    translation_log=translation_log, translate_lang=args.translate,
                    thread_errors=thread_errors)


if __name__ == "__main__":
    main()
