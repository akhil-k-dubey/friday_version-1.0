"""
voice.py — Voice I/O pipeline for FRIDAY.

IN  → sounddevice → Silero VAD → faster-whisper (RTX 4050) → text
OUT → Kokoro ONNX (local neural TTS, ~50-150ms/sentence) → sounddevice

Why Kokoro instead of Edge TTS:
  - 100% local — no network, no rate limits, no "No audio received" failures
  - ~50-150ms generation after first call (model stays warm in memory)
  - No ffplay subprocess needed — audio played directly via sounddevice
  - Same kokoro-v1.0.onnx + voices-v1.0.bin you already downloaded
"""
from __future__ import annotations

import os
import queue
import threading
import time
import warnings
from dataclasses import dataclass, field
from typing import Optional, Iterator

warnings.filterwarnings("ignore", message="pkg_resources is deprecated", category=UserWarning)

from ear import (
    np, sd, sr,
    setup_mic, setup_stt_backend,
    StreamingVoiceCapture, _try_import,
)

# ── Kokoro ONNX local TTS ─────────────────────────────────────────────────
_Kokoro = _try_import("kokoro_onnx", "Kokoro")

# Paths to local model files (already downloaded)
_KOKORO_MODEL  = os.path.join(os.path.dirname(__file__), "kokoro-v1.0.onnx")
_KOKORO_VOICES = os.path.join(os.path.dirname(__file__), "voices-v1.0.bin")

# Voice selection — Irish female (closest to Emily) or American female
# Available en voices: af_sky, af_bella, af_sarah, af_nicole, am_adam,
#                      am_michael, bf_emma, bf_isabella, bm_george, bm_lewis
FRIDAY_VOICE = "bf_emma"   # British female — sounds great, change if you like
FRIDAY_SPEED = 1.1          # Slightly faster than normal for snappier feel
FRIDAY_LANG  = "en-gb"     # British English (matches bf_emma perfectly)


@dataclass
class VoiceIO:
    """
    Full-duplex voice pipeline.
    IN  → StreamingVoiceCapture (Silero VAD + Whisper GPU)
    OUT → Kokoro ONNX (local neural TTS) → sounddevice
    """

    voice_mode:            bool  = True
    mic_enabled:           bool  = False
    tts_enabled:           bool  = True
    recognizer:            Optional[object] = None
    mic_backend:           str   = "none"
    stt_backend:           str   = "whisper"
    whisper_model_size:    str   = "small.en"
    stt_language:          str   = "en-IN"
    post_tts_cooldown_s:   float = 1.2
    debug:                 bool  = False
    whisper_model:         Optional[object] = None
    tts_effective_backend: str   = "none"

    _last_tts_time:     float           = field(default=0.0,                repr=False)
    _playback_stop:     threading.Event = field(default_factory=threading.Event, repr=False)
    _streaming_capture: Optional[StreamingVoiceCapture] = field(default=None, repr=False)
    _kokoro:            Optional[object] = field(default=None, repr=False)
    _kokoro_lock:       threading.Lock  = field(default_factory=threading.Lock, repr=False)
    _current_playback:  Optional[threading.Thread] = field(default=None, repr=False)

    def __post_init__(self) -> None:
        # ── Mic setup ──
        self.recognizer, self.mic_enabled, self.mic_backend = setup_mic(
            self.recognizer, self._debug
        )
        self.stt_backend, self.whisper_model = setup_stt_backend(
            self.whisper_model_size, self._debug
        )

        # ── Kokoro TTS setup ──
        self._init_kokoro()

        # ── Streaming capture ──
        if self.mic_enabled and self.mic_backend == "sounddevice_stream":
            self._streaming_capture = StreamingVoiceCapture(
                stt_backend=self.stt_backend,
                whisper_model=self.whisper_model,
                recognizer=self.recognizer,
                stt_language=self.stt_language,
                debug=self._debug,
            )
            self._streaming_capture.start()
            self._debug("Streaming capture initialized.")

            # Wait for Silero VAD to be ready so we never miss the wake word
            self._debug("Waiting for Silero VAD...")
            ready = self._streaming_capture.vad.wait_ready(timeout=10)
            if ready:
                self._debug("Silero VAD ready.")
            else:
                self._debug("Silero VAD not ready (RMS fallback active).")

    def _init_kokoro(self) -> None:
        """Load Kokoro ONNX model from local files. Keeps model in memory."""
        if _Kokoro is None:
            self._debug("kokoro-onnx not installed — TTS disabled. Run: pip install kokoro-onnx")
            self.tts_enabled = False
            self.tts_effective_backend = "none"
            return

        if not os.path.exists(_KOKORO_MODEL):
            self._debug(f"Kokoro model not found: {_KOKORO_MODEL}")
            self.tts_enabled = False
            self.tts_effective_backend = "none"
            return

        if not os.path.exists(_KOKORO_VOICES):
            self._debug(f"Kokoro voices not found: {_KOKORO_VOICES}")
            self.tts_enabled = False
            self.tts_effective_backend = "none"
            return

        try:
            self._debug("Loading Kokoro ONNX TTS model...")
            t0 = time.monotonic()
            self._kokoro = _Kokoro(_KOKORO_MODEL, _KOKORO_VOICES)
            elapsed = time.monotonic() - t0
            self._debug(f"TTS: Kokoro ONNX ({FRIDAY_VOICE}) loaded in {elapsed:.2f}s [OK]")
            self.tts_enabled = True
            self.tts_effective_backend = "kokoro"

            # Warm-up call so the first real speak() is fast
            self._debug("Warming up Kokoro TTS...")
            self._kokoro.create("Ready.", voice=FRIDAY_VOICE, speed=FRIDAY_SPEED, lang=FRIDAY_LANG)
            self._debug("Kokoro TTS warm-up complete.")

        except Exception as e:
            self._debug(f"Kokoro TTS failed to load: {e}")
            self._kokoro = None
            self.tts_enabled = False
            self.tts_effective_backend = "none"

    def _debug(self, msg: str) -> None:
        if self.debug:
            print(f"[voice-debug] {msg}")

    # ── Shutdown ──────────────────────────────────────────────────────────

    def shutdown(self) -> None:
        if self._streaming_capture:
            self._streaming_capture.stop()
            self._streaming_capture = None
        self._playback_stop.set()
        self.stop_playback()

    # ── Mic mute (for multi-message TTS blocks like boot) ─────────────

    def mute_mic(self) -> None:
        """Manually mute mic capture. Use for multi-speak blocks (boot, etc.).
        While muted, _play_audio() skips its own pause/resume.
        Call unmute_mic() when the block is done."""
        if self._streaming_capture:
            self._streaming_capture.pause_for_tts()

    def unmute_mic(self) -> None:
        """Unmute mic after a mute_mic() block. Flushes stale audio + settles."""
        if self._streaming_capture:
            self._streaming_capture.resume_after_tts()
        self._last_tts_time = time.monotonic()

    # ── Listen ────────────────────────────────────────────────────────────

    def listen(self) -> Optional[str]:
        """Block until user speaks; return transcript or None."""
        if not self.voice_mode or not self.mic_enabled or sr is None:
            return None
        if time.monotonic() - self._last_tts_time < self.post_tts_cooldown_s:
            return None

        if self._streaming_capture:
            self._streaming_capture.flush()
            return self._streaming_capture.listen_for_utterance()

        if self.mic_backend == "speech_recognition_mic" and self.recognizer is not None:
            try:
                with sr.Microphone() as source:
                    self.recognizer.adjust_for_ambient_noise(source, duration=0.2)
                    audio = self.recognizer.listen(source, timeout=5, phrase_time_limit=8)
                transcript = self.recognizer.recognize_google(audio, language=self.stt_language)
                return transcript.strip() if transcript else None
            except Exception as e:
                self._debug(f"STT mic fallback failed: {e}")
        return None

    # ── Core TTS generation ───────────────────────────────────────────────

    def _generate_tts(self, text: str) -> Optional[tuple]:
        """
        Generate audio via Kokoro ONNX — fully local, ~50-150ms per sentence.
        Returns (samples_float32, sample_rate) or None on failure.
        No network, no rate limits, no temp files.
        """
        if self._kokoro is None or not text.strip():
            return None

        # Strip characters that Kokoro doesn't handle well
        clean = text.replace("[", "").replace("]", "").replace("*", "").strip()
        if not clean:
            return None

        try:
            with self._kokoro_lock:
                t0 = time.monotonic()
                samples, sample_rate = self._kokoro.create(
                    clean,
                    voice=FRIDAY_VOICE,
                    speed=FRIDAY_SPEED,
                    lang=FRIDAY_LANG,
                )
                elapsed = (time.monotonic() - t0) * 1000
                self._debug(f"TTS Kokoro ({elapsed:.0f}ms): {clean[:50]!r}")
            return (samples, sample_rate)
        except Exception as e:
            self._debug(f"TTS Kokoro failed: {e}")
            return None

    # ── Playback ──────────────────────────────────────────────────────────

    def _play_audio(self, samples, sample_rate: int) -> None:
        """Play float32 audio array via sounddevice. Blocks until done."""
        if sd is None or samples is None:
            return
        # If mic is already muted (e.g., boot block), skip our own pause/resume
        # to avoid double-unmuting mid-sequence.
        already_muted = (
            self._streaming_capture is not None
            and self._streaming_capture._speaking
        )
        if self._streaming_capture and not already_muted:
            self._streaming_capture.pause_for_tts()
        try:
            sd.play(samples, sample_rate)
            sd.wait()
        except Exception as e:
            self._debug(f"Audio playback failed: {e}")
        finally:
            if self._streaming_capture and not already_muted:
                self._streaming_capture.resume_after_tts()

    # ── speak() — single-shot ─────────────────────────────────────────────

    def speak(self, text: str) -> None:
        """Generate and play TTS synchronously (boot messages, short replies)."""
        if not self.voice_mode or not text.strip() or not self.tts_enabled:
            return
        result = self._generate_tts(text)
        if result:
            samples, sr_rate = result
            try:
                self._play_audio(samples, sr_rate)
                self._last_tts_time = time.monotonic()
                self._debug(f"Spoke: {text[:60]}")
            except Exception as e:
                self._debug(f"speak() playback failed: {e}")
        else:
            self._debug(f"speak() TTS failed for: {text[:40]!r}")

    # ── speak_pipeline() — 3-thread parallel pipeline ────────────────────

    def speak_pipeline(self, sentences: Iterator[str]) -> None:
        """
        TRUE 3-thread parallel pipeline:

          Thread 1 (llm-feeder): pulls text from LLM generator → sentence_q
          Thread 2 (tts-worker): sentence_q → Kokoro generate → audio_q
          Main thread (player):  audio_q → sounddevice.play()

        While sentence N is PLAYING, sentence N+1 is being TTS-generated.
        Kokoro is ~50-150ms vs Edge TTS ~1500ms → massively lower latency.

        Filters [ACTION:...] tags — handled by skill dispatch.
        """
        if not self.voice_mode or not self.tts_enabled:
            return

        self._playback_stop.clear()
        sentence_q: queue.Queue = queue.Queue()
        audio_q:    queue.Queue = queue.Queue()

        # ── Thread 1: drain the LLM sentence generator ──
        def _sentence_feeder():
            for s in sentences:
                s = s.strip()
                if s and "[ACTION:" not in s:
                    sentence_q.put(s)
            sentence_q.put(None)  # sentinel

        threading.Thread(target=_sentence_feeder, daemon=True, name="llm-feeder").start()

        # ── Thread 2: TTS generation via Kokoro (~50-150ms each) ──
        def _tts_worker():
            while True:
                try:
                    text = sentence_q.get(timeout=60.0)
                except queue.Empty:
                    break
                if text is None:
                    break  # sentinel
                if self._playback_stop.is_set():
                    audio_q.put(None)
                    break
                result = self._generate_tts(text)
                audio_q.put(result)  # None on failure — player skips it
            audio_q.put(None)  # sentinel

        threading.Thread(target=_tts_worker, daemon=True, name="tts-worker").start()

        # ── Main thread: playback ──
        while True:
            try:
                result = audio_q.get(timeout=60.0)
            except queue.Empty:
                break
            if result is None:
                break  # sentinel
            if self._playback_stop.is_set():
                continue
            samples, sample_rate = result
            try:
                self._play_audio(samples, sample_rate)
                self._last_tts_time = time.monotonic()
            except Exception as e:
                self._debug(f"Pipeline play failed: {e}")

    def stop_playback(self) -> None:
        """Barge-in: stop sounddevice playback immediately."""
        self._playback_stop.set()
        try:
            if sd is not None:
                sd.stop()
        except Exception:
            pass