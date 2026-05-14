"""
main.py — FRIDAY Advanced AI Assistant.

Architecture:
  1. LLM streams tokens → accumulates sentences
  2. First sentence pre-gen starts immediately (latency hiding)
  3. TTS plays sentence-by-sentence with minimal gaps
  4. Skill actions are detected, executed, and results spoken
  5. Persistent memory logs every interaction

All tools are FREE FOREVER — no API keys required.
"""
from __future__ import annotations

import argparse
import sys
import warnings
from datetime import datetime
from typing import Optional

# Silence pkg_resources deprecation from webrtcvad
warnings.filterwarnings("ignore", message="pkg_resources is deprecated", category=UserWarning)

from assistant import Friday
from brain import Brain
from commands import is_shutdown_command, is_wake_command
from memory import PersistentMemory
from skills import SkillEngine
from voice import VoiceIO


def run(
    default_voice_mode: bool = True,
    debug_voice: bool = False,
    whisper_model_size: str = "small.en",
    stt_language: str = "en-IN",
    llm_backend: str = "ollama",
    llm_model: str = "auto",
    llm_base_url: str = "http://127.0.0.1:11434",
    llm_timeout: float = 60.0,
    llm_enabled: bool = True,
    always_on: bool = False,
) -> None:
    # ── Initialize components ─────────────────────────────────────────────
    persistent_memory = PersistentMemory()
    skill_engine = SkillEngine(
        debug_fn=lambda m: print(f"[skill-debug] {m}") if debug_voice else None
    )

    brain = Brain(
        enabled=llm_enabled and llm_backend == "ollama",
        base_url=llm_base_url,
        model=llm_model,
        timeout_s=llm_timeout,
        debug=debug_voice,
    )

    agent = Friday(
        name="friday",
        brain=brain,
        always_on=always_on,
        persistent_memory=persistent_memory,
        skills=skill_engine,
    )

    voice = VoiceIO(
        debug=debug_voice,
        whisper_model_size=whisper_model_size,
        stt_language=stt_language,
    )
    voice.voice_mode = default_voice_mode

    # ── Boot messages (mic muted for entire block to prevent echo) ──────
    online_message = (
        f"{agent.name.title()} online. "
        f"Systems restored at {datetime.now().strftime('%I:%M %p')}."
    )
    print(online_message)
    voice.mute_mic()  # ← mute mic for ALL boot speech
    voice.speak(online_message)

    if not always_on:
        standby_message = agent.standby()
        print(standby_message)
        voice.speak(standby_message)
    else:
        agent.activated = True
        ready_msg = f"{agent.name.title()} is ready and listening."
        print(ready_msg)
    voice.unmute_mic()  # ← unmute + flush + settle after all boot TTS done

    # ── Status line ───────────────────────────────────────────────────────
    if voice.voice_mode and voice.mic_enabled and voice.tts_enabled:
        status_line = (
            f"Voice: READY "
            f"(mic={voice.mic_backend}, STT={voice.stt_backend}, TTS={voice.tts_effective_backend})"
        )
    elif voice.voice_mode and (voice.mic_enabled or voice.tts_enabled):
        status_line = (
            "Voice: PARTIAL "
            f"(mic={voice.mic_backend}, STT={voice.stt_backend}, "
            f"TTS={voice.tts_effective_backend if voice.tts_enabled else 'off'})"
        )
    elif voice.voice_mode:
        status_line = "Voice: FALLBACK (text-only; mic/TTS unavailable)"
    else:
        status_line = "Voice: DISABLED (text-only mode)"
    print(status_line)

    model_name = brain.active_model or brain.model
    print(f"Brain: {'Ollama (' + model_name + ')' if brain.enabled else 'OFFLINE (rule-based)'}")
    print(f"Skills: {len(skill_engine.skills)} available")
    print("Say 'help' for commands or just talk naturally.\n")
    if debug_voice:
        print("[debug] Debug mode enabled.")

    # ── Main loop ─────────────────────────────────────────────────────────
    while agent.awake:
        user_input: Optional[str] = None

        # Voice input (streaming Silero VAD + whisper GPU)
        if voice.voice_mode and voice.mic_enabled:
            spoken = voice.listen()
            if spoken:
                print(f"You (voice) > {spoken}")
                user_input = spoken

        # Text input fallback (when mic unavailable or voice mode off)
        if not user_input and (not voice.voice_mode or not voice.mic_enabled):
            try:
                user_input = input("You > ")
            except (EOFError, KeyboardInterrupt):
                print("\nPowering down. Until next time.")
                break

        if not user_input or not user_input.strip():
            continue

        normalized = user_input.strip().lower()

        # ── Shutdown ──────────────────────────────────────────────────────
        if is_shutdown_command(normalized):
            response = "Powering down. Until next time."
            agent.awake = False
            print(f"{agent.name.title()} > {response}")
            voice.speak(response)
            persistent_memory.log_interaction(user_input, response)
            break

        # ── Wake word (standby mode) ──────────────────────────────────────
        if not agent.activated:
            if is_wake_command(normalized, agent.name):
                response = agent.wake()
                print(f"{agent.name.title()} > {response}")
                voice.speak(response)
            else:
                response = f"In standby. Say 'wake up {agent.name}' to activate."
                print(f"{agent.name.title()} > {response}")
                voice.speak(response)
            continue

        # ── Already active, wake word repeated ───────────────────────────
        if is_wake_command(normalized, agent.name):
            response = "I am already online and listening."
            print(f"{agent.name.title()} > {response}")
            voice.speak(response)
            continue

        # ── Voice mode toggles ────────────────────────────────────────────
        if normalized in ("voice on", "enable voice", "turn voice on"):
            voice.voice_mode = True
            response = "Voice mode enabled."
            print(f"{agent.name.title()} > {response}")
            voice.speak(response)
            continue

        if normalized in ("voice off", "disable voice", "turn voice off"):
            voice.voice_mode = False
            response = "Voice mode disabled."
            print(f"{agent.name.title()} > {response}")
            continue

        # ── LLM or rule-based response ────────────────────────────────────
        if agent.brain and agent.brain.enabled:
            _handle_brain_response(agent, voice, user_input, persistent_memory)
        else:
            response = agent.handle(user_input)
            print(f"{agent.name.title()} > {response}")
            voice.speak(response)
            persistent_memory.log_interaction(user_input, response)

        if not agent.awake:
            break

    # ── Cleanup ───────────────────────────────────────────────────────────
    voice.shutdown()
    print("FRIDAY shut down cleanly.")


def _handle_brain_response(
    agent: "Friday",
    voice: "VoiceIO",
    user_input: str,
    persistent_memory: "PersistentMemory",
) -> None:
    """
    Stream LLM response sentence-by-sentence with pipelined TTS.

    Skills are dispatched IMMEDIATELY as their [ACTION:...] tags are
    yielded by the LLM, so tasks complete before or during speech,
    not after all speech finishes.
    """
    sentence_gen = agent.brain.stream_sentences(
        user_input,
        memory_hint=" | ".join(agent.memory.recent(5)),
    )

    print(f"{agent.name.title()} > ", end="", flush=True)

    sentences:     list[str] = []
    full_response: str       = ""
    skill_results: list[str] = []

    def _sentence_stream():
        """Yield speakable sentences; execute skills instantly."""
        nonlocal full_response
        for sentence in sentence_gen:
            sentence = sentence.strip()
            if not sentence:
                continue

            full_response = (full_response + " " + sentence).strip()

            # Execute skill IMMEDIATELY if [ACTION:] tag found
            if "[ACTION:" in sentence:
                cleaned, results = agent.process_skill_actions(sentence)
                if results:
                    for result in results:
                        print(f"\n  [Skill] {result}")
                        skill_results.append(result)
                        yield result
            else:
                sentences.append(sentence)
                yield sentence

    # 3-thread pipeline: LLM feeder | TTS generator | Player — all parallel
    voice.speak_pipeline(_sentence_stream())

    # ── Memory ────────────────────────────────────────────────────────────
    agent.memory.add(user_input)
    if full_response:
        persistent_memory.log_interaction(user_input, full_response)
    for result in skill_results:
        if result:
            persistent_memory.log_interaction("[skill_result]", result)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="FRIDAY — Advanced AI Voice Assistant",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python main.py                  # Voice mode, Ollama LLM\n"
            "  python main.py --always-on      # Skip standby, always listen\n"
            "  python main.py --no-voice       # Text-only mode\n"
            "  python main.py --debug-voice    # Show diagnostics\n"
            "  python main.py --no-llm         # Rule-based, no Ollama needed\n"
        ),
    )
    parser.add_argument("--voice",    dest="voice", action="store_true",  help="Voice mode (default).")
    parser.add_argument("--no-voice", dest="voice", action="store_false", help="Text-only mode.")
    parser.add_argument("--always-on",  action="store_true", help="Skip standby, always listening.")
    parser.add_argument("--debug-voice", action="store_true", help="Print voice diagnostics.")
    parser.add_argument(
        "--whisper-model", default="small.en",
        help="Whisper model size: tiny.en, base.en, small.en (default), medium.en",
    )
    parser.add_argument(
        "--stt-language", default="en-IN",
        help="STT language locale (en-IN, en-US, hi-IN).",
    )
    parser.add_argument(
        "--llm-backend", choices=["ollama", "none"], default="ollama",
        help="LLM backend.",
    )
    parser.add_argument("--llm-model",    default="auto",                    help="Ollama model name.")
    parser.add_argument("--llm-base-url", default="http://127.0.0.1:11434", help="Ollama server URL.")
    parser.add_argument("--llm-timeout",  type=float, default=60.0,          help="LLM timeout (seconds).")
    parser.add_argument("--no-llm",       action="store_true",               help="Disable LLM.")
    parser.add_argument("--list-skills",  action="store_true",               help="List skills and exit.")
    parser.set_defaults(voice=True)

    args = parser.parse_args(sys.argv[1:])

    if args.list_skills:
        print(SkillEngine().list_skills())
        sys.exit(0)

    run(
        default_voice_mode=args.voice,
        debug_voice=args.debug_voice,
        whisper_model_size=args.whisper_model,
        stt_language=args.stt_language,
        llm_backend=args.llm_backend,
        llm_model=args.llm_model,
        llm_base_url=args.llm_base_url,
        llm_timeout=args.llm_timeout,
        llm_enabled=not args.no_llm,
        always_on=args.always_on,
    )