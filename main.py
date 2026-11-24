import importlib.util
import os
import re
import subprocess
import sys
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import sounddevice as sd
from faster_whisper import WhisperModel
from kokoro import KPipeline

# --- Configuration ---
BASE_DIR = Path(__file__).parent
COMMANDS_DIR = BASE_DIR / "commands"
SAMPLE_RATE = 16000
WAKE_PHRASE_REGEX = r"^(hey\s+meta|meta)[,\s]*"

# --- Initialize TTS & STT ---
print("Initializing Kokoro TTS...")
tts_pipeline = KPipeline(lang_code="a")  # 'a' = American English

print("Initializing Faster-Whisper...")
stt_model = WhisperModel("base.en", device="cpu", compute_type="int8")


def speak(text: str):
    """Synthesizes and plays speech using Kokoro TTS."""
    print(f"META: {text}")
    generator = tts_pipeline(text, voice="af_heart", speed=1.1)
    for _, _, audio in generator:
        sd.play(audio, samplerate=24000)
        sd.wait()


class PluginManager:
    def __init__(self, commands_dir: Path):
        self.commands_dir = commands_dir
        self.plugins = []

    def load_plugins(self, install_deps: bool = False):
        """Scans commands/ directory, parses config.xml, and loads Python modules."""
        self.plugins.clear()
        if not self.commands_dir.exists():
            self.commands_dir.mkdir(parents=True)
            return

        for folder in self.commands_dir.iterdir():
            if not folder.is_dir():
                continue

            config_path = folder / "config.xml"
            if not config_path.exists():
                continue

            try:
                tree = ET.parse(config_path)
                root = tree.getroot()

                nickname = root.findtext("nickname", default=folder.name).strip()
                address = root.findtext("address", default=folder.name).strip()
                keywords_raw = root.findtext("keywords", default="").strip()
                description = root.findtext("description", default="").strip()

                script_path = folder / f"{address}.py"
                req_path = folder / "requirements.txt"

                if not script_path.exists():
                    print(f"[!] Skipping '{nickname}': {script_path.name} not found.")
                    continue

                if install_deps and req_path.exists() and req_path.stat().st_size > 0:
                    print(f"[*] Installing dependencies for {nickname}...")
                    subprocess.run(
                        [sys.executable, "-m", "pip", "install", "-q", "-r", str(req_path)],
                        check=True,
                    )

                # Dynamically import the plugin module
                spec = importlib.util.spec_from_file_location(f"meta_cmd_{address}", script_path)
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)

                if not hasattr(module, "run"):
                    print(f"[!] Skipping '{nickname}': missing 'run(command_text, speak)' function.")
                    continue

                self.plugins.append({
                    "nickname": nickname,
                    "address": address,
                    "pattern": re.compile(keywords_raw, re.IGNORECASE),
                    "description": description,
                    "module": module,
                })
                print(f"[+] Loaded plugin: {nickname} (/{address})")

            except Exception as e:
                print(f"[!] Error loading plugin in {folder.name}: {e}")

    def route(self, command_text: str):
        """Matches the command against loaded plugin regexes and executes it."""
        for plugin in self.plugins:
            if plugin["pattern"].search(command_text):
                try:
                    # Pass speak() so plugins can give multi-step updates (e.g., "Committing... Done")
                    result = plugin["module"].run(command_text, speak)
                    if result:
                        speak(str(result))
                except Exception as e:
                    speak(f"Plugin {plugin['nickname']} encountered an error.")
                    print(f"[!] Plugin Error ({plugin['nickname']}): {e}")
                return

        speak(f"I heard '{command_text}', but no plugin matched that command.")


def listen_until_silence(threshold=0.015, silence_duration=1.2, max_duration=10.0) -> np.ndarray | None:
    """Listens to the mic, triggers when volume passes threshold, and stops after silence."""
    chunk_samples = int(SAMPLE_RATE * 0.1)  # 100ms chunks
    audio_buffer = []
    speaking = False
    silence_chunks = 0
    max_silence_chunks = int(silence_duration / 0.1)
    max_chunks = int(max_duration / 0.1)

    with sd.InputStream(samplerate=SAMPLE_RATE, channels=1, dtype="float32") as stream:
        while len(audio_buffer) < max_chunks:
            chunk, _ = stream.read(chunk_samples)
            chunk = chunk.flatten()
            rms = np.sqrt(np.mean(chunk**2))

            if rms > threshold:
                if not speaking:
                    speaking = True
                silence_chunks = 0
                audio_buffer.append(chunk)
            elif speaking:
                silence_chunks += 1
                audio_buffer.append(chunk)
                if silence_chunks >= max_silence_chunks:
                    break

    if not audio_buffer:
        return None
    return np.concatenate(audio_buffer)


def main():
    manager = PluginManager(COMMANDS_DIR)
    manager.load_plugins(install_deps=True)
    speak("Meta is online.")

    while True:
        audio = listen_until_silence()
        if audio is None or len(audio) < SAMPLE_RATE * 0.6:
            continue

        # Transcribe speech
        segments, _ = stt_model.transcribe(audio, beam_size=1, language="en")
        transcript = " ".join(seg.text for seg in segments).strip()

        if not transcript:
            continue

        print(f"HEARD: {transcript}")

        # Check for "Hey Meta" wake phrase
        match = re.match(WAKE_PHRASE_REGEX, transcript, re.IGNORECASE)
        if match:
            # Strip "Hey Meta" to isolate the actual command
            clean_command = transcript[match.end():].strip(" .?!")
            if not clean_command:
                speak("Yes?")
                continue

            print(f"COMMAND: {clean_command}")
            manager.route(clean_command)


if __name__ == "__main__":
    main()
