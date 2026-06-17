"""
╔══════════════════════════════════════════════════════════════╗
║            TEXT TO SPEECH SYSTEM                            ║
║            Language: English Only                           ║
║            Edge-TTS Neural + Piper Offline Fallback         ║
╚══════════════════════════════════════════════════════════════╝

INSTALL:
    pip install edge-tts pygame piper-tts

DOWNLOAD A PIPER VOICE (one time):
    python -m piper.download_voices en_US-lessac-medium

Place the downloaded .onnx + .onnx.json files in the same folder
as this script (or pass a full path via VoiceConfig.offline_model_path).
"""

import asyncio
import edge_tts
import os
import time
import re
import json
import hashlib
import logging
import shutil
import wave
import pygame

from pathlib import Path
from typing import Optional, List, Dict
from datetime import datetime
from dataclasses import dataclass
from enum import Enum


# ══════════════════════════════════════════════════════════════
# LOGGING
# ══════════════════════════════════════════════════════════════

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-8s | %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════
# VOICES (English only)
# ══════════════════════════════════════════════════════════════

class Voice(Enum):
    FEMALE = "en-US-AriaNeural"
    MALE   = "en-US-GuyNeural"


# ══════════════════════════════════════════════════════════════
# CONFIG
# ══════════════════════════════════════════════════════════════

@dataclass
class VoiceConfig:
    voice:              str   = Voice.FEMALE.value
    rate:               str   = "-10%"    # Speed: "-20%" slower, "+20%" faster
    pitch:              str   = "+0Hz"    # Pitch: "+5Hz" higher, "-5Hz" lower
    volume:             float = 1.0       # 0.0 - 1.0
    enable_cache:       bool  = True
    enable_history:     bool  = True
    chunk_size:         int   = 400
    retries:            int   = 3
    offline_model_path: str   = "en_US-lessac-medium.onnx"  # Piper voice model


# ══════════════════════════════════════════════════════════════
# SPEECH HISTORY
# ══════════════════════════════════════════════════════════════

class SpeechHistory:

    def __init__(self, path: str = "speech_history.json"):
        self.path = Path(path)
        self.history: List[Dict] = []
        self._load()

    def _load(self):
        if self.path.exists():
            try:
                with open(self.path, "r", encoding="utf-8") as f:
                    self.history = json.load(f)
                logger.info(f"Loaded {len(self.history)} history entries")
            except Exception:
                self.history = []

    def add(self, text: str, voice: str, success: bool):
        self.history.append({
            "timestamp": datetime.now().isoformat(),
            "text":      text,
            "voice":     voice,
            "success":   success
        })
        self._save()

    def _save(self):
        try:
            with open(self.path, "w", encoding="utf-8") as f:
                json.dump(self.history, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.error(f"History save error: {e}")

    def export_txt(self, path: str = "speech_history.txt"):
        with open(path, "w", encoding="utf-8") as f:
            for item in self.history:
                f.write(f"[{item['timestamp']}] {item['text']}\n")
        logger.info(f"History exported to {path}")


# ══════════════════════════════════════════════════════════════
# TEXT PROCESSOR
# ══════════════════════════════════════════════════════════════

class TextProcessor:

    @staticmethod
    def clean(text: str) -> str:
        if not text:
            return ""
        text = text.strip()
        text = re.sub(r'http\S+', '', text)           # Remove URLs
        text = re.sub(r'\s+', ' ', text)              # Collapse whitespace
        text = re.sub(r'[^\w\s.,!?:()-]', '', text)   # Strip weird chars
        # Ensure natural pauses after punctuation
        for p in ['.', '!', '?', ',']:
            text = text.replace(p, p + ' ')
        return re.sub(r'\s+', ' ', text).strip()

    @staticmethod
    def split_chunks(text: str, max_len: int = 400) -> List[str]:
        if len(text) <= max_len:
            return [text]
        sentences = re.split(r'([.!?])', text)
        chunks, current = [], ""
        for i in range(0, len(sentences), 2):
            sentence = sentences[i] + (sentences[i + 1] if i + 1 < len(sentences) else "")
            if len(current) + len(sentence) < max_len:
                current += (" " if current else "") + sentence
            else:
                if current:
                    chunks.append(current)
                current = sentence
        if current:
            chunks.append(current)
        return chunks


# ══════════════════════════════════════════════════════════════
# EDGE TTS ENGINE (online, primary)
# ══════════════════════════════════════════════════════════════

class EdgeTTSEngine:

    def __init__(self, config: VoiceConfig):
        self.config = config

    async def generate(self, text: str, output_file: str) -> bool:
        try:
            communicate = edge_tts.Communicate(
                text=text,
                voice=self.config.voice,
                rate=self.config.rate,
                pitch=self.config.pitch
            )
            await communicate.save(output_file)
            return os.path.exists(output_file)
        except Exception as e:
            logger.error(f"Edge-TTS error: {e}")
            return False


# ══════════════════════════════════════════════════════════════
# PIPER OFFLINE FALLBACK ENGINE (no internet needed)
# ══════════════════════════════════════════════════════════════

class OfflineTTSEngine:
    """
    Local neural TTS using Piper. Used when Edge-TTS fails
    (no internet, rate limited, etc).
    """

    def __init__(self, model_path: str):
        self.available = False
        self.voice = None
        try:
            from piper import PiperVoice
            if not os.path.exists(model_path):
                logger.warning(f"Piper model not found at: {model_path}")
                return
            self.voice = PiperVoice.load(model_path)
            self.available = True
            logger.info(f"Piper offline engine ready: {model_path}")
        except Exception as e:
            logger.warning(f"Piper TTS unavailable: {e}")

    def speak_to_file(self, text: str, output_file: str) -> bool:
        """Synthesize text to a WAV file. Returns True on success."""
        if not self.available:
            return False
        try:
            with wave.open(output_file, "wb") as wav_file:
                self.voice.synthesize_wav(text, wav_file)
            return os.path.exists(output_file)
        except Exception as e:
            logger.error(f"Piper synthesis error: {e}")
            return False


# ══════════════════════════════════════════════════════════════
# MAIN TTS SYSTEM
# ══════════════════════════════════════════════════════════════

class TextToSpeech:
    """
    English TTS System
    - Edge-TTS neural voices (online, free, primary)
    - Piper local neural voice (offline fallback)
    - Smart audio cache
    - Pause / Resume / Stop
    - Speech history
    """

    def __init__(self, config: VoiceConfig = None):
        self.config       = config or VoiceConfig()
        self.processor    = TextProcessor()
        self.edge_engine  = EdgeTTSEngine(self.config)
        self.offline      = OfflineTTSEngine(self.config.offline_model_path)
        self.history      = SpeechHistory()
        self.cache_dir    = Path("voice_cache")
        self.cache_dir.mkdir(exist_ok=True)

        self.is_paused    = False
        self.stop_signal  = False

        self.stats = {
            "total_spoken": 0,
            "cache_hits":   0,
            "edge_hits":    0,
            "offline_hits": 0,
            "failures":     0
        }

        self._init_pygame()
        logger.info(f"TTS Ready | Voice: {self.config.voice}")

    # ── Audio Init ────────────────────────────────────────────

    def _init_pygame(self):
        try:
            pygame.mixer.init(frequency=44100, size=-16, channels=2, buffer=4096)
            logger.info("Audio engine ready")
        except Exception as e:
            logger.error(f"Audio engine error: {e}")

    # ── Cache ─────────────────────────────────────────────────

    def _cache_path(self, text: str, ext: str = "mp3") -> str:
        key = hashlib.md5(
            f"{text}{self.config.voice}{self.config.rate}{self.config.pitch}".encode()
        ).hexdigest()
        return str(self.cache_dir / f"{key}.{ext}")

    # ── Playback (works for both mp3 from Edge and wav from Piper) ──

    def _play(self, file_path: str) -> bool:
        try:
            pygame.mixer.music.load(file_path)
            pygame.mixer.music.set_volume(self.config.volume)
            pygame.mixer.music.play()
            while pygame.mixer.music.get_busy():
                if self.stop_signal:
                    pygame.mixer.music.stop()
                    break
                while self.is_paused:
                    time.sleep(0.1)
                time.sleep(0.05)
            return True
        except Exception as e:
            logger.error(f"Playback error: {e}")
            return False

    # ── Generate (Edge) ──────────────────────────────────────

    async def _generate(self, text: str, output_file: str) -> bool:
        for attempt in range(self.config.retries):
            if await self.edge_engine.generate(text, output_file):
                self.stats["edge_hits"] += 1
                return True
            logger.warning(f"Retry {attempt + 1}/{self.config.retries}")
            await asyncio.sleep(1)
        return False

    # ── Speak ─────────────────────────────────────────────────

    def speak(self, text: str) -> bool:
        if not text:
            return False

        self.stop_signal = False

        try:
            cleaned = self.processor.clean(text)
            print(f"\n🤖 {cleaned}\n")

            chunks = self.processor.split_chunks(cleaned, self.config.chunk_size)
            success = True

            for chunk in chunks:
                if self.stop_signal:
                    break

                cache_path = self._cache_path(chunk, "mp3")

                if self.config.enable_cache and os.path.exists(cache_path):
                    logger.info("Cache hit")
                    self.stats["cache_hits"] += 1
                    success &= self._play(cache_path)
                    continue

                generated = asyncio.run(self._generate(chunk, cache_path))

                if generated:
                    success &= self._play(cache_path)
                else:
                    logger.warning("Edge-TTS failed — using Piper offline fallback")
                    offline_path = self._cache_path(chunk, "wav")

                    if self.config.enable_cache and os.path.exists(offline_path):
                        success &= self._play(offline_path)
                    elif self.offline.speak_to_file(chunk, offline_path):
                        self.stats["offline_hits"] += 1
                        success &= self._play(offline_path)
                    else:
                        logger.error("Both Edge-TTS and Piper failed for this chunk")
                        success = False

                time.sleep(0.1)

            if self.config.enable_history:
                self.history.add(cleaned, self.config.voice, success)

            self.stats["total_spoken"] += 1
            return success

        except Exception as e:
            logger.error(f"Speak error: {e}")
            self.stats["failures"] += 1
            return False

    # ── Controls ──────────────────────────────────────────────

    def stop(self):
        self.stop_signal = True
        pygame.mixer.music.stop()
        logger.info("Stopped")

    def pause(self):
        self.is_paused = True
        pygame.mixer.music.pause()
        logger.info("Paused")

    def resume(self):
        self.is_paused = False
        pygame.mixer.music.unpause()
        logger.info("Resumed")

    # ── Settings ──────────────────────────────────────────────

    def set_voice(self, voice: Voice):
        self.config.voice = voice.value
        self.edge_engine.config = self.config
        logger.info(f"Voice: {voice.name}")

    def set_rate(self, rate: str):
        """rate: '-20%' slower, '+20%' faster"""
        self.config.rate = rate
        self.edge_engine.config = self.config

    def set_pitch(self, pitch: str):
        """pitch: '+5Hz' higher, '-5Hz' lower"""
        self.config.pitch = pitch
        self.edge_engine.config = self.config

    def set_volume(self, volume: float):
        self.config.volume = max(0.0, min(1.0, volume))

    def clear_cache(self):
        try:
            shutil.rmtree(self.cache_dir)
            self.cache_dir.mkdir(exist_ok=True)
            logger.info("Cache cleared")
        except Exception as e:
            logger.warning(f"Cache clear warning: {e}")

    # ── Stats ─────────────────────────────────────────────────

    def print_stats(self):
        print(f"\n{'='*40}")
        print("📊 TTS STATS")
        print(f"{'='*40}")
        for k, v in self.stats.items():
            print(f"  {k:<20}: {v}")
        print(f"{'='*40}\n")


# ══════════════════════════════════════════════════════════════
# SIMPLE GLOBAL HELPER
# ══════════════════════════════════════════════════════════════

_tts: Optional[TextToSpeech] = None

def speak(text: str):
    """One-liner usage: speak('Hello world')"""
    global _tts
    if _tts is None:
        _tts = TextToSpeech()
    return _tts.speak(text)


# ══════════════════════════════════════════════════════════════
# MAIN — TESTING
# ══════════════════════════════════════════════════════════════

if __name__ == "__main__":

    print("\n" + "="*60)
    print("🎙️  ENGLISH TEXT TO SPEECH SYSTEM (Edge + Piper fallback)")
    print("="*60 + "\n")

    tts = TextToSpeech(VoiceConfig(
        voice=Voice.FEMALE.value,
        rate="-10%",
        volume=1.0,
        offline_model_path="en_US-lessac-medium.onnx"  # change path if needed
    ))

    # TEST 1 — Female voice
    print("TEST 1: Female voice")
    tts.speak("Hello! I am your AI assistant. How can I help you today?")

    # TEST 2 — Male voice
    print("TEST 2: Male voice")
    tts.set_voice(Voice.MALE)
    tts.speak("This is the male voice. Sounds different, right?")

    # TEST 3 — Speed change
    print("TEST 3: Slower speed")
    tts.set_rate("-30%")
    tts.speak("This sentence is spoken at a slower pace.")
    tts.set_rate("-10%")  # Reset

    # TEST 4 — Long text (auto chunking)
    print("TEST 4: Long text with auto chunking")
    tts.set_voice(Voice.FEMALE)
    tts.speak(
        "This is a longer piece of text to test the chunking system. "
        "The system will automatically split this into smaller parts. "
        "Each part is converted to audio separately, then played back seamlessly. "
        "This allows the system to handle very long responses without any issues."
    )

    # TEST 5 — Cache test (same text should play instantly)
    print("TEST 5: Cache test (should be instant)")
    tts.speak("Hello! I am your AI assistant. How can I help you today?")

    tts.print_stats()
    tts.history.export_txt()

    print("\n✅ ALL TESTS COMPLETE")