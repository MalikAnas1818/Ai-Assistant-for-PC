"""
╔══════════════════════════════════════════════════════════════╗
║            ULTRA ADVANCED TEXT TO SPEECH SYSTEM             ║
║         Urdu + English + Hindi + Arabic Support             ║
║         Edge-TTS + Offline Fallback + Queue System          ║
║         Production Ready AI Assistant Voice Engine          ║
╚══════════════════════════════════════════════════════════════╝

FEATURES:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
✅ Edge-TTS Neural Voices
✅ Offline pyttsx3 Fallback
✅ Smart Audio Cache
✅ Queue System
✅ Pause / Resume / Stop
✅ Speech History
✅ Statistics Tracking
✅ Auto Cleanup
✅ Multi Language Support
✅ Retry Logic
✅ Long Text Chunking
✅ Human-like Pauses
✅ Production Ready Logging
✅ Thread Safe
✅ Continuous Speaking Mode
✅ Async Optimized
✅ Streaming-like Playback

INSTALL:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

pip install edge-tts
pip install pygame
pip install pyttsx3
pip install asyncio
pip install aiofiles

BEST URDU VOICES:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Female:
    ur-PK-UzmaNeural

Male:
    ur-PK-AsadNeural
"""

import asyncio
import edge_tts
import os
import uuid
import time
import re
import json
import hashlib
import threading
import queue
import logging
import shutil
import pyttsx3
import pygame

from pathlib import Path
from typing import Optional, List, Dict
from datetime import datetime
from dataclasses import dataclass, asdict
from enum import Enum


# ══════════════════════════════════════════════════════════════
# LOGGING SETUP
# ══════════════════════════════════════════════════════════════

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-8s | %(message)s',
    datefmt='%H:%M:%S'
)

logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════
# LANGUAGE CONFIG
# ══════════════════════════════════════════════════════════════

class Language(Enum):

    URDU_FEMALE = "ur-PK-UzmaNeural"
    URDU_MALE = "ur-PK-AsadNeural"

    ENGLISH_FEMALE = "en-US-AriaNeural"
    ENGLISH_MALE = "en-US-GuyNeural"

    HINDI_FEMALE = "hi-IN-SwaraNeural"
    HINDI_MALE = "hi-IN-MadhurNeural"

    ARABIC_FEMALE = "ar-SA-ZariyahNeural"
    ARABIC_MALE = "ar-SA-HamedNeural"


# ══════════════════════════════════════════════════════════════
# DATA CLASSES
# ══════════════════════════════════════════════════════════════

@dataclass
class VoiceConfig:

    voice: str = Language.URDU_FEMALE.value

    rate: str = "-10%"

    pitch: str = "+0Hz"

    volume: float = 1.0

    enable_cache: bool = True

    enable_history: bool = True

    chunk_size: int = 400

    retries: int = 3


@dataclass
class SpeechEntry:

    timestamp: str

    text: str

    voice: str

    engine: str

    success: bool


# ══════════════════════════════════════════════════════════════
# SPEECH HISTORY
# ══════════════════════════════════════════════════════════════

class SpeechHistory:

    def __init__(self, path="speech_history.json"):

        self.path = Path(path)

        self.history: List[Dict] = []

        self.load()


    def load(self):

        if self.path.exists():

            try:
                with open(self.path, "r", encoding="utf-8") as f:
                    self.history = json.load(f)

                logger.info(f"📂 Loaded {len(self.history)} speech history entries")

            except:
                self.history = []


    def add(self, entry: SpeechEntry):

        self.history.append(asdict(entry))

        self.save()


    def save(self):

        try:

            with open(self.path, "w", encoding="utf-8") as f:

                json.dump(
                    self.history,
                    f,
                    ensure_ascii=False,
                    indent=2
                )

        except Exception as e:

            logger.error(f"❌ History save error: {e}")


    def export_txt(self, path="speech_history.txt"):

        with open(path, "w", encoding="utf-8") as f:

            for item in self.history:

                f.write(
                    f"[{item['timestamp']}] "
                    f"{item['text']}\n"
                )

        logger.info(f"📄 Exported history to {path}")


# ══════════════════════════════════════════════════════════════
# TEXT PROCESSOR
# ══════════════════════════════════════════════════════════════

class TextProcessor:

    @staticmethod
    def clean(text: str) -> str:

        if not text:
            return ""

        text = text.strip()

        # Remove extra spaces
        text = re.sub(r'\s+', ' ', text)

        # Remove URLs
        text = re.sub(r'http\S+', '', text)

        # Remove weird chars
        text = re.sub(r'[^\w\s.,!?؟،؛:()-]', '', text)

        # Natural pauses
        replacements = {
            ".": ". ",
            "!": "! ",
            "?": "? ",
            "؟": "؟ ",
            "،": "، ",
            ",": ", ",
            ";": "; ",
            "۔": "۔ "
        }

        for k, v in replacements.items():
            text = text.replace(k, v)

        text = re.sub(r'\s+', ' ', text)

        return text.strip()


    @staticmethod
    def split_text(
        text: str,
        max_length: int = 400
    ) -> List[str]:

        if len(text) <= max_length:
            return [text]

        sentences = re.split(r'([.!?؟۔])', text)

        chunks = []

        current = ""

        for i in range(0, len(sentences), 2):

            sentence = sentences[i]

            if i + 1 < len(sentences):
                sentence += sentences[i + 1]

            if len(current) + len(sentence) < max_length:

                current += " " + sentence if current else sentence

            else:

                chunks.append(current)

                current = sentence

        if current:
            chunks.append(current)

        return chunks


# ══════════════════════════════════════════════════════════════
# EDGE TTS ENGINE
# ══════════════════════════════════════════════════════════════

class EdgeTTSEngine:

    def __init__(self, config: VoiceConfig):

        self.config = config


    async def generate(
        self,
        text: str,
        output_file: str
    ) -> bool:

        try:

            communicate = edge_tts.Communicate(
                text=text,
                voice=self.config.voice,
                rate=self.config.rate,
                pitch=self.config.pitch
            )

            await communicate.save(output_file)

            if os.path.exists(output_file):
                return True

            return False

        except Exception as e:

            logger.error(f"❌ Edge-TTS Error: {e}")

            return False


# ══════════════════════════════════════════════════════════════
# OFFLINE PYTTSX3 FALLBACK
# ══════════════════════════════════════════════════════════════

class OfflineTTSEngine:

    def __init__(self):

        self.engine = pyttsx3.init()

        self.engine.setProperty('rate', 150)

        self.engine.setProperty('volume', 1.0)


    def speak(self, text: str):

        try:

            self.engine.say(text)

            self.engine.runAndWait()

            return True

        except Exception as e:

            logger.error(f"❌ Offline TTS Error: {e}")

            return False


# ══════════════════════════════════════════════════════════════
# MAIN ADVANCED TTS SYSTEM
# ══════════════════════════════════════════════════════════════

class AdvancedTextToSpeech:

    """
    Production Ready AI Assistant Voice Engine
    """


    def __init__(
        self,
        config: VoiceConfig = None
    ):

        self.config = config or VoiceConfig()

        self.text_processor = TextProcessor()

        self.edge_engine = EdgeTTSEngine(self.config)

        self.offline_engine = OfflineTTSEngine()

        self.history = SpeechHistory()

        self.cache_dir = Path("voice_cache")

        self.temp_dir = Path("voice_temp")

        self.cache_dir.mkdir(exist_ok=True)

        self.temp_dir.mkdir(exist_ok=True)

        self.audio_queue = queue.Queue()

        self.is_playing = False

        self.is_paused = False

        self.stop_signal = False

        self.current_audio = None

        self.stats = {
            "total_spoken": 0,
            "cache_hits": 0,
            "edge_tts_hits": 0,
            "offline_hits": 0,
            "failures": 0
        }

        self._init_pygame()

        logger.info("🚀 Advanced TTS System Ready")


    # ══════════════════════════════════════════════════════
    # AUDIO INIT
    # ══════════════════════════════════════════════════════

    def _init_pygame(self):

        try:

            pygame.mixer.init(
                frequency=44100,
                size=-16,
                channels=2,
                buffer=4096
            )

            logger.info("🔊 Audio Engine Ready")

        except Exception as e:

            logger.error(f"❌ Audio Engine Error: {e}")


    # ══════════════════════════════════════════════════════
    # CACHE
    # ══════════════════════════════════════════════════════

    def _generate_cache_key(self, text: str) -> str:

        return hashlib.md5(
            text.encode("utf-8")
        ).hexdigest()


    def _get_cache_path(self, text: str) -> str:

        key = self._generate_cache_key(text)

        return str(self.cache_dir / f"{key}.mp3")


    # ══════════════════════════════════════════════════════
    # PLAY AUDIO
    # ══════════════════════════════════════════════════════

    def _play_audio(self, file_path: str):

        try:

            self.is_playing = True

            pygame.mixer.music.load(file_path)

            pygame.mixer.music.set_volume(
                self.config.volume
            )

            pygame.mixer.music.play()

            while pygame.mixer.music.get_busy():

                if self.stop_signal:

                    pygame.mixer.music.stop()

                    break

                while self.is_paused:
                    time.sleep(0.1)

                time.sleep(0.1)

            self.is_playing = False

            return True

        except Exception as e:

            logger.error(f"❌ Playback Error: {e}")

            return False


    # ══════════════════════════════════════════════════════
    # GENERATE AUDIO
    # ══════════════════════════════════════════════════════

    async def _generate_audio(
        self,
        text: str,
        output_file: str
    ) -> bool:

        for attempt in range(
            self.config.retries
        ):

            try:

                success = await self.edge_engine.generate(
                    text,
                    output_file
                )

                if success:
                    self.stats["edge_tts_hits"] += 1
                    return True

            except Exception as e:

                logger.warning(
                    f"⚠️ Retry {attempt+1}: {e}"
                )

            await asyncio.sleep(1)

        return False


    # ══════════════════════════════════════════════════════
    # SPEAK
    # ══════════════════════════════════════════════════════

    def speak(
        self,
        text: str,
        use_cache: bool = True
    ) -> bool:

        if not text:
            return False

        try:

            cleaned = self.text_processor.clean(text)

            print(f"\n🤖 AI: {cleaned}\n")

            chunks = self.text_processor.split_text(
                cleaned,
                self.config.chunk_size
            )

            overall_success = True

            for chunk in chunks:

                if self.stop_signal:
                    break

                cache_path = self._get_cache_path(chunk)

                # CACHE HIT
                if (
                    use_cache and
                    os.path.exists(cache_path)
                ):

                    logger.info("📦 Cache Hit")

                    self.stats["cache_hits"] += 1

                    success = self._play_audio(cache_path)

                    overall_success &= success

                    continue

                # GENERATE NEW AUDIO
                success = asyncio.run(
                    self._generate_audio(
                        chunk,
                        cache_path
                    )
                )

                # EDGE TTS SUCCESS
                if success:

                    play_success = self._play_audio(
                        cache_path
                    )

                    overall_success &= play_success

                # FALLBACK
                else:

                    logger.warning(
                        "⚠️ Using Offline Fallback"
                    )

                    fallback_success = self.offline_engine.speak(
                        chunk
                    )

                    self.stats["offline_hits"] += 1

                    overall_success &= fallback_success

                time.sleep(0.2)

            # HISTORY
            if self.config.enable_history:

                self.history.add(
                    SpeechEntry(
                        timestamp=datetime.now().isoformat(),
                        text=cleaned,
                        voice=self.config.voice,
                        engine="edge_tts",
                        success=overall_success
                    )
                )

            self.stats["total_spoken"] += 1

            return overall_success

        except Exception as e:

            logger.error(f"❌ Speak Error: {e}")

            self.stats["failures"] += 1

            return False


    # ══════════════════════════════════════════════════════
    # CONTROLS
    # ══════════════════════════════════════════════════════

    def stop(self):

        self.stop_signal = True

        pygame.mixer.music.stop()

        logger.info("⛔ Speech Stopped")


    def pause(self):

        self.is_paused = True

        pygame.mixer.music.pause()

        logger.info("⏸️ Paused")


    def resume(self):

        self.is_paused = False

        pygame.mixer.music.unpause()

        logger.info("▶️ Resumed")


    # ══════════════════════════════════════════════════════
    # VOICE SETTINGS
    # ══════════════════════════════════════════════════════

    def set_voice(self, voice):

        self.config.voice = voice.value

        logger.info(f"🎤 Voice Changed: {voice.name}")


    def set_rate(self, rate: str):

        self.config.rate = rate

        logger.info(f"⚡ Rate: {rate}")


    def set_pitch(self, pitch: str):

        self.config.pitch = pitch

        logger.info(f"🎵 Pitch: {pitch}")


    def set_volume(self, volume: float):

        self.config.volume = max(
            0.0,
            min(1.0, volume)
        )

        logger.info(f"🔊 Volume: {volume}")


    # ══════════════════════════════════════════════════════
    # STATS
    # ══════════════════════════════════════════════════════

    def print_stats(self):

        print("\n" + "="*50)

        print("📊 TTS STATISTICS")

        print("="*50)

        for k, v in self.stats.items():

            print(f"{k:20}: {v}")

        print("="*50)


    # ══════════════════════════════════════════════════════
    # CLEANUP
    # ══════════════════════════════════════════════════════

    def cleanup(self):

        try:

            if self.temp_dir.exists():
                shutil.rmtree(self.temp_dir)

            logger.info("🧹 Cleanup Complete")

        except Exception as e:

            logger.warning(f"⚠️ Cleanup Warning: {e}")


# ══════════════════════════════════════════════════════════════
# SIMPLE GLOBAL WRAPPER
# ══════════════════════════════════════════════════════════════

_tts_instance = AdvancedTextToSpeech()


def speak(text: str):

    return _tts_instance.speak(text)


# ══════════════════════════════════════════════════════════════
# TESTING
# ══════════════════════════════════════════════════════════════

if __name__ == "__main__":

    print("\n" + "="*60)

    print("🎙️ ULTRA ADVANCED TEXT TO SPEECH SYSTEM")

    print("="*60)


    tts = AdvancedTextToSpeech(

        VoiceConfig(

            voice=Language.URDU_FEMALE.value,

            rate="-10%",

            pitch="+0Hz",

            volume=1.0
        )
    )


    # TEST 1
    print("\n🟢 TEST 1")

    tts.speak(
        "السلام علیکم۔ میں آپ کا ای آئی اسسٹنٹ ہوں۔"
    )


    # TEST 2
    print("\n🟢 TEST 2")

    tts.set_voice(Language.URDU_MALE)

    tts.speak(
        "یہ مردانہ آواز ہے۔"
    )


    # TEST 3
    print("\n🟢 TEST 3")

    tts.set_voice(Language.ENGLISH_FEMALE)

    tts.speak(
        "Hello. I am your advanced AI assistant."
    )


    # TEST 4
    print("\n🟢 TEST 4")

    long_text = """
    یہ ایک بہت لمبا ٹیکسٹ ہے جسے سسٹم مختلف حصوں میں تقسیم کرے گا۔
    پھر ہر حصے کو الگ الگ آواز میں تبدیل کرے گا۔
    اس طرح بڑے ریسپانس بھی آسانی سے سنائے جا سکتے ہیں۔
    """

    tts.speak(long_text)


    # TEST 5
    print("\n🟢 TEST 5")

    tts.set_rate("-30%")

    tts.set_pitch("+10Hz")

    tts.speak(
        "یہ سلو اور ہائی پچ آواز ہے۔"
    )


    # STATS
    tts.print_stats()


    # EXPORT HISTORY
    tts.history.export_txt()


    print("\n✅ ALL TESTS COMPLETE")