"""
╔══════════════════════════════════════════════════════════════╗
║         SPEECH RECOGNITION SYSTEM                           ║
║         Language: English Only                              ║
║         Engine: Whisper (offline) + Google (fallback)       ║
╚══════════════════════════════════════════════════════════════╝

INSTALLATION:
    pip install SpeechRecognition pyaudio numpy
    pip install openai-whisper          # Free, offline
    pip install faster-whisper          # Optional: 3x faster
"""

import speech_recognition as sr
import logging
import time
import json
from typing import Optional, List, Dict
from datetime import datetime
from pathlib import Path
import numpy as np

# ─────────────────────────────────────────────
#  Whisper Import
# ─────────────────────────────────────────────
try:
    import whisper
    WHISPER_AVAILABLE = True
except ImportError:
    WHISPER_AVAILABLE = False

try:
    from faster_whisper import WhisperModel as FasterWhisperModel
    FASTER_WHISPER_AVAILABLE = True
except ImportError:
    FASTER_WHISPER_AVAILABLE = False

# ─────────────────────────────────────────────
#  Logging Setup
# ─────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-8s | %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════
#  WHISPER ENGINE
# ══════════════════════════════════════════════════════════════
class WhisperEngine:
    """
    OpenAI Whisper - FREE, offline
    Models: tiny, base, small, medium, large
    """

    def __init__(self, model_size: str = "base", use_faster: bool = True):
        self.model = None
        self.model_size = model_size
        self.use_faster = use_faster and FASTER_WHISPER_AVAILABLE
        self._load_model()

    def _load_model(self):
        try:
            if self.use_faster and FASTER_WHISPER_AVAILABLE:
                logger.info(f"Loading Faster-Whisper '{self.model_size}'...")
                self.model = FasterWhisperModel(
                    self.model_size,
                    device="auto",
                    compute_type="int8"
                )
                logger.info("Faster-Whisper ready! (3x faster than normal Whisper)")

            elif WHISPER_AVAILABLE:
                logger.info(f"Loading OpenAI Whisper '{self.model_size}'...")
                self.model = whisper.load_model(self.model_size)
                logger.info(f"Whisper '{self.model_size}' loaded!")

            else:
                logger.warning("No Whisper library found. Run: pip install openai-whisper")

        except Exception as e:
            logger.error(f"Whisper failed to load: {e}")
            self.model = None

    def transcribe(self, audio_array: np.ndarray) -> Optional[str]:
        """Transcribe audio numpy array to English text"""
        if not self.model:
            return None

        try:
            if self.use_faster and FASTER_WHISPER_AVAILABLE:
                segments, _ = self.model.transcribe(
                    audio_array,
                    language="en",
                    beam_size=5,
                    best_of=5,
                    temperature=0.0,
                    condition_on_previous_text=True,
                    vad_filter=True,
                    vad_parameters=dict(min_silence_duration_ms=1000)  # was 500 — cut too early
                )
                text = " ".join([seg.text for seg in segments]).strip()

            else:
                result = self.model.transcribe(
                    audio_array,
                    language="en",
                    temperature=0.0,
                    best_of=5,
                    beam_size=5,
                    fp16=False
                )
                text = result["text"].strip()

            logger.info(f"Whisper result: {text}")
            return text if text else None

        except Exception as e:
            logger.error(f"Whisper transcription error: {e}")
            return None


# ══════════════════════════════════════════════════════════════
#  TRANSCRIPTION HISTORY
# ══════════════════════════════════════════════════════════════
class TranscriptionHistory:
    """Save all transcriptions to a JSON file"""

    def __init__(self, save_path: str = "transcriptions.json"):
        self.save_path = Path(save_path)
        self.history: List[Dict] = []
        self._load()

    def _load(self):
        if self.save_path.exists():
            try:
                with open(self.save_path, 'r', encoding='utf-8') as f:
                    self.history = json.load(f)
                logger.info(f"Loaded {len(self.history)} previous transcriptions")
            except Exception:
                self.history = []

    def add(self, text: str, engine: str):
        entry = {
            "timestamp": datetime.now().isoformat(),
            "text": text,
            "engine": engine
        }
        self.history.append(entry)
        self._save()
        return entry

    def _save(self):
        try:
            with open(self.save_path, 'w', encoding='utf-8') as f:
                json.dump(self.history, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.error(f"Save error: {e}")

    def get_last(self, n: int = 5) -> List[Dict]:
        return self.history[-n:]

    def export_txt(self, output_path: str = "transcriptions.txt"):
        with open(output_path, 'w', encoding='utf-8') as f:
            for entry in self.history:
                f.write(f"[{entry['timestamp']}] [{entry['engine']}] {entry['text']}\n")
        logger.info(f"Exported to {output_path}")


# ══════════════════════════════════════════════════════════════
#  MAIN SPEECH RECOGNIZER
# ══════════════════════════════════════════════════════════════
class SpeechRecognizer:
    """
    English Speech Recognizer
    - Whisper (offline, free) — primary engine
    - Google (online, free with limits) — fallback
    - Auto retry on failure
    - Transcription history
    - Continuous listening mode
    """

    def __init__(
        self,
        whisper_model: str = "base",
        save_history: bool = True,
        max_retries: int = 3,
        use_faster_whisper: bool = True
    ):
        self.max_retries = max_retries

        # SpeechRecognition setup
        self.recognizer = sr.Recognizer()
        self._configure_recognizer()

        # Whisper Engine
        logger.info("Loading Whisper engine...")
        self.whisper = WhisperEngine(
            model_size=whisper_model,
            use_faster=use_faster_whisper
        )

        # History
        self.history = TranscriptionHistory() if save_history else None

        # Stats
        self.stats = {
            "total_attempts": 0,
            "successful": 0,
            "whisper_hits": 0,
            "google_hits": 0,
            "failed": 0
        }

        logger.info("Speech Recognizer ready! | Language: English")
        logger.info(f"Whisper: {'OK' if self.whisper.model else 'NOT LOADED'}")

    def _configure_recognizer(self):
        self.recognizer.energy_threshold = 300
        self.recognizer.dynamic_energy_threshold = True
        self.recognizer.dynamic_energy_adjustment_damping = 0.15
        self.recognizer.dynamic_energy_ratio = 1.5
        self.recognizer.pause_threshold = 2.5       # 2.5s silence = speech ended (was 1.0 — too fast)
        self.recognizer.non_speaking_duration = 1.5 # Wait longer before cutting off (was 0.3)
        self.recognizer.phrase_threshold = 0.1      # More sensitive phrase start detection

    def _audio_to_numpy(self, audio: sr.AudioData) -> np.ndarray:
        wav_data = audio.get_wav_data(convert_rate=16000, convert_width=2)
        audio_array = np.frombuffer(wav_data, dtype=np.int16).astype(np.float32)
        return audio_array / 32768.0

    def _try_whisper(self, audio: sr.AudioData) -> Optional[str]:
        if not self.whisper.model:
            return None
        result = self.whisper.transcribe(self._audio_to_numpy(audio))
        if result:
            self.stats["whisper_hits"] += 1
        return result

    def _try_google(self, audio: sr.AudioData) -> Optional[str]:
        try:
            logger.info("Trying Google Speech API...")
            text = self.recognizer.recognize_google(audio, language="en-US")
            logger.info(f"Google result: {text}")
            self.stats["google_hits"] += 1
            return text
        except sr.UnknownValueError:
            logger.warning("Google: Could not understand audio")
            return None
        except sr.RequestError as e:
            logger.error(f"Google API error (check internet): {e}")
            return None

    def listen(
        self,
        timeout: int = 10,
        phrase_limit: int = 60,         # 60s max (was 30 — too short)
        calibrate_duration: float = 1.0
    ) -> Optional[str]:
        """
        Listen once and return transcribed text.

        Args:
            timeout: Seconds to wait for speech to begin
            phrase_limit: Max seconds of speech to record (default 60s)
            calibrate_duration: Noise calibration time in seconds

        Returns:
            Transcribed text or None
        """
        self.stats["total_attempts"] += 1

        for attempt in range(1, self.max_retries + 1):
            try:
                if attempt > 1:
                    logger.info(f"Retry {attempt}/{self.max_retries}")
                    time.sleep(0.5)

                with sr.Microphone(sample_rate=16000) as source:
                    logger.info(f"Calibrating ambient noise ({calibrate_duration}s)...")
                    self.recognizer.adjust_for_ambient_noise(source, duration=calibrate_duration)
                    logger.info(f"Energy threshold: {self.recognizer.energy_threshold:.0f}")

                    print("\n🎤 Speak now... (stop speaking to end)")
                    audio = self.recognizer.listen(
                        source,
                        timeout=timeout,
                        phrase_time_limit=phrase_limit
                    )
                    logger.info("Audio captured!")

                # Try Whisper first, then Google as fallback
                result = self._try_whisper(audio) or self._try_google(audio)

                if result:
                    result = result.strip()
                    self.stats["successful"] += 1
                    if self.history:
                        engine = "whisper" if self.stats["whisper_hits"] > 0 else "google"
                        self.history.add(result, engine)
                    return result

                logger.warning(f"Attempt {attempt}: No result")

            except sr.WaitTimeoutError:
                logger.warning(f"Timeout — no speech detected (attempt {attempt})")
            except OSError as e:
                logger.error(f"Microphone error: {e}")
                logger.error("Check: Is microphone connected? Are permissions granted?")
                return None
            except Exception as e:
                logger.error(f"Unexpected error: {e}")

        self.stats["failed"] += 1
        logger.error(f"Failed after {self.max_retries} attempts")
        return None

    def listen_continuous(self, callback, stop_phrase: str = "stop listening", max_duration_minutes: int = 60):
        """
        Listen continuously until stop phrase is spoken.

        Args:
            callback: Function called with each result string
            stop_phrase: Say this to stop the session
            max_duration_minutes: Max session length
        """
        print(f"\n{'='*50}")
        print(f"🎙️  CONTINUOUS LISTENING MODE (English)")
        print(f"   Say '{stop_phrase}' to stop")
        print(f"{'='*50}\n")

        start_time = time.time()
        session_results = []
        count = 0

        while True:
            elapsed = (time.time() - start_time) / 60
            if elapsed >= max_duration_minutes:
                logger.info(f"{max_duration_minutes}-minute limit reached")
                break

            count += 1
            print(f"\n[{count}] 🎤 Listening...")
            result = self.listen()

            if result:
                print(f"[{count}] 📝 {result}")
                session_results.append(result)

                if stop_phrase.lower() in result.lower():
                    print(f"\n✅ Stop phrase detected — ending session")
                    break

                try:
                    callback(result)
                except Exception as e:
                    logger.error(f"Callback error: {e}")
            else:
                print(f"[{count}] Could not understand, try again...")

        print(f"\n{'='*50}")
        print(f"SESSION COMPLETE")
        print(f"  Captured : {len(session_results)} phrases")
        print(f"  Duration : {elapsed:.1f} minutes")
        print(f"{'='*50}")
        return session_results

    def print_stats(self):
        print(f"\n{'='*40}")
        print(f"📊 RECOGNITION STATS")
        print(f"{'='*40}")
        print(f"  Total attempts : {self.stats['total_attempts']}")
        print(f"  Successful     : {self.stats['successful']}")
        print(f"  Whisper hits   : {self.stats['whisper_hits']}")
        print(f"  Google hits    : {self.stats['google_hits']}")
        print(f"  Failed         : {self.stats['failed']}")
        if self.stats['total_attempts'] > 0:
            rate = self.stats['successful'] / self.stats['total_attempts'] * 100
            print(f"  Success rate   : {rate:.1f}%")
        print(f"{'='*40}\n")


# ══════════════════════════════════════════════════════════════
#  SIMPLE HELPER
# ══════════════════════════════════════════════════════════════

_recognizer_cache: Optional[SpeechRecognizer] = None


def listen_command(whisper_model: str = "base") -> Optional[str]:
    """
    Simple one-liner: listen once and return English text.

    Usage:
        text = listen_command()
        text = listen_command(whisper_model="small")  # Better accuracy
    """
    global _recognizer_cache
    if _recognizer_cache is None:
        _recognizer_cache = SpeechRecognizer(whisper_model=whisper_model)
    return _recognizer_cache.listen()


def start_continuous(callback=None):
    """
    Start continuous English listening.

    Usage:
        def my_handler(text):
            print(f"Got: {text}")

        start_continuous(my_handler)
    """
    if callback is None:
        callback = lambda text: print(f"  → {text}")
    recognizer = SpeechRecognizer()
    return recognizer.listen_continuous(callback)


# ══════════════════════════════════════════════════════════════
#  MAIN — TESTING
# ══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import sys

    print("\n" + "="*60)
    print("🎙️  ENGLISH SPEECH RECOGNITION SYSTEM")
    print("="*60)
    print(f"  Whisper        : {'✅ Available' if WHISPER_AVAILABLE else '❌ Run: pip install openai-whisper'}")
    print(f"  Faster-Whisper : {'✅ Available (3x faster)' if FASTER_WHISPER_AVAILABLE else '⚠️  Optional: pip install faster-whisper'}")
    print("="*60 + "\n")

    mode = sys.argv[1] if len(sys.argv) > 1 else "single"

    # ── Single Listen ─────────────────────────────────────────
    if mode == "single":
        print("MODE: Single Listen\n")
        print("🟢 Speak in English...")
        result = listen_command(whisper_model="base")
        if result:
            print(f"\n✅ Result: {result}")
        else:
            print("\n❌ Nothing captured")

    # ── Continuous Mode ───────────────────────────────────────
    elif mode == "continuous":
        print("MODE: Continuous Listening\n")

        captured = []

        def handle(text):
            captured.append(text)

        start_continuous(handle)

        print("\n📄 Full session transcript:")
        for i, r in enumerate(captured, 1):
            print(f"  {i}. {r}")

    # ── Advanced Mode (with stats + history export) ───────────
    elif mode == "advanced":
        print("MODE: Advanced\n")

        recognizer = SpeechRecognizer(
            whisper_model="small",   # Better accuracy than base
            save_history=True,
            max_retries=3,
            use_faster_whisper=True
        )

        print("🎤 Speak now (3 attempts available)...")
        result = recognizer.listen(
            timeout=15,
            phrase_limit=30,
            calibrate_duration=2.0
        )

        if result:
            print(f"\n✅ Final result: {result}")
        else:
            print("\n❌ Recognition failed")

        recognizer.print_stats()

        if recognizer.history:
            recognizer.history.export_txt("transcriptions.txt")
            print("📄 Saved to transcriptions.txt")

    else:
        print("Usage:")
        print("  python speech.py              → Single listen test")
        print("  python speech.py single       → Single listen test")
        print("  python speech.py continuous   → Continuous mode")
        print("  python speech.py advanced     → Full features + stats")