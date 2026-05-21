"""
╔══════════════════════════════════════════════════════════════╗
║         ADVANCED SPEECH RECOGNITION SYSTEM                  ║
║         Supports: Urdu, English | Whisper + Google           ║
║         Author: Upgraded from your original code             ║
╚══════════════════════════════════════════════════════════════╝

INSTALLATION:
    pip install SpeechRecognition pyaudio numpy
    pip install openai-whisper          # Free, offline Whisper
    pip install faster-whisper          # Optional: 3x faster version

WHISPER MODELS (all FREE, offline):
    tiny    → fastest,  least accurate (~39M params)
    base    → fast,     decent accuracy (~74M params)  ← recommended start
    small   → balanced  (~244M params)
    medium  → accurate  (~769M params)
    large   → best      (~1550M params) - needs good GPU/RAM
"""

import speech_recognition as sr
import logging
import time
import os
import json
from typing import Optional, List, Dict
from datetime import datetime
from pathlib import Path
import numpy as np

# ─────────────────────────────────────────────
#  Whisper Import (sahi tarika)
# ─────────────────────────────────────────────
try:
    import whisper  # openai-whisper library
    WHISPER_AVAILABLE = True
except ImportError:
    WHISPER_AVAILABLE = False

# ─────────────────────────────────────────────
#  Faster-Whisper (optional, 3x faster)
# ─────────────────────────────────────────────
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
#  LANGUAGE CONFIG
# ══════════════════════════════════════════════════════════════
LANGUAGE_CONFIG = {
    "urdu": {
        "google_code": "ur-PK",
        "whisper_lang": "ur",
        "display": "اردو / Urdu"
    },
    "english": {
        "google_code": "en-US",
        "whisper_lang": "en",
        "display": "English"
    },
    "hindi": {
        "google_code": "hi-IN",
        "whisper_lang": "hi",
        "display": "हिन्दी / Hindi"
    },
    "arabic": {
        "google_code": "ar-SA",
        "whisper_lang": "ar",
        "display": "عربي / Arabic"
    },
    "auto": {
        "google_code": "en-US",
        "whisper_lang": None,  # Auto-detect
        "display": "Auto Detect"
    }
}


# ══════════════════════════════════════════════════════════════
#  WHISPER ENGINE (sahi implementation)
# ══════════════════════════════════════════════════════════════
class WhisperEngine:
    """
    OpenAI Whisper - BILKUL FREE
    - Internet ki zaroorat NAHI (offline chalega)
    - Urdu support bahut acha hai
    - Large model best hai lekin slow
    """

    def __init__(self, model_size: str = "base", use_faster: bool = True):
        self.model = None
        self.model_size = model_size
        self.use_faster = use_faster and FASTER_WHISPER_AVAILABLE
        self._load_model()

    def _load_model(self):
        try:
            if self.use_faster and FASTER_WHISPER_AVAILABLE:
                logger.info(f"⚡ Loading Faster-Whisper '{self.model_size}'...")
                # compute_type="int8" = RAM efficient, CPU friendly
                self.model = FasterWhisperModel(
                    self.model_size,
                    device="auto",
                    compute_type="int8"
                )
                logger.info(f"✅ Faster-Whisper ready! (3x faster than normal Whisper)")

            elif WHISPER_AVAILABLE:
                logger.info(f"🧠 Loading OpenAI Whisper '{self.model_size}'...")
                self.model = whisper.load_model(self.model_size)
                logger.info(f"✅ Whisper '{self.model_size}' loaded!")

            else:
                logger.warning("⚠️ Koi Whisper library nahi mili. 'pip install openai-whisper' chalao")

        except Exception as e:
            logger.error(f"❌ Whisper load nahi hua: {e}")
            self.model = None

    def transcribe(self, audio_array: np.ndarray, language: Optional[str] = None) -> Optional[str]:
        """Audio numpy array se text nikalo"""
        if not self.model:
            return None

        try:
            if self.use_faster and FASTER_WHISPER_AVAILABLE:
                # Faster-Whisper ka tarika
                segments, info = self.model.transcribe(
                    audio_array,
                    language=language,
                    beam_size=5,
                    best_of=5,
                    temperature=0.0,  # Deterministic output
                    condition_on_previous_text=True,
                    vad_filter=True,  # Voice Activity Detection
                    vad_parameters=dict(min_silence_duration_ms=500)
                )
                text = " ".join([seg.text for seg in segments]).strip()
                if language is None:
                    logger.info(f"🌍 Detected language: {info.language} ({info.language_probability:.0%} confidence)")

            else:
                # Normal Whisper ka tarika
                result = self.model.transcribe(
                    audio_array,
                    language=language,
                    temperature=0.0,
                    best_of=5,
                    beam_size=5,
                    fp16=False  # CPU safe
                )
                text = result["text"].strip()
                if language is None:
                    logger.info(f"🌍 Detected: {result.get('language', 'unknown')}")

            logger.info(f"✅ Whisper result: {text}")
            return text if text else None

        except Exception as e:
            logger.error(f"❌ Whisper transcription error: {e}")
            return None


# ══════════════════════════════════════════════════════════════
#  TRANSCRIPTION HISTORY
# ══════════════════════════════════════════════════════════════
class TranscriptionHistory:
    """Sari transcriptions save karo"""

    def __init__(self, save_path: str = "transcriptions.json"):
        self.save_path = Path(save_path)
        self.history: List[Dict] = []
        self._load()

    def _load(self):
        if self.save_path.exists():
            try:
                with open(self.save_path, 'r', encoding='utf-8') as f:
                    self.history = json.load(f)
                logger.info(f"📂 {len(self.history)} purani transcriptions load hui")
            except Exception:
                self.history = []

    def add(self, text: str, language: str, engine: str, confidence: float = 0.0):
        entry = {
            "timestamp": datetime.now().isoformat(),
            "text": text,
            "language": language,
            "engine": engine,
            "confidence": confidence
        }
        self.history.append(entry)
        self._save()
        return entry

    def _save(self):
        try:
            with open(self.save_path, 'w', encoding='utf-8') as f:
                json.dump(self.history, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.error(f"❌ Save error: {e}")

    def get_last(self, n: int = 5) -> List[Dict]:
        return self.history[-n:]

    def export_txt(self, output_path: str = "transcriptions.txt"):
        with open(output_path, 'w', encoding='utf-8') as f:
            for entry in self.history:
                f.write(f"[{entry['timestamp']}] ({entry['language']}) {entry['text']}\n")
        logger.info(f"📄 Exported to {output_path}")


# ══════════════════════════════════════════════════════════════
#  MAIN ADVANCED SPEECH RECOGNIZER
# ══════════════════════════════════════════════════════════════
class AdvancedSpeechRecognizer:
    """
    Full Advanced Speech Recognizer
    - Whisper (offline, free) primary engine
    - Google fallback (online, free with limits)
    - Auto retry on failure
    - Transcription history
    - Continuous listening mode
    - Multi-language support
    """

    def __init__(
        self,
        language: str = "urdu",
        whisper_model: str = "base",
        save_history: bool = True,
        max_retries: int = 3,
        use_faster_whisper: bool = True
    ):
        self.language = language.lower()
        self.max_retries = max_retries
        self.lang_config = LANGUAGE_CONFIG.get(self.language, LANGUAGE_CONFIG["urdu"])

        # SpeechRecognition setup
        self.recognizer = sr.Recognizer()
        self._configure_recognizer()

        # Whisper Engine
        logger.info("🚀 Whisper engine load ho raha hai...")
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

        logger.info(f"✅ Advanced Speech Recognizer ready!")
        logger.info(f"   Language: {self.lang_config['display']}")
        logger.info(f"   Whisper: {'✅' if self.whisper.model else '❌'}")

    def _configure_recognizer(self):
        """Recognizer settings optimize karo"""
        # Noise handling
        self.recognizer.energy_threshold = 300
        self.recognizer.dynamic_energy_threshold = True
        self.recognizer.dynamic_energy_adjustment_damping = 0.15
        self.recognizer.dynamic_energy_ratio = 1.5

        # Timing
        self.recognizer.pause_threshold = 1.0       # 1 sec silence = end of speech
        self.recognizer.non_speaking_duration = 0.3
        self.recognizer.phrase_threshold = 0.3

    def _audio_to_numpy(self, audio: sr.AudioData) -> np.ndarray:
        """SpeechRecognition audio → Whisper ke liye numpy array"""
        wav_data = audio.get_wav_data(convert_rate=16000, convert_width=2)
        audio_array = np.frombuffer(wav_data, dtype=np.int16).astype(np.float32)
        audio_array = audio_array / 32768.0  # Normalize to [-1, 1]
        return audio_array

    def _try_whisper(self, audio: sr.AudioData) -> Optional[str]:
        """Whisper se recognize karo"""
        if not self.whisper.model:
            return None

        audio_array = self._audio_to_numpy(audio)
        lang = self.lang_config["whisper_lang"]  # None = auto-detect
        result = self.whisper.transcribe(audio_array, language=lang)

        if result:
            self.stats["whisper_hits"] += 1
        return result

    def _try_google(self, audio: sr.AudioData) -> Optional[str]:
        """Google Speech API se recognize karo (internet chahiye)"""
        try:
            logger.info("🌐 Google API try kar raha hoon...")
            google_lang = self.lang_config["google_code"]
            text = self.recognizer.recognize_google(audio, language=google_lang)
            logger.info(f"✅ Google: {text}")
            self.stats["google_hits"] += 1
            return text

        except sr.UnknownValueError:
            logger.warning("❌ Google: Audio samajh nahi aaya")
            return None
        except sr.RequestError as e:
            logger.error(f"❌ Google API error (internet check karo): {e}")
            return None

    def listen(
        self,
        timeout: int = 10,
        phrase_limit: int = 30,
        calibrate_duration: float = 1.0
    ) -> Optional[str]:
        """
        Ek baar suno aur text return karo
        
        Args:
            timeout: Kitne seconds wait karein speech shuru hone ka
            phrase_limit: Max kitne seconds ki speech record ho
            calibrate_duration: Noise calibration time
        
        Returns:
            Recognized text ya None
        """
        self.stats["total_attempts"] += 1

        for attempt in range(1, self.max_retries + 1):
            try:
                if attempt > 1:
                    logger.info(f"🔄 Retry #{attempt}/{self.max_retries}")
                    time.sleep(0.5)

                with sr.Microphone(sample_rate=16000) as source:
                    # Noise calibration
                    logger.info(f"🔊 Background noise calibrate ho raha hai ({calibrate_duration}s)...")
                    self.recognizer.adjust_for_ambient_noise(source, duration=calibrate_duration)
                    current_threshold = self.recognizer.energy_threshold
                    logger.info(f"   Energy threshold: {current_threshold:.0f}")

                    # Listen
                    print("\n🎤 Bol sakte ho... (baat khatam karo to chup raho)")
                    logger.info("🎧 Listening...")

                    audio = self.recognizer.listen(
                        source,
                        timeout=timeout,
                        phrase_time_limit=phrase_limit
                    )
                    logger.info("✅ Audio capture complete!")

                # Try engines in order
                result = None

                # 1st: Whisper (offline, best quality)
                if self.whisper.model:
                    result = self._try_whisper(audio)

                # 2nd: Google fallback
                if not result:
                    result = self._try_google(audio)

                # Result process karo
                if result:
                    result = result.strip()
                    self.stats["successful"] += 1

                    # History save
                    if self.history:
                        engine = "whisper" if self.stats["whisper_hits"] > 0 else "google"
                        self.history.add(result, self.language, engine)

                    return result

                logger.warning(f"⚠️ Attempt {attempt}: Koi result nahi aaya")

            except sr.WaitTimeoutError:
                logger.warning(f"⏱️ Timeout - koi awaaz nahi aayi (attempt {attempt})")
            except OSError as e:
                logger.error(f"❌ Microphone error: {e}")
                logger.error("   💡 Check karo: microphone connected hai? Permissions di hain?")
                return None
            except Exception as e:
                logger.error(f"❌ Unexpected error: {e}")

        self.stats["failed"] += 1
        logger.error(f"❌ {self.max_retries} attempts ke baad bhi fail")
        return None

    def listen_continuous(
        self,
        callback,
        stop_phrase: str = "band karo",
        max_duration_minutes: int = 60
    ):
        """
        Continuously suno jab tak stop phrase na bolo
        
        Args:
            callback: Function jo har result pe call ho (text param lega)
            stop_phrase: Yeh bolo to stop ho jaye
            max_duration_minutes: Maximum session time
        """
        logger.info(f"🔁 Continuous mode shuru - '{stop_phrase}' bolo band karne ke liye")
        print(f"\n{'='*50}")
        print(f"🎙️  CONTINUOUS LISTENING MODE")
        print(f"   '{stop_phrase}' bolo band karne ke liye")
        print(f"{'='*50}\n")

        start_time = time.time()
        session_results = []
        count = 0

        while True:
            # Time check
            elapsed = (time.time() - start_time) / 60
            if elapsed >= max_duration_minutes:
                logger.info(f"⏰ {max_duration_minutes} minute limit complete")
                break

            count += 1
            print(f"\n[{count}] 🎤 Sun raha hoon...")

            result = self.listen()

            if result:
                print(f"[{count}] 📝 Text: {result}")
                session_results.append(result)

                # Stop check
                if stop_phrase.lower() in result.lower():
                    print(f"\n✅ '{stop_phrase}' detect hua - band ho raha hai!")
                    break

                # Callback call karo
                try:
                    callback(result)
                except Exception as e:
                    logger.error(f"❌ Callback error: {e}")

            else:
                print(f"[{count}] ❓ Kuch samajh nahi aaya, dobara try karo...")

        # Session summary
        print(f"\n{'='*50}")
        print(f"📊 SESSION COMPLETE")
        print(f"   Total captured: {len(session_results)} phrases")
        print(f"   Duration: {elapsed:.1f} minutes")
        print(f"{'='*50}")

        return session_results

    def print_stats(self):
        """Recognition statistics print karo"""
        print(f"\n{'='*40}")
        print(f"📊 RECOGNITION STATS")
        print(f"{'='*40}")
        print(f"   Total attempts : {self.stats['total_attempts']}")
        print(f"   Successful     : {self.stats['successful']}")
        print(f"   Whisper hits   : {self.stats['whisper_hits']}")
        print(f"   Google hits    : {self.stats['google_hits']}")
        print(f"   Failed         : {self.stats['failed']}")
        if self.stats['total_attempts'] > 0:
            rate = self.stats['successful'] / self.stats['total_attempts'] * 100
            print(f"   Success rate   : {rate:.1f}%")
        print(f"{'='*40}\n")


# ══════════════════════════════════════════════════════════════
#  SIMPLE HELPER FUNCTIONS
# ══════════════════════════════════════════════════════════════

# Global instance (reuse karo, baar baar load mat karo)
_recognizer_cache: Dict[str, AdvancedSpeechRecognizer] = {}


def listen_command(
    language: str = "urdu",
    whisper_model: str = "base"
) -> Optional[str]:
    """
    Simple one-liner function - ek baar suno aur text lo
    
    Usage:
        text = listen_command("urdu")
        text = listen_command("english")
        text = listen_command("auto")  # Language auto detect
    """
    cache_key = f"{language}_{whisper_model}"

    if cache_key not in _recognizer_cache:
        _recognizer_cache[cache_key] = AdvancedSpeechRecognizer(
            language=language,
            whisper_model=whisper_model
        )

    return _recognizer_cache[cache_key].listen()


def start_continuous(language: str = "urdu", callback=None):
    """
    Continuous listening shuru karo
    
    Usage:
        def my_handler(text):
            print(f"Mila: {text}")
        
        start_continuous("urdu", my_handler)
    """
    if callback is None:
        callback = lambda text: print(f"  → {text}")

    recognizer = AdvancedSpeechRecognizer(language=language)
    return recognizer.listen_continuous(callback)


# ══════════════════════════════════════════════════════════════
#  MAIN - TEST KARO
# ══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("\n" + "="*60)
    print("🎙️  ADVANCED SPEECH RECOGNITION SYSTEM")
    print("="*60)
    print(f"   Whisper available    : {'✅ YES' if WHISPER_AVAILABLE else '❌ NO - pip install openai-whisper'}")
    print(f"   Faster-Whisper avail : {'✅ YES (3x fast!)' if FASTER_WHISPER_AVAILABLE else '⚠️  NO - pip install faster-whisper'}")
    print("="*60 + "\n")

    import sys

    # ─── Single Listen Mode ───────────────────────────────
    if len(sys.argv) == 1 or sys.argv[1] == "single":
        print("📌 Mode: Single Listen\n")

        # Urdu test
        print("🔵 Test 1: Urdu mein bolo")
        urdu_result = listen_command("urdu", whisper_model="base")
        if urdu_result:
            print(f"   ✅ Urdu result: {urdu_result}\n")
        else:
            print("   ❌ Kuch capture nahi hua\n")

        # English test
        print("🟢 Test 2: English mein bolo")
        eng_result = listen_command("english", whisper_model="base")
        if eng_result:
            print(f"   ✅ English result: {eng_result}\n")
        else:
            print("   ❌ Kuch capture nahi hua\n")

        # Auto detect test
        print("🟡 Test 3: Koi bhi zubaan mein bolo (auto detect)")
        auto_result = listen_command("auto", whisper_model="base")
        if auto_result:
            print(f"   ✅ Auto result: {auto_result}\n")

    # ─── Continuous Mode ──────────────────────────────────
    elif sys.argv[1] == "continuous":
        lang = sys.argv[2] if len(sys.argv) > 2 else "urdu"
        print(f"📌 Mode: Continuous Listening ({lang})\n")

        results = []

        def handle_text(text):
            results.append(text)
            print(f"  💬 Captured: {text}")

        start_continuous(lang, handle_text)

        print("\n📄 Saara captured text:")
        for i, r in enumerate(results, 1):
            print(f"  {i}. {r}")

    # ─── Advanced Mode ────────────────────────────────────
    elif sys.argv[1] == "advanced":
        print("📌 Mode: Advanced with Stats\n")

        recognizer = AdvancedSpeechRecognizer(
            language="urdu",
            whisper_model="small",  # Better accuracy
            save_history=True,
            max_retries=3,
            use_faster_whisper=True
        )

        print("🎤 Kuch bolo (3 attempts available)...")
        result = recognizer.listen(
            timeout=15,
            phrase_limit=30,
            calibrate_duration=2.0
        )

        if result:
            print(f"\n✅ Final result: {result}")
        else:
            print("\n❌ Recognition fail hua")

        recognizer.print_stats()

        # History export
        if recognizer.history:
            recognizer.history.export_txt("my_transcriptions.txt")
            print("📄 Transcriptions.txt mein save ho gayi!")

    else:
        print("Usage:")
        print("  python speech.py              → Single listen test")
        print("  python speech.py single       → Single listen test")
        print("  python speech.py continuous   → Continuous mode (Urdu)")
        print("  python speech.py continuous english  → Continuous (English)")
        print("  python speech.py advanced     → Full features + stats")