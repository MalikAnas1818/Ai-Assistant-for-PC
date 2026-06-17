"""
╔══════════════════════════════════════════════════════════════╗
║            INTENT CLASSIFIER                                ║
║            Language: English Only                           ║
╚══════════════════════════════════════════════════════════════╝
"""

import re
import logging
import json
import time
from typing import Dict, List, Optional, Callable, Any
from enum import Enum
from dataclasses import dataclass
from datetime import datetime
from difflib import SequenceMatcher


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
# INTENT TYPES
# ══════════════════════════════════════════════════════════════

class IntentType(Enum):
    # Web
    OPEN_YOUTUBE      = "open_youtube"
    YOUTUBE_SEARCH    = "youtube_search"
    OPEN_CHROME       = "open_chrome"
    GOOGLE_SEARCH     = "google_search"
    # File System
    CREATE_FOLDER     = "create_folder"
    DELETE_FILE       = "delete_file"
    OPEN_FILE         = "open_file"
    # System
    TAKE_SCREENSHOT   = "take_screenshot"
    SHUTDOWN_PC       = "shutdown_pc"
    RESTART_PC        = "restart_pc"
    SLEEP_PC          = "sleep_pc"
    # System Info
    GET_TIME          = "get_time"
    GET_DATE          = "get_date"
    GET_WEATHER       = "get_weather"
    # Apps
    OPEN_CALCULATOR   = "open_calculator"
    OPEN_NOTEPAD      = "open_notepad"
    # Multimedia
    PLAY_MUSIC        = "play_music"
    PLAY_VIDEO        = "play_video"
    PAUSE_MEDIA       = "pause_media"
    STOP_MEDIA        = "stop_media"
    # Communication
    SEND_WHATSAPP     = "send_whatsapp"
    SEND_EMAIL        = "send_email"
    # Fallback
    UNKNOWN           = "unknown"


# ══════════════════════════════════════════════════════════════
# DATA CLASSES
# ══════════════════════════════════════════════════════════════

@dataclass
class ClassificationResult:
    intent:           IntentType
    confidence:       float
    parameters:       Dict[str, Any]
    raw_command:      str
    cleaned_command:  str
    timestamp:        datetime
    matched_keywords: List[str]


@dataclass
class IntentPattern:
    keywords:            List[str]
    intent:              IntentType
    parameter_extractor: Optional[Callable] = None
    confidence_boost:    float = 0.0


# ══════════════════════════════════════════════════════════════
# INTENT CLASSIFIER
# ══════════════════════════════════════════════════════════════

class IntentClassifier:
    """
    English intent classifier with:
    - Keyword + fuzzy matching
    - Confidence scoring
    - Parameter extraction
    - Classification history
    - Callback support
    """

    def __init__(self, fuzzy_threshold: float = 0.7):
        self.patterns:   Dict[IntentType, IntentPattern] = {}
        self.history:    List[ClassificationResult] = []
        self.callbacks:  List[Callable] = []
        self.max_history     = 100
        self.fuzzy_threshold = fuzzy_threshold
        self._register_patterns()
        logger.info(f"Intent Classifier ready | {len(self.patterns)} intents")

    # ── Pattern Registration ──────────────────────────────────

    def _register_patterns(self):
        reg = self.register  # shorthand

        # Web
        reg(IntentType.OPEN_YOUTUBE,   ["open youtube", "youtube open", "start youtube"], boost=0.15)
        reg(IntentType.YOUTUBE_SEARCH, ["search on youtube", "youtube search", "find on youtube"],
            extractor=self._extract_query)
        reg(IntentType.OPEN_CHROME,    ["open chrome", "chrome open", "start chrome", "launch chrome"], boost=0.15)
        reg(IntentType.GOOGLE_SEARCH,  ["google search", "search google", "search for", "look up", "find"],
            extractor=self._extract_query)

        # File System
        reg(IntentType.CREATE_FOLDER,  ["create folder", "new folder", "make folder"],
            extractor=self._extract_folder_name)
        reg(IntentType.DELETE_FILE,    ["delete file", "remove file", "delete"],
            extractor=self._extract_file_name)
        reg(IntentType.OPEN_FILE,      ["open file", "launch file"],
            extractor=self._extract_file_name)

        # System
        reg(IntentType.TAKE_SCREENSHOT, ["take screenshot", "screenshot", "capture screen"], boost=0.2)
        reg(IntentType.SHUTDOWN_PC,     ["shutdown", "shut down", "power off", "turn off computer"], boost=0.25)
        reg(IntentType.RESTART_PC,      ["restart", "reboot", "restart computer"], boost=0.25)
        reg(IntentType.SLEEP_PC,        ["sleep", "sleep mode", "put to sleep"], boost=0.15)

        # System Info
        reg(IntentType.GET_TIME,    ["what time is it", "current time", "tell me the time", "what's the time"], boost=0.2)
        reg(IntentType.GET_DATE,    ["what's the date", "today's date", "current date", "what day is it"], boost=0.2)
        reg(IntentType.GET_WEATHER, ["weather", "what's the weather", "temperature", "forecast"],
            extractor=self._extract_location)

        # Apps
        reg(IntentType.OPEN_CALCULATOR, ["open calculator", "calculator", "launch calculator"], boost=0.15)
        reg(IntentType.OPEN_NOTEPAD,    ["open notepad", "notepad", "launch notepad"], boost=0.15)

        # Multimedia
        reg(IntentType.PLAY_MUSIC, ["play music", "play song", "play some music"],
            extractor=self._extract_query)
        reg(IntentType.PLAY_VIDEO, ["play video", "watch video"],
            extractor=self._extract_query)
        reg(IntentType.PAUSE_MEDIA, ["pause", "pause music", "pause video"], boost=0.15)
        reg(IntentType.STOP_MEDIA,  ["stop", "stop music", "stop video"], boost=0.15)

        # Communication
        reg(IntentType.SEND_WHATSAPP, ["send whatsapp", "whatsapp message", "message on whatsapp"],
            extractor=self._extract_message)
        reg(IntentType.SEND_EMAIL,    ["send email", "email", "send mail"],
            extractor=self._extract_message)

    def register(
        self,
        intent: IntentType,
        keywords: List[str],
        extractor: Optional[Callable] = None,
        boost: float = 0.0
    ):
        self.patterns[intent] = IntentPattern(
            keywords=keywords,
            intent=intent,
            parameter_extractor=extractor,
            confidence_boost=boost
        )

    # ── Text Cleaning ─────────────────────────────────────────

    def _clean(self, command: str) -> str:
        command = command.lower().strip()
        command = re.sub(r'[^\w\s]', ' ', command)
        return re.sub(r'\s+', ' ', command).strip()

    def _similarity(self, a: str, b: str) -> float:
        return SequenceMatcher(None, a, b).ratio()

    # ── Parameter Extractors ──────────────────────────────────

    def _extract_after(self, command: str, keywords: List[str], key: str, default: str = "") -> Dict:
        for kw in sorted(keywords, key=len, reverse=True):  # longest match first
            if kw in command:
                value = command.split(kw, 1)[-1].strip()
                if value:
                    return {key: value}
        return {key: default}

    def _extract_query(self, command: str, keywords: List[str]) -> Dict:
        return self._extract_after(command, keywords, "query")

    def _extract_folder_name(self, command: str, keywords: List[str]) -> Dict:
        return self._extract_after(command, keywords, "folder_name", "New Folder")

    def _extract_file_name(self, command: str, keywords: List[str]) -> Dict:
        return self._extract_after(command, keywords, "file_name")

    def _extract_location(self, command: str, keywords: List[str]) -> Dict:
        return self._extract_after(command, keywords, "location", "current")

    def _extract_message(self, command: str, keywords: List[str]) -> Dict:
        return self._extract_after(command, keywords, "message")

    # ── Classification ────────────────────────────────────────

    def classify(self, command: str) -> ClassificationResult:
        raw     = command
        cleaned = self._clean(command)

        best_intent     = IntentType.UNKNOWN
        best_confidence = 0.0
        matched_kws     = []
        params          = {}

        for intent, pattern in self.patterns.items():
            for kw in pattern.keywords:

                # Exact keyword match
                if kw in cleaned:
                    confidence = 0.8 + pattern.confidence_boost
                    matched_kws.append(kw)
                    if confidence > best_confidence:
                        best_confidence = confidence
                        best_intent     = intent
                        params = pattern.parameter_extractor(cleaned, pattern.keywords) \
                                 if pattern.parameter_extractor else {}

                # Fuzzy match
                else:
                    sim = self._similarity(kw, cleaned)
                    if sim > self.fuzzy_threshold:
                        confidence = sim * 0.7 + pattern.confidence_boost
                        matched_kws.append(f"{kw} (fuzzy:{sim:.2f})")
                        if confidence > best_confidence:
                            best_confidence = confidence
                            best_intent     = intent
                            params = pattern.parameter_extractor(cleaned, pattern.keywords) \
                                     if pattern.parameter_extractor else {}

        result = ClassificationResult(
            intent=best_intent,
            confidence=min(best_confidence, 1.0),
            parameters=params,
            raw_command=raw,
            cleaned_command=cleaned,
            timestamp=datetime.now(),
            matched_keywords=matched_kws
        )

        # Save to history
        self.history.append(result)
        if len(self.history) > self.max_history:
            self.history.pop(0)

        # Run callbacks
        for cb in self.callbacks:
            try:
                cb(result)
            except Exception as e:
                logger.warning(f"Callback error: {e}")

        logger.info(f"→ {result.intent.value} (confidence: {result.confidence:.2f})")
        return result

    # ── Callbacks ─────────────────────────────────────────────

    def add_callback(self, callback: Callable[[ClassificationResult], None]):
        self.callbacks.append(callback)

    def remove_callback(self, callback: Callable):
        self.callbacks.remove(callback)

    # ── History & Stats ───────────────────────────────────────

    def get_history(self, limit: int = 10) -> List[Dict]:
        return [
            {
                "intent":     r.intent.value,
                "confidence": f"{r.confidence:.2f}",
                "command":    r.raw_command,
                "parameters": r.parameters,
                "time":       r.timestamp.strftime("%H:%M:%S")
            }
            for r in self.history[-limit:]
        ]

    def get_stats(self) -> Dict[str, Any]:
        total = len(self.history)
        if not total:
            return {"total": 0}
        unknown = sum(1 for r in self.history if r.intent == IntentType.UNKNOWN)
        avg_conf = sum(r.confidence for r in self.history) / total
        return {
            "total":            total,
            "recognized":       total - unknown,
            "unknown":          unknown,
            "success_rate":     f"{(total - unknown) / total * 100:.1f}%",
            "avg_confidence":   f"{avg_conf:.2f}"
        }

    def print_stats(self):
        stats = self.get_stats()
        print(f"\n{'='*45}")
        print("📊 CLASSIFICATION STATS")
        print(f"{'='*45}")
        for k, v in stats.items():
            print(f"  {k:<20}: {v}")
        print(f"{'='*45}\n")

    def print_history(self, limit: int = 10):
        history = self.get_history(limit)
        print(f"\n{'='*45}")
        print(f"📜 HISTORY (last {limit})")
        print(f"{'='*45}")
        for i, item in enumerate(history, 1):
            print(f"\n  {i}. \"{item['command']}\"")
            print(f"     Intent: {item['intent']} | Confidence: {item['confidence']}")
            if item['parameters']:
                print(f"     Params: {item['parameters']}")
        print(f"{'='*45}\n")

    def export_history(self, filename: str = "classification_history.json"):
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(self.get_history(len(self.history)), f, indent=2)
        logger.info(f"History exported to {filename}")

    def clear_history(self):
        self.history.clear()

    def set_fuzzy_threshold(self, threshold: float):
        self.fuzzy_threshold = max(0.0, min(1.0, threshold))
        logger.info(f"Fuzzy threshold: {self.fuzzy_threshold}")


# ══════════════════════════════════════════════════════════════
# QUICK HELPER
# ══════════════════════════════════════════════════════════════

_classifier: Optional[IntentClassifier] = None

def classify(command: str) -> ClassificationResult:
    """One-liner usage: result = classify('open youtube')"""
    global _classifier
    if _classifier is None:
        _classifier = IntentClassifier()
    return _classifier.classify(command)


# ══════════════════════════════════════════════════════════════
# MAIN — TESTING
# ══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("\n" + "="*50)
    print("🧠 INTENT CLASSIFIER — ENGLISH")
    print("="*50 + "\n")

    clf = IntentClassifier()

    # Warn on low confidence
    def on_classify(result: ClassificationResult):
        if result.confidence < 0.6:
            print(f"  ⚠️  Low confidence: {result.confidence:.2f}")

    clf.add_callback(on_classify)

    test_commands = [
        "open youtube",
        "search on youtube python tutorial",
        "open chrome",
        "search google for machine learning",
        "create folder project files",
        "take screenshot",
        "shut down",
        "what time is it",
        "what's the weather in London",
        "play some music",
        "pause",
        "send whatsapp hey how are you",
        "unknown gibberish command xyz",
    ]

    print("Testing commands...\n")

    for cmd in test_commands:
        result = clf.classify(cmd)
        print(f"  Command    : {cmd}")
        print(f"  Intent     : {result.intent.value}")
        print(f"  Confidence : {result.confidence:.2f}")
        if result.parameters:
            print(f"  Params     : {result.parameters}")
        print()
        time.sleep(0.1)

    clf.print_stats()
    clf.print_history(limit=5)
    clf.export_history()

    print("✅ All tests complete!")