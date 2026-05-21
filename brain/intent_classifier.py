import re
import logging
import json
from typing import Dict, List, Optional, Callable, Any
from enum import Enum
from dataclasses import dataclass, asdict
from datetime import datetime
from difflib import SequenceMatcher
import threading


# =========================
# LOGGING SETUP
# =========================

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# =========================
# ENUMS & DATA CLASSES
# =========================

class IntentType(Enum):
    """All available intents"""
    # Web Intents
    OPEN_YOUTUBE = "open_youtube"
    YOUTUBE_SEARCH = "youtube_search"
    OPEN_CHROME = "open_chrome"
    GOOGLE_SEARCH = "google_search"

    # File System Intents
    CREATE_FOLDER = "create_folder"
    DELETE_FILE = "delete_file"
    OPEN_FILE = "open_file"

    # System Intents
    TAKE_SCREENSHOT = "take_screenshot"
    SHUTDOWN_PC = "shutdown_pc"
    RESTART_PC = "restart_pc"
    SLEEP_PC = "sleep_pc"

    # System Info Intents
    GET_TIME = "get_time"
    GET_DATE = "get_date"
    GET_WEATHER = "get_weather"

    # Application Intents
    OPEN_CALCULATOR = "open_calculator"
    OPEN_NOTEPAD = "open_notepad"

    # Multimedia Intents
    PLAY_MUSIC = "play_music"
    PLAY_VIDEO = "play_video"
    PAUSE_MUSIC = "pause_music"
    STOP_MUSIC = "stop_music"

    # Communication Intents
    SEND_WHATSAPP_MESSAGE = "send_whatsapp_message"
    SEND_EMAIL = "send_email"

    # General
    UNKNOWN = "unknown"


@dataclass
class ClassificationResult:
    """Result from intent classification"""
    intent: IntentType
    confidence: float  # 0.0 to 1.0
    parameters: Dict[str, Any]
    raw_command: str
    cleaned_command: str
    timestamp: datetime
    matched_keywords: List[str]
    execution_required: bool = True


@dataclass
class IntentPattern:
    """Pattern for intent matching"""
    keywords: List[str]
    intent: IntentType
    parameter_extractor: Optional[Callable] = None
    confidence_boost: float = 0.0


# =========================
# ADVANCED INTENT CLASSIFIER
# =========================

class AdvancedIntentClassifier:
    """
    Advanced intent classifier with:
    - Multiple keyword patterns
    - Fuzzy matching
    - Confidence scoring
    - Parameter extraction
    - Intent history
    - Extensible design
    """

    def __init__(self):
        """Initialize the classifier"""
        self.intent_patterns: Dict[IntentType, IntentPattern] = {}
        self.classification_history: List[ClassificationResult] = []
        self.max_history = 100
        self.callbacks: List[Callable] = []
        self.fuzzy_threshold = 0.7

        # Register all patterns
        self._register_patterns()

        logger.info("✅ Advanced Intent Classifier initialized")

    # =========================
    # PATTERN REGISTRATION
    # =========================

    def _register_patterns(self) -> None:
        """Register all intent patterns"""

        # YOUTUBE INTENTS
        self.register_intent(
            IntentType.OPEN_YOUTUBE,
            keywords=[
                "youtube open",
                "open youtube",
                "youtube kholo",
                "youtube chalao",
                "youtube start"
            ],
            confidence_boost=0.15
        )

        self.register_intent(
            IntentType.YOUTUBE_SEARCH,
            keywords=[
                "youtube per search karo",
                "youtube pe search karo",
                "youtube search",
                "youtube pe",
                "youtube per",
                "youtube par dhundho"
            ],
            parameter_extractor=self._extract_search_query
        )

        # CHROME INTENTS
        self.register_intent(
            IntentType.OPEN_CHROME,
            keywords=[
                "chrome kholo",
                "open chrome",
                "chrome open",
                "chrome start",
                "chrome chalao"
            ],
            confidence_boost=0.15
        )

        # GOOGLE SEARCH INTENTS
        self.register_intent(
            IntentType.GOOGLE_SEARCH,
            keywords=[
                "google per search karo",
                "google pe search karo",
                "google search",
                "search karo",
                "google par dhundho",
                "search",
                "dhundho"
            ],
            parameter_extractor=self._extract_search_query
        )

        # FILE SYSTEM INTENTS
        self.register_intent(
            IntentType.CREATE_FOLDER,
            keywords=[
                "folder banao",
                "new folder",
                "folder create",
                "naiya folder",
                "folder bana"
            ],
            parameter_extractor=self._extract_folder_name
        )

        self.register_intent(
            IntentType.DELETE_FILE,
            keywords=[
                "delete karo",
                "file delete",
                "remove file",
                "file hatao"
            ],
            parameter_extractor=self._extract_file_name
        )

        self.register_intent(
            IntentType.OPEN_FILE,
            keywords=[
                "file kholo",
                "open file",
                "file open",
                "file start"
            ],
            parameter_extractor=self._extract_file_name
        )

        # SYSTEM INTENTS
        self.register_intent(
            IntentType.TAKE_SCREENSHOT,
            keywords=[
                "screenshot lo",
                "screen shot lo",
                "take screenshot",
                "screenshot lelo",
                "screenshot capture"
            ],
            confidence_boost=0.2
        )

        self.register_intent(
            IntentType.SHUTDOWN_PC,
            keywords=[
                "shutdown",
                "pc band karo",
                "computer shutdown",
                "shutdown karo",
                "band karo"
            ],
            confidence_boost=0.25
        )

        self.register_intent(
            IntentType.RESTART_PC,
            keywords=[
                "restart",
                "pc restart",
                "computer restart",
                "restart karo",
                "dobara start karo"
            ],
            confidence_boost=0.25
        )

        self.register_intent(
            IntentType.SLEEP_PC,
            keywords=[
                "sleep mode",
                "sleep karo",
                "pc sleep",
                "computer sleep"
            ],
            confidence_boost=0.15
        )

        # SYSTEM INFO INTENTS
        self.register_intent(
            IntentType.GET_TIME,
            keywords=[
                "time kya hua hai",
                "current time",
                "time batao",
                "kitna time hua",
                "time kya hai"
            ],
            confidence_boost=0.2
        )

        self.register_intent(
            IntentType.GET_DATE,
            keywords=[
                "date kya hai",
                "today date",
                "date batao",
                "aaj ki tarikh kya hai",
                "current date"
            ],
            confidence_boost=0.2
        )

        self.register_intent(
            IntentType.GET_WEATHER,
            keywords=[
                "weather kya hai",
                "mausam kaisa hai",
                "temperature",
                "bahar kaisa hai"
            ],
            parameter_extractor=self._extract_location
        )

        # APPLICATION INTENTS
        self.register_intent(
            IntentType.OPEN_CALCULATOR,
            keywords=[
                "calculator kholo",
                "open calculator",
                "calculator open",
                "calculator start"
            ],
            confidence_boost=0.15
        )

        self.register_intent(
            IntentType.OPEN_NOTEPAD,
            keywords=[
                "notepad kholo",
                "open notepad",
                "notepad open",
                "notepad start"
            ],
            confidence_boost=0.15
        )

        # MULTIMEDIA INTENTS
        self.register_intent(
            IntentType.PLAY_MUSIC,
            keywords=[
                "music chalao",
                "play music",
                "music play",
                "gana chalao",
                "gana play"
            ],
            parameter_extractor=self._extract_query
        )

        self.register_intent(
            IntentType.PLAY_VIDEO,
            keywords=[
                "video chalao",
                "play video",
                "video play"
            ],
            parameter_extractor=self._extract_query
        )

        self.register_intent(
            IntentType.PAUSE_MUSIC,
            keywords=[
                "pause",
                "music pause",
                "roko",
                "music roko"
            ],
            confidence_boost=0.15
        )

        self.register_intent(
            IntentType.STOP_MUSIC,
            keywords=[
                "stop",
                "music stop",
                "band karo",
                "music band"
            ],
            confidence_boost=0.15
        )

        # COMMUNICATION INTENTS
        self.register_intent(
            IntentType.SEND_WHATSAPP_MESSAGE,
            keywords=[
                "whatsapp message",
                "whatsapp bhejo",
                "message bhejo",
                "whatsapp send"
            ],
            parameter_extractor=self._extract_message
        )

        self.register_intent(
            IntentType.SEND_EMAIL,
            keywords=[
                "email bhejo",
                "send email",
                "email send",
                "mail bhejo"
            ],
            parameter_extractor=self._extract_message
        )

        logger.info(f"✅ Registered {len(self.intent_patterns)} intent patterns")

    def register_intent(
        self,
        intent: IntentType,
        keywords: List[str],
        parameter_extractor: Optional[Callable] = None,
        confidence_boost: float = 0.0
    ) -> None:
        """
        Register a custom intent pattern
        """
        pattern = IntentPattern(
            keywords=keywords,
            intent=intent,
            parameter_extractor=parameter_extractor,
            confidence_boost=confidence_boost
        )
        self.intent_patterns[intent] = pattern
        logger.info(f"✅ Registered intent: {intent.value}")

    # =========================
    # TEXT PROCESSING
    # =========================

    def clean_command(self, command: str) -> str:
        """Clean and normalize command"""
        command = command.lower().strip()
        command = re.sub(r'\s+', ' ', command)
        command = re.sub(r'[^\w\s۔؟،ؤئے]', ' ', command)  # Keep Urdu chars
        command = re.sub(r'\s+', ' ', command)
        return command.strip()

    def similarity_score(self, text1: str, text2: str) -> float:
        """Calculate similarity between two texts"""
        return SequenceMatcher(None, text1, text2).ratio()

    # =========================
    # PARAMETER EXTRACTION
    # =========================

    def _extract_search_query(self, command: str, keywords: List[str]) -> Dict[str, Any]:
        """Extract search query"""
        for keyword in keywords:
            if keyword in command:
                query = command.split(keyword, 1)[-1].strip()
                if query:
                    return {"query": query}
        return {"query": ""}

    def _extract_folder_name(self, command: str, keywords: List[str]) -> Dict[str, Any]:
        """Extract folder name"""
        for keyword in keywords:
            if keyword in command:
                name = command.split(keyword, 1)[-1].strip()
                if name:
                    return {"folder_name": name}
        return {"folder_name": "New Folder"}

    def _extract_file_name(self, command: str, keywords: List[str]) -> Dict[str, Any]:
        """Extract file name"""
        for keyword in keywords:
            if keyword in command:
                name = command.split(keyword, 1)[-1].strip()
                if name:
                    return {"file_name": name}
        return {"file_name": ""}

    def _extract_location(self, command: str, keywords: List[str]) -> Dict[str, Any]:
        """Extract location"""
        for keyword in keywords:
            if keyword in command:
                location = command.split(keyword, 1)[-1].strip()
                if location:
                    return {"location": location}
        return {"location": "current"}

    def _extract_query(self, command: str, keywords: List[str]) -> Dict[str, Any]:
        """Extract query"""
        for keyword in keywords:
            if keyword in command:
                query = command.split(keyword, 1)[-1].strip()
                if query:
                    return {"query": query}
        return {"query": ""}

    def _extract_message(self, command: str, keywords: List[str]) -> Dict[str, Any]:
        """Extract message"""
        for keyword in keywords:
            if keyword in command:
                message = command.split(keyword, 1)[-1].strip()
                if message:
                    return {"message": message}
        return {"message": ""}

    # =========================
    # INTENT CLASSIFICATION
    # =========================

    def classify(self, command: str) -> ClassificationResult:
        """
        Classify intent from command
        """
        raw_command = command
        cleaned_command = self.clean_command(command)

        best_intent = IntentType.UNKNOWN
        best_confidence = 0.0
        matched_keywords = []
        parameters = {}

        # Check each registered intent
        for intent, pattern in self.intent_patterns.items():
            for keyword in pattern.keywords:
                # Exact match
                if keyword in cleaned_command:
                    confidence = 0.8 + pattern.confidence_boost
                    matched_keywords.append(keyword)

                    if confidence > best_confidence:
                        best_confidence = confidence
                        best_intent = intent
                        parameters = pattern.parameter_extractor(
                            cleaned_command,
                            pattern.keywords
                        ) if pattern.parameter_extractor else {}

                # Fuzzy match
                else:
                    similarity = self.similarity_score(keyword, cleaned_command)
                    if similarity > self.fuzzy_threshold:
                        confidence = similarity * 0.7 + pattern.confidence_boost
                        matched_keywords.append(f"{keyword} (fuzzy)")

                        if confidence > best_confidence:
                            best_confidence = confidence
                            best_intent = intent
                            parameters = pattern.parameter_extractor(
                                cleaned_command,
                                pattern.keywords
                            ) if pattern.parameter_extractor else {}

        # Create result
        result = ClassificationResult(
            intent=best_intent,
            confidence=min(best_confidence, 1.0),
            parameters=parameters,
            raw_command=raw_command,
            cleaned_command=cleaned_command,
            timestamp=datetime.now(),
            matched_keywords=matched_keywords
        )

        # Add to history
        self.classification_history.append(result)
        if len(self.classification_history) > self.max_history:
            self.classification_history.pop(0)

        # Call callbacks
        for callback in self.callbacks:
            try:
                callback(result)
            except Exception as e:
                logger.warning(f"⚠️ Callback error: {e}")

        logger.info(
            f"✅ Classified: {result.intent.value} "
            f"(confidence: {result.confidence:.2f})"
        )

        return result

    # =========================
    # CALLBACKS
    # =========================

    def add_callback(self, callback: Callable[[ClassificationResult], None]) -> None:
        """Add callback for classification"""
        self.callbacks.append(callback)
        logger.info(f"✅ Callback added: {callback.__name__}")

    def remove_callback(self, callback: Callable) -> None:
        """Remove callback"""
        if callback in self.callbacks:
            self.callbacks.remove(callback)
            logger.info(f"✅ Callback removed: {callback.__name__}")

    # =========================
    # HISTORY & STATS
    # =========================

    def get_history(self, limit: int = 10) -> List[Dict]:
        """Get classification history"""
        history = []
        for result in self.classification_history[-limit:]:
            history.append({
                "intent": result.intent.value,
                "confidence": f"{result.confidence:.2f}",
                "command": result.raw_command,
                "parameters": result.parameters,
                "timestamp": result.timestamp.strftime("%H:%M:%S")
            })
        return history

    def get_stats(self) -> Dict[str, Any]:
        """Get statistics"""
        if not self.classification_history:
            return {"total_classifications": 0}

        total = len(self.classification_history)
        unknown_count = sum(
            1 for r in self.classification_history
            if r.intent == IntentType.UNKNOWN
        )
        avg_confidence = sum(
            r.confidence for r in self.classification_history
        ) / total

        return {
            "total_classifications": total,
            "recognized": total - unknown_count,
            "unknown": unknown_count,
            "success_rate": f"{((total - unknown_count) / total * 100):.1f}%",
            "average_confidence": f"{avg_confidence:.2f}"
        }

    def print_stats(self) -> None:
        """Print statistics"""
        stats = self.get_stats()
        print("\n" + "="*50)
        print("📊 CLASSIFICATION STATISTICS")
        print("="*50)
        for key, value in stats.items():
            print(f"{key.replace('_', ' ').title():.<35} {value}")
        print("="*50 + "\n")

    def print_history(self, limit: int = 10) -> None:
        """Print history"""
        history = self.get_history(limit)
        print("\n" + "="*50)
        print(f"📜 CLASSIFICATION HISTORY (Last {limit})")
        print("="*50)
        for i, item in enumerate(history, 1):
            print(f"\n{i}. {item['command']}")
            print(f"   Intent: {item['intent']} | Confidence: {item['confidence']}")
            if item['parameters']:
                print(f"   Parameters: {item['parameters']}")
            print(f"   Time: {item['timestamp']}")
        print("="*50 + "\n")

    def export_history(self, filename: str) -> None:
        """Export history to JSON"""
        history = self.get_history(len(self.classification_history))
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(history, f, ensure_ascii=False, indent=2)
        logger.info(f"✅ History exported to: {filename}")

    # =========================
    # UTILITIES
    # =========================

    def clear_history(self) -> None:
        """Clear classification history"""
        self.classification_history.clear()
        logger.info("🗑️ History cleared")

    def set_fuzzy_threshold(self, threshold: float) -> None:
        """Set fuzzy matching threshold (0.0 to 1.0)"""
        self.fuzzy_threshold = max(0.0, min(1.0, threshold))
        logger.info(f"✅ Fuzzy threshold set to: {self.fuzzy_threshold}")


# =========================
# UTILITY FUNCTIONS
# =========================

def classify_intent(command: str) -> ClassificationResult:
    """Quick function to classify intent"""
    classifier = AdvancedIntentClassifier()
    return classifier.classify(command)


# =========================
# TESTING & DEMONSTRATION
# =========================

def main():
    """
    Comprehensive test of the intent classifier
    """

    print("\n" + "="*50)
    print("🧠 ADVANCED INTENT CLASSIFIER")
    print("="*50 + "\n")

    classifier = AdvancedIntentClassifier()

    # Add callback
    def on_classification(result: ClassificationResult):
        if result.confidence < 0.6:
            print(f"⚠️ Low confidence: {result.confidence:.2f}")

    classifier.add_callback(on_classification)

    # Test commands
    test_commands = [
        "youtube kholo",
        "youtube pe python tutorial search karo",
        "open chrome",
        "google search karo machine learning ke bare mein",
        "folder banao project files",
        "screenshot lo",
        "pc band karo",
        "time kya hua hai",
        "music chalao",
        "unknown command",
        "whatsapp message bhejo hello",
    ]

    print("📝 Testing various commands...\n")

    for command in test_commands:
        result = classifier.classify(command)

        print(f"Command: {command}")
        print(f"Intent: {result.intent.value}")
        print(f"Confidence: {result.confidence:.2f}")
        if result.parameters:
            print(f"Parameters: {result.parameters}")
        print()
        time.sleep(0.3)

    # Show statistics
    classifier.print_stats()
    classifier.print_history(limit=5)

    # Export
    classifier.export_history("classification_history.json")

    print("✅ All tests complete!")


if __name__ == "__main__":
    try:
        import time
        main()
    except KeyboardInterrupt:
        print("\n\n⛔ Program interrupted by user")
    except Exception as e:
        logger.error(f"Fatal error: {e}")