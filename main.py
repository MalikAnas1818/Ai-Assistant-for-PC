"""
🤖 Advanced AI Voice Assistant - Main Controller
مکمل AI Voice Assistant - مرکزی کنٹرولر

Features:
- Speech Recognition (سننا) -> voice.speech_to_text
- Intent Classification (سمجھنا) -> brain.intent_classifier
- LLM Agent (سوچنا) -> brain.llm_agent
- Voice Synthesis (بولنا) -> voice.text_to_speech
- Action Execution (کام کرنا) -> actions.*
- Error Handling & Logging
- Statistics & History
"""

import os
import sys
import logging
import time
import json
from typing import Dict, List, Optional, Callable, Any
from enum import Enum
from dataclasses import dataclass, asdict
from datetime import datetime
from collections import deque

# =====================================================================
# PIPELINE IMPORTS (Mapping to your specified folder structure)
# =====================================================================
try:
    # Voice Modules
    from voice.speech_to_text import AdvancedSpeechRecognizer, Language, RecognitionConfig
    from voice.text_to_speech import AdvancedVoiceSynthesizer, VoiceConfig
    
    # Brain Modules
    from brain.intent_classifier import AdvancedIntentClassifier, IntentType
    from brain.llm_agent import AdvancedLLMAgent, ModelType
    
    # Actions Modules (Imported contextually or directly if structured)
    # individual actions can also be grouped inside actions/ mapping
    IMPORTS_AVAILABLE = True
except ImportError as e:
    logging.warning(f"⚠️ Pipeline imports not fully available yet: {e}")
    IMPORTS_AVAILABLE = False


# =========================
# LOGGING SETUP
# =========================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('ai_assistant.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


# =========================
# ENUMS & DATA CLASSES
# =========================

class AssistantState(Enum):
    """Assistant states"""
    IDLE = "idle"
    LISTENING = "listening"
    PROCESSING = "processing"
    EXECUTING = "executing"
    SPEAKING = "speaking"
    ERROR = "error"
    SHUTDOWN = "shutdown"


@dataclass
class AssistantConfig:
    """Assistant configuration"""
    name: str = "AI Assistant"
    user_name: str = "Anis"
    language: str = "ur-PK"
    voice_rate: str = "-15%"
    voice_pitch: str = "+0Hz"
    model: str = "llama-3.1-8b-instant"
    enable_logging: bool = True
    enable_callbacks: bool = True
    max_history: int = 100
    auto_responses: bool = True
    debug_mode: bool = False


@dataclass
class CommandExecutionResult:
    """Result of command execution"""
    command: str
    intent: str
    success: bool
    response: str
    execution_time: float
    timestamp: datetime
    action_taken: Optional[str] = None
    details: Dict[str, Any] = None


# =========================
# ADVANCED AI ASSISTANT
# =========================

class AdvancedAIAssistant:
    """
    Advanced AI Voice Assistant aligned with the custom pipeline directory.
    """

    def __init__(self, config: AssistantConfig = None):
        """Initialize the AI Assistant"""
        self.config = config or AssistantConfig()
        self.state = AssistantState.IDLE
        self.is_running = False

        # Initialize pipeline modules placeholder
        self.speech_recognizer = None
        self.voice_synthesizer = None
        self.intent_classifier = None
        self.llm_agent = None

        # History & statistics
        self.command_history: deque = deque(maxlen=self.config.max_history)
        self.execution_results: List[CommandExecutionResult] = []
        self.callbacks: List[Callable] = []
        self.start_time = datetime.now()

        # Initialize modules from the folders
        self._initialize_pipeline_modules()

        logger.info(f"✅ {self.config.name} initialized for {self.config.user_name}")

    def _initialize_pipeline_modules(self) -> None:
        """Initialize sub-modules from voice/ and brain/ packages"""
        if not IMPORTS_AVAILABLE:
            logger.error("❌ Cannot initialize modules due to missing pipeline files.")
            return

        try:
            logger.info("🚀 Loading Pipeline Modules...")

            # 1. Speech Recognition (voice/speech_to_text.py)
            try:
                speech_config = RecognitionConfig(language=Language.URDU)
                self.speech_recognizer = AdvancedSpeechRecognizer(speech_config)
                logger.info("✅ Speech Recognizer loaded [voice.speech_to_text]")
            except Exception as e:
                logger.warning(f"⚠️ Speech Recognizer Error: {e}")

            # 2. Voice Synthesis (voice/text_to_speech.py)
            try:
                voice_config = VoiceConfig(
                    voice="ur-PK-UzmaNeural",
                    rate=self.config.voice_rate,
                    pitch=self.config.voice_pitch
                )
                self.voice_synthesizer = AdvancedVoiceSynthesizer(voice_config)
                logger.info("✅ Voice Synthesizer loaded [voice.text_to_speech]")
            except Exception as e:
                logger.warning(f"⚠️ Voice Synthesizer Error: {e}")

            # 3. Intent Classifier (brain/intent_classifier.py)
            try:
                self.intent_classifier = AdvancedIntentClassifier()
                logger.info("✅ Intent Classifier loaded [brain.intent_classifier]")
            except Exception as e:
                logger.warning(f"⚠️ Intent Classifier Error: {e}")

            # 4. LLM Agent (brain/llm_agent.py)
            try:
                self.llm_agent = AdvancedLLMAgent(
                    model=ModelType.LLAMA_3_1_8B,
                    temperature=0.3
                )
                logger.info("✅ LLM Agent loaded [brain.llm_agent]")
            except Exception as e:
                logger.warning(f"⚠️ LLM Agent Error: {e}")

            logger.info("✅ Core pipeline modules initialization completed.")

        except Exception as e:
            logger.error(f"❌ Critical initialization error: {e}")

    # =========================
    # STATE MANAGEMENT
    # =========================

    def set_state(self, state: AssistantState) -> None:
        self.state = state
        logger.info(f"📊 State: {state.value}")
        self._trigger_callbacks({"event": "state_change", "state": state.value})

    def get_state(self) -> AssistantState:
        return self.state

    # =========================
    # PIPELINE FLOW IMPLEMENTATION
    # =========================

    def listen(self) -> Optional[str]:
        """[voice.speech_to_text] Listening phase"""
        try:
            self.set_state(AssistantState.LISTENING)
            if not self.speech_recognizer:
                logger.error("❌ Speech Recognizer not available")
                return None

            logger.info("🎤 Listening...")
            result = self.speech_recognizer.recognize()

            if result:
                command = result.text
                logger.info(f"✅ Heard: {command}")
                self.set_state(AssistantState.IDLE)
                return command
            else:
                logger.warning("⚠️ No speech detected")
                self.set_state(AssistantState.IDLE)
                return None
        except Exception as e:
            logger.error(f"❌ Listen error: {e}")
            self.set_state(AssistantState.ERROR)
            return None

    def classify_intent(self, command: str) -> Dict[str, Any]:
        """[brain.intent_classifier] Understanding phase"""
        try:
            self.set_state(AssistantState.PROCESSING)
            if not self.intent_classifier:
                logger.error("❌ Intent Classifier not available")
                return {"intent": "unknown", "confidence": 0}

            result = self.intent_classifier.classify(command)
            logger.info(f"🧠 Intent Detected: {result.intent.value} (conf: {result.confidence:.2f})")

            return {
                "intent": result.intent.value,
                "confidence": result.confidence,
                "parameters": result.parameters,
                "matched_keywords": result.matched_keywords
            }
        except Exception as e:
            logger.error(f"❌ Classification error: {e}")
            self.set_state(AssistantState.ERROR)
            return {"intent": "unknown", "confidence": 0}

    def think(self, command: str) -> Dict[str, Any]:
        """[brain.llm_agent] Thinking fallback phase"""
        try:
            if not self.llm_agent:
                logger.error("❌ LLM Agent not available")
                return {"reply": "I cannot think right now"}

            logger.info("💭 Thinking...")
            response = self.llm_agent.ask(command, use_context=True)
            return response.content
        except Exception as e:
            logger.error(f"❌ Think error: {e}")
            return {"reply": f"Error occurred: {e}"}

    def speak(self, text: str) -> bool:
        """[voice.text_to_speech] Speaking phase"""
        try:
            self.set_state(AssistantState.SPEAKING)
            if not self.voice_synthesizer:
                logger.error("❌ Voice Synthesizer not available")
                self.set_state(AssistantState.IDLE)
                return False

            logger.info(f"🗣️ Speaking: {text[:50]}...")
            success = self.voice_synthesizer.speak(text)
            self.set_state(AssistantState.IDLE)
            return success
        except Exception as e:
            logger.error(f"❌ Speak error: {e}")
            self.set_state(AssistantState.ERROR)
            return False

    # =========================
    # ACTIONS EXECUTION (actions/ folder routing)
    # =========================

    def execute_action(self, intent: str, parameters: Dict[str, Any]) -> bool:
        """Routes execution to specialized modules inside actions/ folder"""
        try:
            self.set_state(AssistantState.EXECUTING)
            logger.info(f"⚡ Executing Action: {intent}")

            # WhatsApp Actions [actions/whatsapp.py]
            if intent == IntentType.SEND_WHATSAPP_MESSAGE.value:
                from actions.whatsapp import send_whatsapp
                return send_whatsapp(parameters.get("message", ""))

            # Folder/File Actions [actions/folder.py]
            elif intent == IntentType.CREATE_FOLDER.value:
                from actions.folder import create_new_folder
                return create_new_folder(parameters.get("folder_name", "New Folder"))

            # System Actions [actions/system.py]
            elif intent in [IntentType.SHUTDOWN_PC.value, IntentType.RESTART_PC.value, IntentType.SLEEP_PC.value]:
                from actions.system import handle_system_power
                return handle_system_power(intent)

            # Browser Actions [actions/browser.py]
            elif intent in [IntentType.OPEN_YOUTUBE.value, IntentType.YOUTUBE_SEARCH.value, IntentType.GOOGLE_SEARCH.value]:
                from actions.browser import handle_browser_action
                return handle_browser_action(intent, parameters)

            # Informational local actions
            elif intent == IntentType.GET_TIME.value:
                self.speak(f"وقت ہے {datetime.now().strftime('%H:%M')}")
                return True

            else:
                logger.warning(f"⚠️ No pipeline action mapped for intent: {intent}")
                return False

        except Exception as e:
            logger.error(f"❌ Action Execution error: {e}")
            self.set_state(AssistantState.ERROR)
            return False
        finally:
            if self.state == AssistantState.EXECUTING:
                self.set_state(AssistantState.IDLE)

    # =========================
    # SYSTEM CONTROL & LOGOOP
    # =========================

    def process_command(self, command: str) -> CommandExecutionResult:
        start_time = time.time()
        try:
            logger.info(f"\n{'='*60}\n📝 Command Input: {command}")
            
            intent_data = self.classify_intent(command)
            intent = intent_data.get("intent", "unknown")
            confidence = intent_data.get("confidence", 0)

            self.command_history.append({
                "command": command,
                "timestamp": datetime.now(),
                "intent": intent,
                "confidence": confidence
            })

            action_taken = None
            if confidence > 0.6 and intent != "unknown":
                parameters = intent_data.get("parameters", {})
                success = self.execute_action(intent, parameters)
                action_taken = intent if success else None
            else:
                logger.info("🔄 Confidence low or unknown intent. Handing over to LLM Agent...")
                self.speak("جی، مجھے سوچنے دیں")
                response = self.think(command)
                reply = response.get("reply", "معذرت، میں سمجھ نہیں پایا")
                if reply:
                    self.speak(reply)
                    action_taken = "llm_response"

            execution_time = time.time() - start_time
            result = CommandExecutionResult(
                command=command, intent=intent, success=confidence > 0.5,
                response=intent_data.get("reply", ""), execution_time=execution_time,
                timestamp=datetime.now(), action_taken=action_taken, details=intent_data
            )
            self.execution_results.append(result)
            self._trigger_callbacks({"event": "command_executed", "result": asdict(result)})
            return result

        except Exception as e:
            logger.error(f"❌ Process loop error: {e}")
            return CommandExecutionResult(
                command=command, intent="error", success=False,
                response=str(e), execution_time=time.time() - start_time, timestamp=datetime.now()
            )

    def start(self) -> None:
        """Starts the main assistant loop"""
        try:
            self.is_running = True
            print("\n" + "="*60)
            logger.info(f"🤖 {self.config.name} Started Successfully")
            print("="*60 + "\n")

            self.speak(f"اسلام علیکم {self.config.user_name}، میں آپ کا اسسٹنٹ ہوں۔ میں آپ کی کیا مدد کر سکتا ہوں؟")

            while self.is_running:
                try:
                    command = self.listen()
                    if not command:
                        continue

                    print(f"\n👤 {self.config.user_name}: {command}")

                    if any(word in command.lower() for word in ["exit", "stop", "quit", "خدا حافظ", "بائے"]):
                        self.speak(f"خدا حافظ {self.config.user_name}۔ آپ کا دن اچھا گزرے!")
                        break

                    self.process_command(command)

                except KeyboardInterrupt:
                    break
                except Exception as e:
                    logger.error(f"❌ Loop exception: {e}")
        finally:
            self.shutdown()

    def shutdown(self) -> None:
        self.set_state(AssistantState.SHUTDOWN)
        self.is_running = False
        self.print_stats()
        logger.info("✅ System shutdown complete.")

    def add_callback(self, callback: Callable) -> None:
        self.callbacks.append(callback)

    def _trigger_callbacks(self, data: Dict[str, Any]) -> None:
        if self.config.enable_callbacks:
            for callback in self.callbacks:
                try: callback(data)
                except: pass

    # Stats modules
    def get_stats(self) -> Dict[str, Any]:
        if not self.execution_results: return {"total_commands": 0}
        successful = sum(1 for r in self.execution_results if r.success)
        avg_time = sum(r.execution_time for r in self.execution_results) / len(self.execution_results)
        return {
            "total_commands": len(self.execution_results),
            "successful": successful,
            "success_rate": f"{(successful / len(self.execution_results) * 100):.1f}%",
            "average_execution_time": f"{avg_time:.2f}s",
            "uptime": str(datetime.now() - self.start_time)
        }

    def print_stats(self) -> None:
        stats = self.get_stats()
        print("\n" + "="*60 + "\n📊 ASSISTANT STATISTICS\n" + "="*60)
        for key, value in stats.items():
            print(f"{key.replace('_', ' ').title():.<40} {value}")
        print("="*60 + "\n")


# =========================
# MAIN ENTRY POINT
# =========================
def main():
    config = AssistantConfig(
        name="NovaMind",
        user_name="Anis",
        language="ur-PK"
    )
    assistant = AdvancedAIAssistant(config)
    assistant.start()

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n⛔ Program stopped.")
    except Exception as e:
        logger.error(f"Fatal crash: {e}")