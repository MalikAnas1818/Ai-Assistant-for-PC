"""
╔══════════════════════════════════════════════════════╗
║            CLEAN AI ASSISTANT PIPELINE              ║
║      STT → INTENT → LLM → FORMAT → TTS             ║
╚══════════════════════════════════════════════════════╝
"""

import time
import logging

from voice.speech_to_text import listen_command
from voice.text_to_speech import speak

from brain.intent_classifier import AdvancedIntentClassifier
from brain.llm_agent import AdvancedLLMAgent, ModelType

from brain.response_formatter import get_ai_speech_text


# =========================
# LOGGING
# =========================
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# =========================
# AI ASSISTANT CLASS
# =========================
class AIAssistant:

    def __init__(self):

        print("\n🤖 Initializing AI Assistant...\n")

        self.intent_classifier = AdvancedIntentClassifier()

        self.llm = AdvancedLLMAgent(
            model=ModelType.LLAMA_3_1_8B
        )

        self.running = True

        print("✅ Assistant Ready!\n")


    # =========================
    # PROCESS USER INPUT
    # =========================
    def process(self, text: str):

        print(f"\n🧑 User: {text}")

        # 1. INTENT CLASSIFICATION
        intent_result = self.intent_classifier.classify(text)

        print(f"🧠 Intent: {intent_result.intent.value}")
        print(f"📊 Confidence: {intent_result.confidence:.2f}")

        # =========================
        # 2. HIGH CONFIDENCE ACTION
        # =========================
        if intent_result.confidence > 0.75 and intent_result.intent.value != "unknown":

            reply = self.execute_action(intent_result)

            print(f"⚡ Action: {reply}")

            speak(reply)
            return


        # =========================
        # 3. LLM FALLBACK
        # =========================
        response = self.llm.ask(text)

        reply = get_ai_speech_text(response)

        print(f"🤖 AI: {reply}")

        speak(reply)


    # =========================
    # EXECUTE ACTIONS
    # =========================
    def execute_action(self, intent_result):

        intent = intent_result.intent.value
        params = intent_result.parameters

        if intent == "open_youtube":
            return "Main YouTube open kar raha hoon"

        elif intent == "youtube_search":
            query = params.get("query", "")
            return f"YouTube pe search kar raha hoon {query}"

        elif intent == "open_chrome":
            return "Chrome open kar raha hoon"

        elif intent == "google_search":
            query = params.get("query", "")
            return f"Google pe search kar raha hoon {query}"

        elif intent == "create_folder":
            name = params.get("folder_name", "New Folder")
            return f"Folder bana raha hoon {name}"

        elif intent == "take_screenshot":
            return "Screenshot le raha hoon"

        elif intent == "get_time":
            return f"Current time hai {time.strftime('%H:%M:%S')}"

        elif intent == "get_date":
            return f"Aaj ki date hai {time.strftime('%Y-%m-%d')}"

        elif intent == "play_music":
            return "Music chala raha hoon"

        elif intent == "pause_music":
            return "Music pause kar diya"

        elif intent == "stop_music":
            return "Music stop kar diya"

        elif intent == "send_whatsapp_message":
            msg = params.get("message", "")
            return f"WhatsApp message send kar raha hoon: {msg}"

        else:
            return "Main aapka command execute kar raha hoon"


    # =========================
    # VOICE MODE
    # =========================
    def run_voice_mode(self):

        print("\n🎤 Voice Mode Started...\n")

        while self.running:

            try:
                text = listen_command("urdu")

                if not text:
                    continue

                if text.lower() in ["exit", "stop", "band karo"]:
                    speak("Goodbye!")
                    break

                self.process(text)

            except KeyboardInterrupt:
                print("\n⛔ Stopping Assistant...")
                break

            except Exception as e:
                logger.error(f"Error: {e}")
                speak("Kuch error ho gaya hai")


    # =========================
    # TEXT MODE
    # =========================
    def run_text_mode(self):

        print("\n💬 Text Mode Started (type 'exit')\n")

        while True:

            text = input("You: ")

            if text.lower() == "exit":
                break

            self.process(text)


# =========================
# MAIN ENTRY
# =========================
if __name__ == "__main__":

    assistant = AIAssistant()

    print("""
=============================
🤖 AI ASSISTANT MODES
=============================
1. Voice Mode
2. Text Mode
=============================
""")

    mode = input("Select mode (1/2): ")

    if mode == "1":
        assistant.run_voice_mode()
    else:
        assistant.run_text_mode()