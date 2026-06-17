# main.py
# AI Assistant — Main Pipeline
#
# Pipeline:
#   Microphone → speech_to_text → intent_classifier → llm_agent
#               → response_formatter → text_to_speech → Speaker

import logging
import sys

from voice.speech_to_text import SpeechRecognizer
from voice.text_to_speech import TextToSpeech, VoiceConfig, Voice
from brain.intent_classifier import IntentClassifier
from brain.llm_agent import LLMAgent, Model
from brain.response_formatter import get_speech_text

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
# ASSISTANT
# ══════════════════════════════════════════════════════════════

class AIAssistant:
    """
    Full pipeline:
    Voice In → Intent Check → LLM Brain → Voice Out
    """

    STOP_PHRASES = {"stop", "exit", "quit", "goodbye", "bye"}

    def __init__(self):
        logger.info("Initializing AI Assistant...")

        self.stt        = SpeechRecognizer(whisper_model="base")
        self.tts        = TextToSpeech(VoiceConfig(voice=Voice.FEMALE.value))
        self.classifier = IntentClassifier()
        self.llm        = LLMAgent(model=Model.LLAMA_8B, temperature=0.3)

        logger.info("AI Assistant ready!")

    # ── Single Turn ───────────────────────────────────────────

    def process(self, user_input: str) -> str:
        """
        Run one full pipeline cycle on a text input.
        Returns the assistant's speech text.
        """
        logger.info(f"User: {user_input}")

        # Step 1 — Quick local intent check (no API call needed for simple actions)
        local = self.classifier.classify(user_input)
        logger.info(f"Local intent: {local.intent.value} ({local.confidence:.2f})")

        # Step 2 — Send to LLM for final decision + response
        ai_response = self.llm.ask(user_input)

        # Step 3 — Format into speakable text
        speech_text = get_speech_text(ai_response)
        logger.info(f"Response: {speech_text}")

        return speech_text

    # ── Voice Loop ────────────────────────────────────────────

    def listen_and_respond(self):
        """Listen once, process, speak."""
        user_input = self.stt.listen()

        if not user_input:
            self.tts.speak("I didn't catch that. Please try again.")
            return True  # keep running

        print(f"\n  You : {user_input}")

        # Stop command check
        if user_input.lower().strip() in self.STOP_PHRASES:
            self.tts.speak("Goodbye! Have a great day.")
            return False  # stop loop

        speech_text = self.process(user_input)
        self.tts.speak(speech_text)
        return True  # keep running

    # ── Run ───────────────────────────────────────────────────

    def run(self):
        """Start the continuous voice loop."""
        print("\n" + "="*55)
        print("  AI ASSISTANT")
        print(f"  Say one of {self.STOP_PHRASES} to exit")
        print("="*55 + "\n")

        self.tts.speak("Hello! I'm your AI assistant. How can I help you?")

        while True:
            try:
                keep_running = self.listen_and_respond()
                if not keep_running:
                    break
            except KeyboardInterrupt:
                print("\n\nStopped by user.")
                self.tts.speak("Goodbye!")
                break
            except Exception as e:
                logger.error(f"Pipeline error: {e}")
                self.tts.speak("Something went wrong. Please try again.")


# ══════════════════════════════════════════════════════════════
# ENTRY POINT
# ══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "voice"

    assistant = AIAssistant()

    # ── Voice Mode (default) ──────────────────────────────────
    if mode == "voice":
        assistant.run()

    # ── Text Mode (type instead of speak — good for testing) ──
    elif mode == "text":
        print("\n" + "="*55)
        print("  AI ASSISTANT — TEXT MODE")
        print("  Type 'exit' to quit")
        print("="*55 + "\n")

        while True:
            try:
                user_input = input("You: ").strip()
                if not user_input:
                    continue
                if user_input.lower() in AIAssistant.STOP_PHRASES:
                    print("AI : Goodbye!")
                    break
                speech_text = assistant.process(user_input)
                print(f"AI : {speech_text}\n")
            except KeyboardInterrupt:
                print("\nGoodbye!")
                break

    else:
        print("Usage:")
        print("  python main.py          → voice mode (mic input)")
        print("  python main.py voice    → voice mode (mic input)")
        print("  python main.py text     → text mode  (keyboard input)")