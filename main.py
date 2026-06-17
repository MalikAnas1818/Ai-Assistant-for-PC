# main.py
# AI Assistant — Main Pipeline
#
# Pipeline:
#   Microphone → speech_to_text → intent_classifier → llm_agent
#               → response_formatter → action_executor → text_to_speech → Speaker

import logging
import sys

from voice.speech_to_text import SpeechRecognizer
from voice.text_to_speech import TextToSpeech, VoiceConfig, Voice
from brain.intent_classifier import IntentClassifier
from brain.llm_agent import LLMAgent, Model
from brain.agent import Agent
from brain.response_formatter import get_speech_text
from actions.browser import AdvancedBrowserManager, SearchEngine
from actions.folder import AdvancedFileManager
from actions.system import shutdown_pc, restart_pc, lock_pc
from actions.whatsapp import send_instant_message


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
# ACTION EXECUTOR
# ══════════════════════════════════════════════════════════════

class ActionExecutor:
    """
    Executes actions based on classified intent + parameters.
    Each method maps to one IntentType.
    """

    def __init__(self):
        self.browser = AdvancedBrowserManager()
        self.files   = AdvancedFileManager()

    def execute(self, intent: str, params: dict) -> bool:
        """
        Route intent to the correct action function.

        Args:
            intent: intent string from LLM/classifier (e.g. "open_youtube")
            params: parameters dict extracted from command

        Returns:
            True if action succeeded, False otherwise
        """
        handler = self._handlers().get(intent)

        if handler:
            try:
                return handler(params)
            except Exception as e:
                logger.error(f"Action error [{intent}]: {e}")
                return False
        else:
            logger.warning(f"No handler for intent: {intent}")
            return False

    def _handlers(self) -> dict:
        return {
            # Web
            "open_youtube":    self._open_youtube,
            "youtube_search":  self._youtube_search,
            "open_chrome":     self._open_chrome,
            "google_search":   self._google_search,
            # File System
            "create_folder":   self._create_folder,
            "delete_file":     self._delete_file,
            "open_file":       self._open_file,
            "move_file":       self._move_file,
            "copy_file":       self._copy_file,
            "list_files":       self._list_files,
            # System
            "take_screenshot": self._take_screenshot,
            "shutdown_pc":     self._shutdown,
            "restart_pc":      self._restart,
            "sleep_pc":        self._sleep,
            # System Info
            "get_time":        self._get_time,
            "get_date":        self._get_date,
            "get_weather":     self._get_weather,
            # Apps
            "open_calculator": self._open_calculator,
            "open_notepad":    self._open_notepad,
            # Multimedia
            "play_music":      self._play_music,
            "play_video":      self._play_video,
            "pause_media":     self._pause_media,
            "stop_media":      self._stop_media,
            # Communication
            "send_whatsapp":   self._send_whatsapp,
            "send_email":      self._send_email,
        }

    # ── Web ───────────────────────────────────────────────────

    def _open_youtube(self, params: dict) -> bool:
        return self.browser.open_youtube()

    def _youtube_search(self, params: dict) -> bool:
        query = params.get("query", "")
        return self.browser.youtube_search(query) if query else self.browser.open_youtube()

    def _open_chrome(self, params: dict) -> bool:
        return self.browser.open_website("https://google.com")

    def _google_search(self, params: dict) -> bool:
        query = params.get("query", "")
        return self.browser.google_search(query) if query else self.browser.open_google()

    # ── File System ───────────────────────────────────────────

    def _create_folder(self, params: dict) -> bool:
        name   = params.get("folder_name", "New Folder")
        result = self.files.create_folder(name, drive="C")
        return result is not None

    def _delete_file(self, params: dict) -> bool:
        path = params.get("file_name", "")
        if not path:
            logger.warning("delete_file: no file_name provided")
            return False
        return self.files.delete_file(path)

    def _open_file(self, params: dict) -> bool:
        import subprocess
        path = params.get("file_name", "")
        if not path:
            logger.warning("open_file: no file_name provided")
            return False
        try:
            subprocess.Popen(["start", path], shell=True)
            return True
        except Exception as e:
            logger.error(f"open_file error: {e}")
            return False
        
    def _move_file(self, params: dict) -> bool:
        source = params.get("source_path", "")
        dest   = params.get("destination_path", "")
        if not source or not dest:
            logger.warning("move_file: source ya destination missing hai")
            return False
        return self.files.move_file(source, dest)
    
    def _copy_file(self, params: dict) -> bool:
        source = params.get("source_path", "")
        dest   = params.get("destination_path", "")
        if not source or not dest:
            logger.warning("copy_file: source ya destination missing hai")
            return False
        return self.files.copy_file(source, dest)
    
    def _list_files(self, params: dict) -> bool:
        folder = params.get("folder_path", "C:\\")
        items  = self.files.list_files(folder)
        logger.info(f"Found {len(items)} items")
        return True

    # ── System ────────────────────────────────────────────────

    def _take_screenshot(self, params: dict) -> bool:
        try:
            import pyautogui
            from datetime import datetime
            filename = f"screenshot_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
            pyautogui.screenshot(filename)
            logger.info(f"Screenshot saved: {filename}")
            return True
        except Exception as e:
            logger.error(f"Screenshot error: {e}")
            return False

    def _shutdown(self, params: dict) -> bool:
        shutdown_pc()
        return True

    def _restart(self, params: dict) -> bool:
        restart_pc()
        return True

    def _sleep(self, params: dict) -> bool:
        import os
        os.system("rundll32.exe powrprof.dll,SetSuspendState 0,1,0")
        return True

    # ── System Info ───────────────────────────────────────────

    def _get_time(self, params: dict) -> bool:
        from datetime import datetime
        now = datetime.now().strftime("%I:%M %p")
        logger.info(f"Current time: {now}")
        return True

    def _get_date(self, params: dict) -> bool:
        from datetime import datetime
        today = datetime.now().strftime("%A, %B %d, %Y")
        logger.info(f"Today's date: {today}")
        return True

    def _get_weather(self, params: dict) -> bool:
        location = params.get("location", "current")
        return self.browser.google_search(f"weather {location}")

    # ── Apps ──────────────────────────────────────────────────

    def _open_calculator(self, params: dict) -> bool:
        import os
        os.system("calc")
        return True

    def _open_notepad(self, params: dict) -> bool:
        import os
        os.system("notepad")
        return True

    # ── Multimedia ────────────────────────────────────────────

    def _play_music(self, params: dict) -> bool:
        query = params.get("query", "")
        search = query + " music" if query else "music"
        return self.browser.youtube_search(search)

    def _play_video(self, params: dict) -> bool:
        query = params.get("query", "")
        return self.browser.youtube_search(query) if query else self.browser.open_youtube()

    def _pause_media(self, params: dict) -> bool:
        try:
            import pyautogui
            pyautogui.press("space")
            return True
        except Exception as e:
            logger.error(f"pause_media error: {e}")
            return False

    def _stop_media(self, params: dict) -> bool:
        return self._pause_media(params)

    # ── Communication ─────────────────────────────────────────

    def _send_whatsapp(self, params: dict) -> bool:
        contact = params.get("contact", "")
        message = params.get("message", "")
        if not contact or not message:
            logger.warning("send_whatsapp: missing contact or message")
            return False
        return send_instant_message(contact, message)

    def _send_email(self, params: dict) -> bool:
        return self.browser.open_website("https://mail.google.com/#compose")


# ══════════════════════════════════════════════════════════════
# ASSISTANT
# ══════════════════════════════════════════════════════════════

class AIAssistant:
    """
    Full pipeline:
    Voice In → LLM Brain → Action Executor → Voice Out
    """

    STOP_PHRASES = {"stop", "exit", "quit", "goodbye", "bye"}

    def __init__(self):
        logger.info("Initializing AI Assistant...")
        
        
        self.stt      = SpeechRecognizer(whisper_model="base")
        self.tts      = TextToSpeech(VoiceConfig(voice=Voice.FEMALE.value))
        self.llm      = LLMAgent(model=Model.LLAMA_8B, temperature=0.3)
        self.executor = ActionExecutor()
        self.agent    = Agent(self.llm, self.executor)

        logger.info("AI Assistant ready!")

    # ── Single Turn ───────────────────────────────────────────

    def process(self, user_input: str) -> str:
        logger.info(f"User: {user_input}")
        
        if self.agent.is_multi_step(user_input):
            result = self.agent.run(user_input)
            return result
        
        ai_response = self.llm.ask(user_input)
        content = ai_response.content
        if content.get("type") == "action":
            intent = content.get("intent", "unknown")
            params = content.get("parameters", {})
            if intent != "unknown":
                self.executor.execute(intent, params)
                
        from brain.response_formatter import get_speech_text
        return get_speech_text(ai_response)

    # ── Voice Loop ────────────────────────────────────────────

    def listen_and_respond(self) -> bool:
        """Listen once, process, speak. Returns False to stop."""
        user_input = self.stt.listen()

        if not user_input:
            self.tts.speak("I didn't catch that. Please try again.")
            return True

        print(f"\n  You : {user_input}")

        if user_input.lower().strip() in self.STOP_PHRASES:
            self.tts.speak("Goodbye! Have a great day.")
            return False

        speech_text = self.process(user_input)
        self.tts.speak(speech_text)
        return True

    def run(self):
        """Start the continuous voice loop."""
        print("\n" + "="*55)
        print("  AI ASSISTANT — VOICE MODE")
        print(f"  Say one of {self.STOP_PHRASES} to exit")
        print("="*55 + "\n")

        self.tts.speak("Hello! I'm your AI assistant. How can I help you?")

        while True:
            try:
                if not self.listen_and_respond():
                    break
            except KeyboardInterrupt:
                print("\nStopped.")
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

    # Voice mode (default)
    if mode == "voice":
        assistant.run()

    # Text mode — type commands instead of speaking (best for testing)
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
        print("  python main.py text     → text mode  (keyboard, for testing)")