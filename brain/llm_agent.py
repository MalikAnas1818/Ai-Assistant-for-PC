"""
╔══════════════════════════════════════════════════════════════╗
║            LLM AGENT                                        ║
║            Groq API | English Only                          ║
╚══════════════════════════════════════════════════════════════╝

INSTALL:
    pip install groq python-dotenv

.env file:
    GROQ_API_KEY=your_key_here
"""

import json
import logging
import os
import re
import time
from collections import deque
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, Iterator, List, Optional

from dotenv import load_dotenv
from groq import Groq


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
# ENV & CLIENT
# ══════════════════════════════════════════════════════════════

load_dotenv()
_api_key = os.getenv("GROQ_API_KEY")
if not _api_key:
    raise ValueError("GROQ_API_KEY not found in .env file")
client = Groq(api_key=_api_key)


# ══════════════════════════════════════════════════════════════
# ENUMS
# ══════════════════════════════════════════════════════════════

class ResponseType(Enum):
    ACTION    = "action"
    CHAT      = "chat"
    QUESTION  = "question"
    REASONING = "reasoning"
    ERROR     = "error"


class Model(Enum):
    LLAMA_8B   = "llama-3.1-8b-instant"      # Fast, lightweight
    LLAMA_70B  = "llama-3.1-70b-versatile"   # Accurate, slower
    MIXTRAL    = "mixtral-8x7b-32768"         # Long context
    GEMMA_7B   = "gemma-7b-it"               # Efficient


# ══════════════════════════════════════════════════════════════
# DATA CLASSES
# ══════════════════════════════════════════════════════════════

@dataclass
class AIResponse:
    response_type:   ResponseType
    content:         Dict[str, Any]
    raw_response:    str
    timestamp:       datetime
    processing_time: float
    model_used:      str
    tokens_used:     Optional[Dict[str, int]] = None


@dataclass
class Message:
    role:      str   # "user" | "assistant" | "system"
    content:   str
    timestamp: datetime


# ══════════════════════════════════════════════════════════════
# SYSTEM PROMPT
# ══════════════════════════════════════════════════════════════

SYSTEM_PROMPT = """You are an advanced AI desktop assistant. You understand English commands and respond in English only.

Your job is to classify user input and respond with structured JSON.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
RESPONSE FORMATS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

1. ACTION — when user wants to do something:
{
  "type": "action",
  "intent": "open_youtube|youtube_search|open_chrome|google_search|create_folder|delete_file|open_file|take_screenshot|shutdown_pc|restart_pc|sleep_pc|play_music|pause_media|stop_media|send_whatsapp|send_email|get_time|get_date|get_weather|open_calculator|open_notepad|unknown",
  "parameters": {
    "query": "search term if any",
    "file_name": "if applicable",
    "folder_name": "if applicable",
    "message": "if applicable",
    "location": "if applicable"
  },
  "confidence": 0.95,
  "reasoning": "brief explanation"
}

2. CHAT — normal conversation:
{
  "type": "chat",
  "reply": "your response here",
  "tone": "helpful|informative|friendly|technical"
}

3. QUESTION — need clarification:
{
  "type": "question",
  "clarification": "what you need to know",
  "options": ["option1", "option2"]
}

4. REASONING — complex analysis:
{
  "type": "reasoning",
  "thinking_process": "step by step reasoning",
  "conclusion": "final answer",
  "confidence": 0.85
}
EXAMPLES of correct JSON output:

User: "Documents folder in report.pdf ko Desktop move in"
Output: {"type": "action", "intent": "move_file", "parameters": {"source_path": "C:\\Users\\YourName\\Documents\\report.pdf", "destination_path": "C:\\Users\\YourName\\Desktop\\report.pdf"}}

User: "Start the song Shape of You by Ed Sheeran"
Output: {"type": "action", "intent": "youtube_search", "parameters": {"query": "Arijit Singh Tum Hi Ho"}}

User: "What are the files in my Downloads folder?"
Output: {"type": "action", "intent": "list_files", "parameters": {"folder_path": "C:\\Users\\YourName\\Downloads"}}

User: "Create a new folder named Projects in my Documents"
Output: {"type": "action", "intent": "create_folder", "parameters": {"folder_name": "Projects"}}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
RULES
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
- Always return valid JSON
- English only — do not respond in any other language
- Use confidence 0.0–1.0
- Keep responses concise
- Use "unknown" intent if unsure
- Never make up information
"""


# ══════════════════════════════════════════════════════════════
# LLM AGENT
# ══════════════════════════════════════════════════════════════

class LLMAgent:
    """
    Groq-powered English AI agent with:
    - Conversation memory
    - Multiple model support
    - Streaming
    - Callbacks
    - Stats tracking
    """

    def __init__(
        self,
        model:       Model = Model.LLAMA_8B,
        temperature: float = 0.3,
        max_history: int   = 50,
        system_prompt: str = SYSTEM_PROMPT
    ):
        self.model         = model
        self.temperature   = temperature
        self.system_prompt = system_prompt
        self.history: deque[Message] = deque(maxlen=max_history)
        self.responses:    List[AIResponse] = []
        self.callbacks:    List[Callable]   = []
        self.start_time    = datetime.now()

        self.stats = {
            "total":      0,
            "successful": 0,
            "failed":     0,
            "tokens":     0
        }

        logger.info(f"LLM Agent ready | Model: {model.value}")

    # ── Message Helpers ───────────────────────────────────────

    def _build_messages(self, query: str, use_context: bool) -> List[Dict]:
        messages = [{"role": "system", "content": self.system_prompt}]
        if use_context:
            for msg in self.history:
                messages.append({"role": msg.role, "content": msg.content})
        messages.append({"role": "user", "content": query})
        return messages

    def _parse(self, text: str) -> Dict[str, Any]:
        """Parse JSON from model response"""
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            match = re.search(r'\{.*\}', text, re.DOTALL)
            if match:
                try:
                    return json.loads(match.group())
                except json.JSONDecodeError:
                    pass
            return {"type": "chat", "reply": text}

    # ── Core Ask ──────────────────────────────────────────────

    def ask(self, query: str, use_context: bool = True) -> AIResponse:
        """
        Send a query and get a structured response.

        Args:
            query:       User input
            use_context: Include conversation history

        Returns:
            AIResponse with parsed content
        """
        t0 = time.time()
        self.stats["total"] += 1

        try:
            messages = self._build_messages(query, use_context)

            raw = client.chat.completions.create(
                model=self.model.value,
                messages=messages,
                temperature=self.temperature,
                max_tokens=1024
            )

            raw_text   = raw.choices[0].message.content
            content    = self._parse(raw_text)
            resp_type  = ResponseType(content.get("type", "chat"))
            tokens     = {
                "prompt":     raw.usage.prompt_tokens,
                "completion": raw.usage.completion_tokens,
                "total":      raw.usage.total_tokens
            }

            response = AIResponse(
                response_type=resp_type,
                content=content,
                raw_response=raw_text,
                timestamp=datetime.now(),
                processing_time=time.time() - t0,
                model_used=self.model.value,
                tokens_used=tokens
            )

            # Save to history
            self.history.append(Message("user",      query,    datetime.now()))
            self.history.append(Message("assistant", raw_text, datetime.now()))
            self.responses.append(response)
            if len(self.responses) > 100:
                self.responses.pop(0)

            self.stats["successful"] += 1
            self.stats["tokens"]     += tokens["total"]

            logger.info(
                f"→ {resp_type.value} | "
                f"{response.processing_time:.2f}s | "
                f"{tokens['total']} tokens"
            )

            for cb in self.callbacks:
                try:
                    cb(response)
                except Exception as e:
                    logger.warning(f"Callback error: {e}")

            return response

        except Exception as e:
            self.stats["failed"] += 1
            logger.error(f"API error: {e}")
            return AIResponse(
                response_type=ResponseType.ERROR,
                content={"error": str(e)},
                raw_response="",
                timestamp=datetime.now(),
                processing_time=time.time() - t0,
                model_used=self.model.value
            )

    # ── Streaming ─────────────────────────────────────────────

    def stream(self, query: str) -> Iterator[str]:
        """Stream response token by token"""
        messages = self._build_messages(query, use_context=True)
        try:
            with client.chat.completions.create(
                model=self.model.value,
                messages=messages,
                temperature=self.temperature,
                stream=True
            ) as stream:
                for chunk in stream:
                    token = chunk.choices[0].delta.content
                    if token:
                        yield token
        except Exception as e:
            logger.error(f"Stream error: {e}")
            yield f"Error: {e}"

    # ── Controls ──────────────────────────────────────────────

    def set_model(self, model: Model):
        self.model = model
        logger.info(f"Model → {model.value}")

    def set_temperature(self, temperature: float):
        self.temperature = max(0.0, min(1.0, temperature))

    def set_system_prompt(self, prompt: str):
        self.system_prompt = prompt
        logger.info("System prompt updated")

    def clear_history(self):
        self.history.clear()
        logger.info("Conversation history cleared")

    # ── Callbacks ─────────────────────────────────────────────

    def add_callback(self, callback: Callable[[AIResponse], None]):
        self.callbacks.append(callback)

    def remove_callback(self, callback: Callable):
        if callback in self.callbacks:
            self.callbacks.remove(callback)

    # ── Stats & Export ────────────────────────────────────────

    def get_stats(self) -> Dict[str, Any]:
        total = self.stats["total"]
        if not total:
            return {"total": 0}
        avg_time = (
            sum(r.processing_time for r in self.responses) / len(self.responses)
            if self.responses else 0
        )
        return {
            "total":            total,
            "successful":       self.stats["successful"],
            "failed":           self.stats["failed"],
            "success_rate":     f"{self.stats['successful'] / total * 100:.1f}%",
            "total_tokens":     self.stats["tokens"],
            "avg_response_time": f"{avg_time:.2f}s",
            "uptime":           str(datetime.now() - self.start_time)
        }

    def print_stats(self):
        stats = self.get_stats()
        print(f"\n{'='*45}")
        print("📊 AGENT STATS")
        print(f"{'='*45}")
        for k, v in stats.items():
            print(f"  {k:<22}: {v}")
        print(f"{'='*45}\n")

    def export_conversation(self, filename: str = "conversation.json"):
        data = {
            "metadata": {
                "start_time": self.start_time.isoformat(),
                "end_time":   datetime.now().isoformat(),
                "model":      self.model.value,
                "messages":   len(self.history)
            },
            "conversation": [
                {"role": m.role, "content": m.content, "timestamp": m.timestamp.isoformat()}
                for m in self.history
            ]
        }
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        logger.info(f"Conversation exported to {filename}")

    def chat(self, initial_query: str):
        """Interactive multi-turn conversation in terminal"""
        print(f"\n{'='*50}")
        print("💬 CHAT MODE  |  type 'exit' to quit")
        print(f"{'='*50}\n")

        print(f"You: {initial_query}")
        resp = self.ask(initial_query)
        print(f"AI : {resp.content.get('reply', json.dumps(resp.content))}\n")

        while True:
            try:
                user_input = input("You: ").strip()
                if not user_input or user_input.lower() in ("exit", "quit", "bye"):
                    print("AI : Goodbye!")
                    break
                resp = self.ask(user_input)
                print(f"AI : {resp.content.get('reply', json.dumps(resp.content))}\n")
            except KeyboardInterrupt:
                print("\nAI : Conversation ended.")
                break


# ══════════════════════════════════════════════════════════════
# QUICK HELPERS
# ══════════════════════════════════════════════════════════════

_agent: Optional[LLMAgent] = None

def ask_ai(query: str) -> Dict[str, Any]:
    """One-liner: ask_ai('open youtube')"""
    global _agent
    if _agent is None:
        _agent = LLMAgent()
    return _agent.ask(query).content


# ══════════════════════════════════════════════════════════════
# MAIN — TESTING
# ══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("\n" + "="*50)
    print("🤖 LLM AGENT — ENGLISH")
    print("="*50 + "\n")

    agent = LLMAgent(model=Model.LLAMA_8B, temperature=0.3)

    test_queries = [
        "open youtube",
        "search on youtube python tutorial",
        "open chrome",
        "shut down my computer",
        "what time is it?",
        "what's the weather in London?",
        "play some music",
        "hello, how are you?",
        "create a folder called my projects",
        "send a whatsapp message to John saying I'll be late",
    ]

    print("Testing queries...\n")

    for query in test_queries:
        print(f"  You: {query}")
        resp = agent.ask(query)
        print(f"  Type: {resp.response_type.value}")
        if resp.response_type == ResponseType.ACTION:
            print(f"  Intent: {resp.content.get('intent')} | Params: {resp.content.get('parameters')}")
        else:
            print(f"  Reply: {resp.content.get('reply', '')[:80]}")
        print(f"  Time: {resp.processing_time:.2f}s | Tokens: {resp.tokens_used['total']}\n")
        time.sleep(0.3)

    agent.print_stats()
    agent.export_conversation()

    print("✅ All tests complete!")