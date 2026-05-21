from groq import Groq
from dotenv import load_dotenv
import os
import json
import logging
import time
import re
from typing import Dict, List, Optional, Callable, Any
from enum import Enum
from dataclasses import dataclass, asdict
from datetime import datetime
import threading
from collections import deque


# =========================
# LOGGING SETUP
# =========================

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# =========================
# LOAD ENV
# =========================

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    logger.error("❌ GROQ_API_KEY not found in .env file")
    raise ValueError("GROQ_API_KEY is required")

client = Groq(api_key=GROQ_API_KEY)


# =========================
# ENUMS & DATA CLASSES
# =========================

class ResponseType(Enum):
    """Types of responses from AI"""
    ACTION = "action"
    CHAT = "chat"
    QUESTION = "question"
    REASONING = "reasoning"
    ERROR = "error"


class IntentType(Enum):
    """Available intents"""
    # Web
    OPEN_YOUTUBE = "open_youtube"
    YOUTUBE_SEARCH = "youtube_search"
    OPEN_CHROME = "open_chrome"
    GOOGLE_SEARCH = "google_search"

    # Files
    CREATE_FOLDER = "create_folder"
    DELETE_FILE = "delete_file"
    OPEN_FILE = "open_file"

    # System
    TAKE_SCREENSHOT = "take_screenshot"
    SHUTDOWN_PC = "shutdown_pc"
    RESTART_PC = "restart_pc"
    SLEEP_PC = "sleep_pc"

    # Info
    GET_TIME = "get_time"
    GET_DATE = "get_date"
    GET_WEATHER = "get_weather"

    # Apps
    OPEN_CALCULATOR = "open_calculator"
    OPEN_NOTEPAD = "open_notepad"

    # Multimedia
    PLAY_MUSIC = "play_music"
    PAUSE_MUSIC = "pause_music"
    STOP_MUSIC = "stop_music"

    # Communication
    SEND_WHATSAPP_MESSAGE = "send_whatsapp_message"
    SEND_EMAIL = "send_email"

    # General
    UNKNOWN = "unknown"


class ModelType(Enum):
    """Available Groq models"""
    LLAMA_3_1_8B = "llama-3.1-8b-instant"
    LLAMA_3_1_70B = "llama-3.1-70b-versatile"
    MIXTRAL_8X7B = "mixtral-8x7b-32768"
    GEMMA_7B = "gemma-7b-it"


@dataclass
class AIResponse:
    """Response from AI agent"""
    response_type: ResponseType
    content: Dict[str, Any]
    raw_response: str
    timestamp: datetime
    processing_time: float
    model_used: str
    tokens_used: Optional[Dict[str, int]] = None
    confidence: float = 1.0


@dataclass
class ConversationMessage:
    """Message in conversation"""
    role: str  # user, assistant, system
    content: str
    timestamp: datetime
    type: str = "text"


# =========================
# ADVANCED LLM AGENT
# =========================

class AdvancedLLMAgent:
    """
    Advanced LLM-based AI agent with:
    - Conversation memory
    - Multiple models
    - Streaming support
    - Tool/intent recognition
    - Reasoning chains
    - Error recovery
    - Statistics tracking
    """

    def __init__(
        self,
        model: ModelType = ModelType.LLAMA_3_1_8B,
        max_conversation_history: int = 50,
        temperature: float = 0.3,
        enable_streaming: bool = False
    ):
        """Initialize the AI agent"""
        self.model = model
        self.client = client
        self.max_history = max_conversation_history
        self.temperature = temperature
        self.enable_streaming = enable_streaming

        # Conversation history
        self.conversation_history: deque = deque(maxlen=max_conversation_history)
        self.response_history: List[AIResponse] = []

        # Statistics
        self.total_requests = 0
        self.successful_requests = 0
        self.failed_requests = 0
        self.total_tokens_used = 0
        self.start_time = datetime.now()

        # Callbacks
        self.callbacks: List[Callable] = []

        # System prompt
        self.system_prompt = self._create_system_prompt()

        logger.info(f"✅ Advanced LLM Agent initialized with model: {model.value}")

    # =========================
    # SYSTEM PROMPT
    # =========================

    def _create_system_prompt(self) -> str:
        """Create comprehensive system prompt"""
        return """You are an advanced AI desktop assistant brain with knowledge and reasoning capabilities.

Your capabilities:
1. Understand user commands and intents
2. Classify actions vs conversations
3. Answer questions and provide information
4. Use tools and external functions
5. Maintain conversation context
6. Provide detailed reasoning when needed

RESPONSE FORMAT:

For action commands (return valid JSON):
{
  "type": "action",
  "intent": "open_youtube|youtube_search|open_chrome|google_search|create_folder|delete_file|open_file|take_screenshot|shutdown_pc|restart_pc|sleep_pc|play_music|pause_music|stop_music|send_whatsapp_message|send_email|get_time|get_date|get_weather|open_calculator|open_notepad|unknown",
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

For normal conversation (return JSON):
{
  "type": "chat",
  "reply": "your response",
  "tone": "helpful|informative|friendly|technical",
  "reasoning": "brief explanation if needed"
}

For questions requiring more info:
{
  "type": "question",
  "clarification": "what you need to clarify",
  "options": ["option1", "option2"],
  "reasoning": "why you need clarification"
}

For complex reasoning:
{
  "type": "reasoning",
  "thinking_process": "your reasoning chain",
  "conclusion": "final answer",
  "confidence": 0.85
}

IMPORTANT RULES:
- Always return valid JSON
- Use confidence scores (0.0 to 1.0)
- Keep responses concise but informative
- Maintain context from conversation history
- If unsure, ask for clarification
- Handle both Urdu and English commands
- Return "unknown" intent if not clear
- Never make up information
"""

    def set_system_prompt(self, prompt: str) -> None:
        """Set custom system prompt"""
        self.system_prompt = prompt
        logger.info("✅ System prompt updated")

    # =========================
    # CONVERSATION MANAGEMENT
    # =========================

    def add_message(
        self,
        role: str,
        content: str,
        message_type: str = "text"
    ) -> None:
        """Add message to conversation history"""
        message = ConversationMessage(
            role=role,
            content=content,
            timestamp=datetime.now(),
            type=message_type
        )
        self.conversation_history.append(message)
        logger.info(f"📝 Message added: {role} ({len(content)} chars)")

    def get_conversation_context(self, limit: int = 10) -> List[Dict[str, str]]:
        """Get recent conversation messages for context"""
        messages = []
        for msg in list(self.conversation_history)[-limit:]:
            messages.append({
                "role": msg.role,
                "content": msg.content
            })
        return messages

    def clear_conversation(self) -> None:
        """Clear conversation history"""
        self.conversation_history.clear()
        logger.info("🗑️ Conversation history cleared")

    def get_conversation_summary(self) -> Dict[str, Any]:
        """Get conversation summary"""
        if not self.conversation_history:
            return {"messages": 0, "summary": "No conversation yet"}

        user_messages = sum(1 for m in self.conversation_history if m.role == "user")
        assistant_messages = sum(1 for m in self.conversation_history if m.role == "assistant")

        return {
            "total_messages": len(self.conversation_history),
            "user_messages": user_messages,
            "assistant_messages": assistant_messages,
            "duration": str(datetime.now() - self.start_time)
        }

    # =========================
    # MODEL MANAGEMENT
    # =========================

    def set_model(self, model: ModelType) -> None:
        """Change the model"""
        self.model = model
        logger.info(f"✅ Model changed to: {model.value}")

    def set_temperature(self, temperature: float) -> None:
        """Set temperature (0.0 = deterministic, 1.0 = creative)"""
        self.temperature = max(0.0, min(1.0, temperature))
        logger.info(f"✅ Temperature set to: {self.temperature}")

    def get_available_models(self) -> List[str]:
        """Get list of available models"""
        return [model.value for model in ModelType]

    # =========================
    # AI INFERENCE
    # =========================

    def _prepare_messages(self, user_input: str) -> List[Dict[str, str]]:
        """Prepare messages for API call"""
        messages = [{"role": "system", "content": self.system_prompt}]

        # Add conversation context
        context = self.get_conversation_context(limit=10)
        messages.extend(context)

        # Add current user input
        messages.append({"role": "user", "content": user_input})

        return messages

    def _parse_response(self, response_text: str) -> Dict[str, Any]:
        """Parse and validate JSON response"""
        try:
            # Try direct JSON parsing
            return json.loads(response_text)
        except json.JSONDecodeError:
            # Try to extract JSON from response
            json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
            if json_match:
                try:
                    return json.loads(json_match.group())
                except json.JSONDecodeError:
                    pass

            # Fallback to chat response
            return {
                "type": "chat",
                "reply": response_text
            }

    def ask(
        self,
        query: str,
        use_context: bool = True,
        return_reasoning: bool = False
    ) -> AIResponse:
        """
        Main function to ask the AI agent
        """
        start_time = time.time()

        try:
            self.total_requests += 1

            logger.info(f"🤖 Processing query: {query[:50]}...")

            # Prepare messages
            messages = self._prepare_messages(query) if use_context else [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": query}
            ]

            # Call Groq API
            response = self.client.chat.completions.create(
                model=self.model.value,
                messages=messages,
                temperature=self.temperature,
                max_tokens=1024
            )

            raw_response = response.choices[0].message.content
            processing_time = time.time() - start_time

            # Parse response
            content = self._parse_response(raw_response)

            # Determine response type
            response_type = ResponseType(content.get("type", "chat"))

            # Create AI response
            ai_response = AIResponse(
                response_type=response_type,
                content=content,
                raw_response=raw_response,
                timestamp=datetime.now(),
                processing_time=processing_time,
                model_used=self.model.value,
                tokens_used={
                    "prompt": response.usage.prompt_tokens,
                    "completion": response.usage.completion_tokens,
                    "total": response.usage.total_tokens
                }
            )

            # Update statistics
            self.successful_requests += 1
            self.total_tokens_used += ai_response.tokens_used["total"]

            # Add to history
            self.conversation_history.append(
                ConversationMessage(
                    role="user",
                    content=query,
                    timestamp=datetime.now()
                )
            )

            self.conversation_history.append(
                ConversationMessage(
                    role="assistant",
                    content=raw_response,
                    timestamp=datetime.now()
                )
            )

            self.response_history.append(ai_response)
            if len(self.response_history) > 100:
                self.response_history.pop(0)

            logger.info(
                f"✅ Response generated in {processing_time:.2f}s "
                f"({ai_response.tokens_used['total']} tokens)"
            )

            # Call callbacks
            for callback in self.callbacks:
                try:
                    callback(ai_response)
                except Exception as e:
                    logger.warning(f"⚠️ Callback error: {e}")

            return ai_response

        except Exception as e:
            self.failed_requests += 1
            logger.error(f"❌ API Error: {e}")

            return AIResponse(
                response_type=ResponseType.ERROR,
                content={"error": str(e)},
                raw_response="",
                timestamp=datetime.now(),
                processing_time=time.time() - start_time,
                model_used=self.model.value
            )

    def stream_response(self, query: str):
        """Stream response token by token"""
        messages = self._prepare_messages(query)

        try:
            logger.info("🔄 Streaming response...")

            with self.client.chat.completions.create(
                model=self.model.value,
                messages=messages,
                temperature=self.temperature,
                stream=True
            ) as response:
                full_response = ""
                for chunk in response:
                    if chunk.choices[0].delta.content:
                        token = chunk.choices[0].delta.content
                        full_response += token
                        yield token

                return full_response

        except Exception as e:
            logger.error(f"❌ Streaming error: {e}")
            yield f"Error: {e}"

    # =========================
    # CALLBACKS
    # =========================

    def add_callback(self, callback: Callable[[AIResponse], None]) -> None:
        """Add callback for responses"""
        self.callbacks.append(callback)
        logger.info(f"✅ Callback added: {callback.__name__}")

    def remove_callback(self, callback: Callable) -> None:
        """Remove callback"""
        if callback in self.callbacks:
            self.callbacks.remove(callback)
            logger.info(f"✅ Callback removed: {callback.__name__}")

    # =========================
    # STATISTICS & ANALYSIS
    # =========================

    def get_stats(self) -> Dict[str, Any]:
        """Get agent statistics"""
        if self.total_requests == 0:
            return {"total_requests": 0}

        success_rate = (self.successful_requests / self.total_requests) * 100
        avg_processing_time = sum(
            r.processing_time for r in self.response_history
        ) / len(self.response_history) if self.response_history else 0

        return {
            "total_requests": self.total_requests,
            "successful": self.successful_requests,
            "failed": self.failed_requests,
            "success_rate": f"{success_rate:.1f}%",
            "total_tokens_used": self.total_tokens_used,
            "average_processing_time": f"{avg_processing_time:.2f}s",
            "uptime": str(datetime.now() - self.start_time)
        }

    def print_stats(self) -> None:
        """Print statistics"""
        stats = self.get_stats()
        print("\n" + "="*50)
        print("📊 AI AGENT STATISTICS")
        print("="*50)
        for key, value in stats.items():
            print(f"{key.replace('_', ' ').title():.<35} {value}")
        print("="*50 + "\n")

    def print_conversation_summary(self) -> None:
        """Print conversation summary"""
        summary = self.get_conversation_summary()
        print("\n" + "="*50)
        print("💬 CONVERSATION SUMMARY")
        print("="*50)
        for key, value in summary.items():
            print(f"{key.replace('_', ' ').title():.<35} {value}")
        print("="*50 + "\n")

    def get_response_history(self, limit: int = 10) -> List[Dict]:
        """Get response history"""
        history = []
        for response in self.response_history[-limit:]:
            history.append({
                "type": response.response_type.value,
                "processing_time": f"{response.processing_time:.2f}s",
                "tokens": response.tokens_used["total"] if response.tokens_used else 0,
                "timestamp": response.timestamp.strftime("%H:%M:%S")
            })
        return history

    def export_conversation(self, filename: str) -> None:
        """Export conversation to JSON"""
        data = {
            "metadata": {
                "start_time": self.start_time.isoformat(),
                "end_time": datetime.now().isoformat(),
                "total_messages": len(self.conversation_history),
                "model": self.model.value
            },
            "conversation": [
                {
                    "role": msg.role,
                    "content": msg.content,
                    "timestamp": msg.timestamp.isoformat()
                }
                for msg in self.conversation_history
            ]
        }

        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

        logger.info(f"✅ Conversation exported to: {filename}")

    # =========================
    # ADVANCED FEATURES
    # =========================

    def multi_turn_conversation(self, initial_query: str) -> None:
        """Start interactive multi-turn conversation"""
        logger.info("🎯 Starting multi-turn conversation mode")
        print(f"\n{'='*50}")
        print("💬 MULTI-TURN CONVERSATION MODE")
        print(f"{'='*50}\n")

        # Initial response
        print(f"You: {initial_query}")
        response = self.ask(initial_query)
        print(f"AI: {response.content.get('reply', json.dumps(response.content))}\n")

        # Continue conversation
        while True:
            try:
                user_input = input("You: ").strip()
                if not user_input or user_input.lower() in ["exit", "quit", "bye"]:
                    print("\nAI: Goodbye! Have a great day! 👋")
                    break

                response = self.ask(user_input)
                print(f"AI: {response.content.get('reply', json.dumps(response.content))}\n")

            except KeyboardInterrupt:
                print("\n\nAI: Conversation interrupted. See you next time!")
                break

    def analyze_intent(self, query: str) -> Dict[str, Any]:
        """Analyze and extract intent from query"""
        specialized_prompt = f"""Analyze this query and extract the intent:
        
Query: "{query}"

Return JSON with:
- detected_intent: the action type
- confidence: 0-1
- parameters: extracted data
- alternative_intents: if any
- interpretation: what the user wants
"""
        response = self.ask(specialized_prompt)
        return response.content

    def generate_response_with_reasoning(self, query: str) -> Dict[str, Any]:
        """Generate response with detailed reasoning"""
        response = self.ask(query, return_reasoning=True)
        return {
            "response": response.content,
            "reasoning": response.content.get("reasoning", ""),
            "confidence": response.content.get("confidence", 0.9),
            "processing_time": response.processing_time
        }


# =========================
# UTILITY FUNCTIONS
# =========================

def quick_ask(query: str, model: ModelType = ModelType.LLAMA_3_1_8B) -> Dict[str, Any]:
    """Quick function to ask without maintaining state"""
    agent = AdvancedLLMAgent(model=model)
    response = agent.ask(query, use_context=False)
    return response.content


# =========================
# TESTING & DEMONSTRATION
# =========================

def main():
    """Comprehensive test of the AI agent"""

    print("\n" + "="*50)
    print("🤖 ADVANCED LLM AI AGENT")
    print("="*50 + "\n")

    # Initialize agent
    agent = AdvancedLLMAgent(
        model=ModelType.LLAMA_3_1_8B,
        temperature=0.3,
        max_conversation_history=50
    )

    # Add callback
    def on_response(response: AIResponse):
        logger.info(f"Response type: {response.response_type.value}")

    agent.add_callback(on_response)

    # Test queries
    test_queries = [
        "youtube pe python tutorial search karo",
        "chrome open karo",
        "mera computer shutdown kar do",
        "hello kya haal hai?",
        "weather kaisa hai Lahore mein?",
        "music play kar",
    ]

    print("📝 Testing various queries...\n")

    for query in test_queries:
        print(f"You: {query}")
        response = agent.ask(query)

        print(f"Type: {response.response_type.value}")
        print(f"Response: {response.content}")
        print(f"Processing Time: {response.processing_time:.2f}s")
        print(f"Tokens Used: {response.tokens_used['total']}\n")
        time.sleep(0.5)

    # Show statistics
    agent.print_stats()
    agent.print_conversation_summary()

    # Export conversation
    agent.export_conversation("ai_conversation.json")

    print("✅ All tests complete!")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⛔ Program interrupted by user")
    except Exception as e:
        logger.error(f"Fatal error: {e}")

# =========================
# SIMPLE ASK_AI WRAPPER
# =========================

_agent = AdvancedLLMAgent()

def ask_ai(query: str) -> dict:
    response = _agent.ask(query)
    return response.content