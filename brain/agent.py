"""
Agent — multi-step tasks handle karta hai.
LLM se pehle plan maangta hai, phir ek ek step ActionExecutor se chalata hai.
"""
import json
import logging
from brain.llm_agent import LLMAgent, Model

logger = logging.getLogger(__name__)

PLAN_PROMPT = """
User ne yeh kaha hai: "{user_input}"

Yeh kaam karne ke liye steps batao as a JSON list.
Har step mein sirf yeh intents use karo:
move_file, copy_file, create_folder, delete_file, open_file,
list_files, open_youtube, youtube_search, google_search,
play_music, send_whatsapp, shutdown_pc, restart_pc,
get_time, get_date, open_calculator, open_notepad, take_screenshot

Format (sirf JSON, koi text nahi):
[
  {{"intent": "move_file", "parameters": {{"source_path": "...", "destination_path": "..."}}}},
  {{"intent": "youtube_search", "parameters": {{"query": "..."}}}}
]
"""

class Agent:
    def __init__(self, llm: LLMAgent, executor):
        self.llm = llm
        self.executor = executor

    def is_multi_step(self, user_input: str) -> bool:
        """Check karo kya kaam mein multiple actions hain."""
        keywords = ["aur", "phir", "then", "and", "also", "ke baad", "baad mein", "saath mein"]
        return any(kw in user_input.lower() for kw in keywords)

    def run(self, user_input: str) -> str:
        """Multi-step task plan karo aur chalao."""
        prompt = PLAN_PROMPT.format(user_input=user_input)
        
        # LLM se plan maango
        raw = self.llm.ask(prompt)
        plan_text = raw.content.get("text", "[]") if isinstance(raw.content, dict) else "[]"
        
        try:
            # JSON extract karo
            start = plan_text.find("[")
            end   = plan_text.rfind("]") + 1
            steps = json.loads(plan_text[start:end])
        except Exception as e:
            logger.error(f"Plan parse nahi hua: {e}")
            return "Sorry, plan samajh nahi aaya. Thoda clearly bolein."

        if not steps:
            return "Koi steps nahi mile plan mein."

        results = []
        for i, step in enumerate(steps):
            intent = step.get("intent", "")
            params = step.get("parameters", {})
            logger.info(f"Step {i+1}: {intent} | {params}")
            
            success = self.executor.execute(intent, params)
            status  = "✓ Done" if success else "✗ Failed"
            results.append(f"Step {i+1} ({intent}): {status}")

        return "\n".join(results)