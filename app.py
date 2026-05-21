import streamlit as st
import time
from datetime import datetime
import threading

# Import assistant core from the pipeline
try:
    from main import AdvancedAIAssistant, AssistantConfig, AssistantState
except ImportError:
    st.error("❌ 'main.py' not found. Please place this script in the root directory of your assistant project.")

# =====================================================================
# STREAMLIT CONFIGURATION
# =====================================================================
st.set_page_config(
    page_title="Advanced AI Voice Assistant",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for UI styling
st.markdown("""
    <style>
    .main { background-color: #0f1116; color: #ffffff; }
    .stButton>button { width: 100%; border-radius: 20px; background-color: #242936; color: white; }
    .stButton>button:hover { background-color: #3b4252; border-color: #4c566a; }
    .status-box { padding: 15px; border-radius: 10px; text-align: center; font-weight: bold; font-size: 1.2rem; }
    </style>
""", unsafe_allow_html=True)

# Initialize Assistant in Session State to prevent resetting on refresh
if "assistant" not in st.session_state:
    config = AssistantConfig(name="Advanced AI Assistant", user_name="Anis")
    st.session_state.assistant = AdvancedAIAssistant(config)
    st.session_state.chat_history = []
    st.session_state.is_listening = False

assistant = st.session_state.assistant

# =====================================================================
# SIDEBAR - STATS & CONFIG
# =====================================================================
with st.sidebar:
    st.title("⚙️ Control Panel")
    st.subheader(f"User: {assistant.config.user_name}")
    st.divider()
    
    # Live Status Display Setup
    state_colors = {
        "idle": ("🟢 IDLE", "#1e291b", "#4ade80"),
        "listening": ("🎤 LISTENING...", "#3b211d", "#f87171"),
        "processing": ("🧠 PROCESSING...", "#262e3b", "#60a5fa"),
        "executing": ("⚡ EXECUTING...", "#3b2a1a", "#fbbf24"),
        "speaking": ("🗣️ SPEAKING...", "#2e1a3b", "#c084fc"),
        "error": ("❌ ERROR", "#3b1a1a", "#ef4444"),
        "shutdown": ("🛑 SHUTDOWN", "#222", "#fff")
    }
    
    curr_state = assistant.get_state().value
    label, bg, text_col = state_colors.get(curr_state, ("Unknown", "#222", "#fff"))
    
    st.markdown(f'<div class="status-box" style="background-color: {bg}; color: {text_col};">{label}</div>', unsafe_allow_html=True)
    st.divider()
    
    # Performance Statistics
    st.subheader("📊 Performance")
    stats = assistant.get_stats()
    if "total_commands" in stats and stats["total_commands"] > 0:
        st.metric("Total Commands", stats["total_commands"])
        st.metric("Success Rate", stats["success_rate"])
        st.metric("Avg Execution Time", stats["average_execution_time"])
        st.caption(f"Uptime: {stats['uptime'].split('.')[0]}")
    else:
        st.info("No commands executed yet.")

# =====================================================================
# MAIN DASHBOARD
# =====================================================================
st.title("🤖 Advanced AI Voice Assistant")
st.caption("Full Urdu and English Voice Pipeline Controller")

# Text Input and Microphone layout alignment
col1, col2 = st.columns([5, 1])

with col2:
    # Microphone Trigger Button
    if st.button("🎤 Listen", use_container_width=True):
        st.session_state.is_listening = True
        
with col1:
    # Text Command Fallback Input Box
    text_command = st.text_input("Type here or click 'Listen' to talk...", key="input_box", placeholder="e.g., Open YouTube or play some music...")

# Voice Command Processing Flow
if st.session_state.is_listening:
    with st.spinner("🎤 Assistant is listening... Speak now..."):
        command = assistant.listen()
        st.session_state.is_listening = False  # Reset listening trigger flag
        if command:
            with st.spinner("🧠 Processing pipeline context..."):
                result = assistant.process_command(command)
                st.session_state.chat_history.append({"user": command, "assistant": result.action_taken or "LLM Response"})
            st.rerun()
        else:
            st.warning("⚠️ No voice input detected or microphone is unavailable.")

# Manual Text Command Processing Flow
if text_command:
    with st.spinner("🧠 Processing query..."):
        result = assistant.process_command(text_command)
        st.session_state.chat_history.append({"user": text_command, "assistant": result.action_taken or "LLM Response"})
    st.rerun()

st.divider()

# =====================================================================
# CHAT HISTORY / LOG DISPLAY
# =====================================================================
st.subheader("📜 Recent Chat & Action Logs")

if st.session_state.chat_history:
    for chat in reversed(st.session_state.chat_history):
        with st.chat_message("user"):
            st.write(chat["user"])
        with st.chat_message("assistant"):
            st.write(f"**Action Executed:** `{chat['assistant']}`")
else:
    st.caption("Your conversation history log will appear right here.")