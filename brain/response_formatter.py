# brain/response_formatter.py

# Maps each intent to a human-readable English response.
# {query}, {file_name}, etc. are filled in from parameters.

INTENT_MESSAGES = {
    # Web
    "open_youtube":      "Opening YouTube.",
    "youtube_search":    "Searching YouTube for {query}.",
    "open_chrome":       "Opening Chrome.",
    "google_search":     "Searching Google for {query}.",
    # File System
    "create_folder":     "Creating folder '{folder_name}'.",
    "delete_file":       "Deleting file '{file_name}'.",
    "open_file":         "Opening file '{file_name}'.",
    # System
    "take_screenshot":   "Taking a screenshot.",
    "shutdown_pc":       "Shutting down the computer.",
    "restart_pc":        "Restarting the computer.",
    "sleep_pc":          "Putting the computer to sleep.",
    # System Info
    "get_time":          "Let me check the current time.",
    "get_date":          "Let me check today's date.",
    "get_weather":       "Checking the weather for {location}.",
    # Apps
    "open_calculator":   "Opening Calculator.",
    "open_notepad":      "Opening Notepad.",
    # Multimedia
    "play_music":        "Playing music.",
    "play_video":        "Playing video.",
    "pause_media":       "Pausing.",
    "stop_media":        "Stopping.",
    # Communication
    "send_whatsapp":     "Sending WhatsApp message.",
    "send_email":        "Sending email.",
    # Fallback
    "unknown":           "I'm not sure how to handle that.",
}


def get_speech_text(response) -> str:
    """
    Convert an AIResponse into a plain English string for TTS.

    Args:
        response: AIResponse object (must have a .content dict)

    Returns:
        Human-readable English string
    """
    content = response.content
    resp_type = content.get("type", "")

    if resp_type == "chat":
        return content.get("reply", "I didn't get a reply.")

    elif resp_type == "action":
        intent = content.get("intent", "unknown")
        params = content.get("parameters", {})
        template = INTENT_MESSAGES.get(intent, "Executing your command.")
        try:
            return template.format(**params)
        except KeyError:
            return template

    elif resp_type == "question":
        return content.get("clarification", "Could you clarify that?")

    elif resp_type == "reasoning":
        return content.get("conclusion", "Let me think about that.")

    elif resp_type == "error":
        return "Something went wrong. Please try again."

    return "I didn't understand that."