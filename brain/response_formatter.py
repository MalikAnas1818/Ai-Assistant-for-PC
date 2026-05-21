# brain/response_formatter.py

def get_ai_speech_text(response):
    content = response.content

    if content["type"] == "chat":
        return content.get("reply", "")

    elif content["type"] == "action":
        intent = content.get("intent", "")
        params = content.get("parameters", {})

        if intent == "open_file":
            return f"Main file {params.get('file_name', '')} khol raha hoon"

        elif intent == "open_youtube":
            return "Main YouTube open kar raha hoon"

        elif intent == "google_search":
            return f"Google par search kar raha hoon {params.get('query', '')}"

        elif intent == "shutdown_pc":
            return "Main computer shutdown kar raha hoon"

        elif intent == "play_music":
            return "Music chala raha hoon"

        else:
            return "Main aapka command execute kar raha hoon"

    elif content["type"] == "question":
        return content.get("clarification", "")

    return "Samajh nahi aaya"