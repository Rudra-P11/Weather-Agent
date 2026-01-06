from openai import OpenAI
from dotenv import load_dotenv
import requests
import os
import json

load_dotenv()

client = OpenAI(
    api_key=os.getenv("GEMINI_API_KEY"),
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
)

# ----------------------------
# TOOL (FUNCTION)
# ----------------------------
def get_weather(location: str) -> str:
    url = f"https://wttr.in/{location}"
    params = {"format": "%C + %t"}
    headers = {"User-Agent": "Mozilla/5.0"}

    try:
        response = requests.get(
            url,
            params=params,
            headers=headers,
            timeout=5
        )
        response.raise_for_status()
        return f"The current weather in {location} is {response.text.strip()}"
    except requests.exceptions.RequestException:
        return "Sorry, I couldn't fetch the weather right now."


# ----------------------------
# MAIN LOOP
# ----------------------------
def main():
    user_query = input("~> ")

    messages = [
        {
            "role": "system",
            "content": (
                "You are a helpful assistant. "
                "If the user asks for weather information, "
                "call the appropriate tool with the correct city name. "
                "Only call tools when necessary."
            )
        },
        {"role": "user", "content": user_query}
    ]

    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get the current weather for a city",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {
                            "type": "string",
                            "description": "City name, e.g. Dharwad, Paris, New York"
                        }
                    },
                    "required": ["location"]
                }
            }
        }
    ]

    response = client.chat.completions.create(
        model="gemini-2.5-flash",
        messages=messages,
        tools=tools,
        tool_choice="auto"
    )

    message = response.choices[0].message

    # ----------------------------
    # TOOL CALL HANDLING
    # ----------------------------
    if message.tool_calls:
        tool_call = message.tool_calls[0]
        args = json.loads(tool_call.function.arguments)

        weather_result = get_weather(args["location"])

        # Send tool result back to LLM
        messages.append(message)
        messages.append({
            "role": "tool",
            "tool_call_id": tool_call.id,
            "content": weather_result
        })

        final_response = client.chat.completions.create(
            model="gemini-2.5-flash",
            messages=messages
        )

        print("Agent~>", final_response.choices[0].message.content)
    else:
        print("Agent~>", message.content)


if __name__ == "__main__":
    main()
