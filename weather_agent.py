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

def get_weather(location: str) -> str:
    url = f"https://wttr.in/{location.strip()}"
    params = {"format": "%C + %t"}
    headers = {"User-Agent": "Mozilla/5.0"}

    try:
        response = requests.get(url, params=params, headers=headers, timeout=5)
        response.raise_for_status()
        return f"The current weather in {location} is {response.text.strip()}"
    except requests.exceptions.RequestException:
        return "Sorry, I couldn't fetch the weather right now."

def main():
    print("Hi there, I'm here to assist your queries. Type 'exit' to quit.\n")

    waiting_for_city = False

    while True:
        user_query = input("~> ").strip()

        if user_query.lower() in {"exit", "quit"}:
            print("Agent~> Goodbye!")
            break

        if waiting_for_city:
            user_query = f"weather in {user_query}"
            waiting_for_city = False

        messages = [
            {
                "role": "system",
                "content": (
                    "You are a helpful assistant. "
                    "If the user asks for weather information, "
                    "call the appropriate tool with the correct city name. "
                    "If the user asks about weather without providing a city, "
                    "ask for the city."
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
                                "description": "City name, e.g. Dharwad, Mumbai, New Delhi"
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

        if message.tool_calls:
            tool_call = message.tool_calls[0]
            args = json.loads(tool_call.function.arguments)

            weather_result = get_weather(args["location"])

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
            if "city" in message.content.lower():
                waiting_for_city = True

if __name__ == "__main__":
    main()
