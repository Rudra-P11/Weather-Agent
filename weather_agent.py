from openai import OpenAI
from dotenv import load_dotenv
import requests

load_dotenv()

client = OpenAI(
    api_key="GEMINI_API_KEY",
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
)

def get_weather(location: str):
    url = f"https://wttr.in/{location.lower()}?format=%C+%t"
    response = requests.get(url)

    if response.status_code == 200:
        return f"The current weather in {location} is: {response.text}"
    return f"Could not retrieve weather data for {location}."

def main():
    user_query = input("~> ")
    response = client.chat.completions.create(
        model="gemini-2.5-flash",
        messages=[
            {"role": "user", "content": user_query}
        ]
    )
    print(f"Agent~> " + response.choices[0].message.content)

print(get_weather("mumbai"))