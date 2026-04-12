from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
client = OpenAI()

while True:
    user_input = input("\nYou: ")

    if user_input.lower() == "exit":
        print("Goodbye!")
        break

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "user", "content": user_input}
        ]
    )

    print("AI:", response.choices[0].message.content)
