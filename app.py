from openai import OpenAI
from dotenv import load_dotenv
import json

load_dotenv()
client = OpenAI()

while True:
    user_input = input("\nYou: ")

    if user_input.lower() == "exit":
        break

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": "You are an assistant that ONLY responds in valid JSON."
            },
            {
                "role": "user",
                "content": f'''
{user_input}

Respond ONLY in JSON with this exact format:
{{
  "product_name": string,
  "price": number,
  "category": string
}}
'''
            }
        ]
    )

    output = response.choices[0].message.content

    output = output.replace("```json", "").replace("```", "").strip()

    print("\nRaw Output:", output)

    try:
        data = json.loads(output)

        print("\nParsed Output:")
        print("Product Name:", data["product_name"])
        print("Price:", data["price"])
        print("Category:", data["category"])

    except Exception as e:
        print("Error parsing JSON:", e)
