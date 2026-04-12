from openai import OpenAI
from dotenv import load_dotenv
import json

load_dotenv()
client = OpenAI()

def get_product_info(user_input):
    try:
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
  "product_name": "string",
  "price": 0,
  "category": "string"
}}
'''
                }
            ]
        )

        output = response.choices[0].message.content
        output = output.replace("```json", "").replace("```", "").strip()

        data = json.loads(output)
        return data

    except json.JSONDecodeError:
        return {"error": "Could not parse JSON response"}

    except Exception as e:
        return {"error": str(e)}

while True:
    user_input = input("\nYou: ")

    if user_input.lower() == "exit":
        print("Goodbye!")
        break

    result = get_product_info(user_input)
    print("\nResult:", result)
