from openai import OpenAI
from dotenv import load_dotenv
import json

load_dotenv()
client = OpenAI()

def validate_product_data(data):
    required_keys = ["product_name", "price", "category"]

    for key in required_keys:
        if key not in data:
            return False, f"Missing key: {key}"

    if not isinstance(data["product_name"], str):
        return False, "product_name must be a string"

    if not isinstance(data["price"], (int, float)):
        return False, "price must be a number"

    if not isinstance(data["category"], str):
        return False, "category must be a string"

    return True, "Valid"

def get_product_info(user_input):
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": """
You are a strict backend API.

Rules:
- Always return ONLY valid JSON
- Do not include explanations
- Do not include extra text
- Do not include markdown or code blocks
- Follow the schema exactly
- If unsure, still return valid JSON
"""
                },
                {
                    "role": "user",
                    "content": f"""
User request: {user_input}

Return ONLY valid JSON.

Rules:
- product_name must be a string
- price must be a number
- category must be a string
- if exact information is unknown, use "unknown" for strings and 0 for price

Schema:
{{
  "product_name": "string",
  "price": 0,
  "category": "string"
}}
"""
                }
            ]
        )

        output = response.choices[0].message.content
        output = output.replace("```json", "").replace("```", "").strip()

        data = json.loads(output)

        is_valid, message = validate_product_data(data)
        if not is_valid:
            return {"error": message, "raw": data}

        return data

    except json.JSONDecodeError:
        return {"error": "Could not parse JSON response"}

    except Exception as e:
        return {"error": str(e)}

print("=" * 40)
print("       AI Product Assistant")
print("Type 'exit' to quit")
print("=" * 40)

while True:
    user_input = input("\nYou: ").strip()

    if user_input.lower() == "exit":
        print("\nGoodbye!")
        break

    if not user_input:
        print("Please enter something.")
        continue

    result = get_product_info(user_input)

    print("\nAI Response:")
    print("-" * 40)

    if "error" in result:
        print("Error:", result["error"])
    else:
        print("Product Name:", result["product_name"])
        print("Price:", result["price"])
        print("Category:", result["category"])

    print("-" * 40)
