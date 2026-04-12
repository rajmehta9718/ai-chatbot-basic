from openai import OpenAI
from dotenv import load_dotenv
import json
import time

load_dotenv()
client = OpenAI()

def validate_products_data(data):
    if "products" not in data:
        return False, "Missing key: products"

    if not isinstance(data["products"], list):
        return False, "products must be a list"

    if len(data["products"]) != 3:
        return False, "products must contain exactly 3 items"

    for i, product in enumerate(data["products"]):
        if not isinstance(product, dict):
            return False, f"Item {i} must be an object"

        required_keys = ["product_name", "price", "category"]
        for key in required_keys:
            if key not in product:
                return False, f"Missing key '{key}' in item {i}"

        if not isinstance(product["product_name"], str):
            return False, f"product_name must be a string in item {i}"

        if not isinstance(product["price"], (int, float)):
            return False, f"price must be a number in item {i}"

        if not isinstance(product["category"], str):
            return False, f"category must be a string in item {i}"

    return True, "Valid"

def get_products_info(user_input, max_retries=2):
    for attempt in range(max_retries + 1):
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
- Return exactly 3 products
"""
                    },
                    {
                        "role": "user",
                        "content": f"""
User request: {user_input}

Return ONLY valid JSON.

Rules:
- products must be a list of exactly 3 items
- product_name must be a string
- price must be a number
- category must be a string
- if exact information is unknown, use "unknown" for strings and 0 for price

Schema:
{{
  "products": [
    {{
      "product_name": "string",
      "price": 0,
      "category": "string"
    }}
  ]
}}
"""
                    }
                ]
            )

            output = response.choices[0].message.content
            output = output.replace("```json", "").replace("```", "").strip()

            data = json.loads(output)

            is_valid, message = validate_products_data(data)
            if not is_valid:
                raise ValueError(message)

            return data

        except Exception as e:
            if attempt < max_retries:
                print(f"\nRetrying... attempt {attempt + 1} failed: {e}")
                time.sleep(1)
            else:
                return {"error": str(e)}

print("=" * 45)
print("      AI Product Recommender")
print("Type 'exit' to quit")
print("=" * 45)

while True:
    user_input = input("\nYou: ").strip()

    if user_input.lower() == "exit":
        print("\nGoodbye!")
        break

    if not user_input:
        print("Please enter something.")
        continue

    result = get_products_info(user_input)

    print("\nAI Response:")
    print("-" * 45)

    if "error" in result:
        print("Error:", result["error"])
    else:
        for i, product in enumerate(result["products"], start=1):
            print(f"Product {i}:")
            print("  Product Name:", product["product_name"])
            print("  Price:", product["price"])
            print("  Category:", product["category"])
            print()

    print("-" * 45)
