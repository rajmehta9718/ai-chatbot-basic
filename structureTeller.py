from openai import OpenAI
from dotenv import load_dotenv
import json
import time

load_dotenv()
client = OpenAI()


def call_model(messages):
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages
    )
    return response.choices[0].message.content


def clean_output(output):
    return output.replace("```json", "").replace("```", "").strip()


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


def validate_ideas_data(data):
    if "ideas" not in data:
        return False, "Missing key: ideas"

    if not isinstance(data["ideas"], list):
        return False, "ideas must be a list"

    if len(data["ideas"]) != 3:
        return False, "ideas must contain exactly 3 items"

    for i, idea in enumerate(data["ideas"]):
        if not isinstance(idea, dict):
            return False, f"Item {i} must be an object"

        required_keys = ["idea_name", "description", "target_market"]
        for key in required_keys:
            if key not in idea:
                return False, f"Missing key '{key}' in item {i}"

        if not isinstance(idea["idea_name"], str):
            return False, f"idea_name must be a string in item {i}"

        if not isinstance(idea["description"], str):
            return False, f"description must be a string in item {i}"

        if not isinstance(idea["target_market"], str):
            return False, f"target_market must be a string in item {i}"

    return True, "Valid"


def get_products_info(user_input, max_retries=2):
    system_prompt = """
You are a strict backend API.

Rules:
- Always return ONLY valid JSON
- Do not include explanations
- Do not include extra text
- Do not include markdown or code blocks
- Follow the schema exactly
- Return exactly 3 products
"""

    user_prompt = f"""
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

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]

    for attempt in range(max_retries + 1):
        try:
            output = call_model(messages)
            output = clean_output(output)
            data = json.loads(output)

            is_valid, message = validate_products_data(data)
            if not is_valid:
                raise ValueError(message)

            return data

        except Exception as e:
            if attempt < max_retries:
                time.sleep(1)
            else:
                return {"error": str(e)}


def get_startup_ideas(user_input, max_retries=2):
    system_prompt = """
You are a strict backend API.

Rules:
- Always return ONLY valid JSON
- Do not include explanations
- Do not include extra text
- Do not include markdown or code blocks
- Follow the schema exactly
- Return exactly 3 startup ideas
"""

    user_prompt = f"""
User request: {user_input}

Return ONLY valid JSON.

Rules:
- ideas must be a list of exactly 3 items
- idea_name must be a string
- description must be a string
- target_market must be a string
- if exact information is unknown, use "unknown"

Schema:
{{
  "ideas": [
    {{
      "idea_name": "string",
      "description": "string",
      "target_market": "string"
    }}
  ]
}}
"""

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]

    for attempt in range(max_retries + 1):
        try:
            output = call_model(messages)
            output = clean_output(output)
            data = json.loads(output)

            is_valid, message = validate_ideas_data(data)
            if not is_valid:
                raise ValueError(message)

            return data

        except Exception as e:
            if attempt < max_retries:
                time.sleep(1)
            else:
                return {"error": str(e)}


def print_products(result):
    for i, product in enumerate(result["products"], start=1):
        print(f"Product {i}:")
        print("  Product Name:", product["product_name"])
        print("  Price:", product["price"])
        print("  Category:", product["category"])
        print()


def print_ideas(result):
    for i, idea in enumerate(result["ideas"], start=1):
        print(f"Idea {i}:")
        print("  Name:", idea["idea_name"])
        print("  Description:", idea["description"])
        print("  Market:", idea["target_market"])
        print()


def main():
    print("=" * 45)
    print("         AI Assistant")
    print("1. Product Recommender")
    print("2. Startup Idea Generator")
    print("Type 'exit' anytime to quit")
    print("=" * 45)

    mode = input("Enter 1 or 2: ").strip()

    while True:
        user_input = input("\nYou: ").strip()

        if user_input.lower() == "exit":
            print("\nGoodbye!")
            break

        if not user_input:
            print("Please enter something.")
            continue

        if mode == "1":
            result = get_products_info(user_input)
        elif mode == "2":
            result = get_startup_ideas(user_input)
        else:
            print("Invalid mode selected.")
            break

        print("\nAI Response:")
        print("-" * 45)

        if "error" in result:
            print("Error:", result["error"])
        else:
            if mode == "1":
                print_products(result)
            else:
                print_ideas(result)

        print("-" * 45)


if __name__ == "__main__":
    main()
