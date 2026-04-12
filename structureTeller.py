from openai import OpenAI
from dotenv import load_dotenv
import json
import time

load_dotenv()
client = OpenAI()

def validate_products_data(data):

    if not isinstance(data, dict):
       return False, f"Item must be an object"

    required_keys = ["product_shape", "product_volume", "product_area"]
    for key in required_keys:
       if key not in data:
          return False, f"Missing key '{key}' in item"

        if not isinstance(data["product_shape"], str):
       return False, f"product_shape must be a string in item"
    
    if not isinstance(data["product_area"], (int, float)):
       return False, f"area must be a number in item"
    
    if not isinstance(data["product_volume"], (int, float)):
       return False, f"category must be a string in item"
          
    return True, "Valid"
def get_structure_info(user_input, max_retries=2):
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
"""
                                          },
                    {
                        "role": "user",
                        "content": f"""
Return the shape, area and volume of {user_input}

Return ONLY valid JSON.

Rules:
- product_shape must be a string
- product_area must be a number
- product_volume must be a number
- if exact information is unknown, use "unknown" for strings and 0 for numbers
                        
Schema:
{
    {
      "product_shape": "string",
      "product_area": 0,
      "product_volume": 0
    }
}
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
print("      AI Structure Teller")
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

    result = get_structure_info(user_input)
    print("\nAI Response:")
    print("-" * 45)
        
    if "error" in result:
        print("Error:", result["error"])
    else:
        print("  Shape:", result["product_shape"])
        print("  Area:", result["product_area"])
        print("  Volume:", result["product_volume"])
        print()
    
    print("-" * 45)
