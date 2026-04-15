from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
client = OpenAI()

text = "AI is transforming the world"

response = client.embeddings.create(
    model="text-embedding-3-small",
    input=text
)

embedding = response.data[0].embedding

print("Embedding length:", len(embedding))
print("First 10 values:", embedding[:10])
