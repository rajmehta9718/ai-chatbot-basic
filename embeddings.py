from openai import OpenAI
from dotenv import load_dotenv
import math

load_dotenv()
client = OpenAI()


def get_embedding(text):
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=text
    )
    return response.data[0].embedding


def cosine_similarity(vec1, vec2):
    dot_product = sum(a*b for a, b in zip(vec1, vec2))
    
    norm1 = math.sqrt(sum(a*a for a in vec1))
    norm2 = math.sqrt(sum(b*b for b in vec2))
    
    return dot_product / (norm1 * norm2)

text1 = "AI is amazing"
text2 = "Artificial Intelligence is amazing"
text3 = "Bananas are yellow"

emb1 = get_embedding(text1)
emb2 = get_embedding(text2)
emb3 = get_embedding(text3)

sim1 = cosine_similarity(emb1, emb2)
sim2 = cosine_similarity(emb1, emb3)

print("Similarity (AI vs AI alt):", sim1)
print("Similarity (AI vs Banana):", sim2)
