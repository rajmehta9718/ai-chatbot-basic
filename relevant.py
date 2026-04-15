from openai import OpenAI
from dotenv import load_dotenv
import math

load_dotenv()
client = OpenAI()


# -------------------------------
# Step 1: Get Embedding
# -------------------------------
def get_embedding(text):
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=text
    )
    return response.data[0].embedding


# -------------------------------
# Step 2: Cosine Similarity
# -------------------------------
def cosine_similarity(vec1, vec2):
    dot_product = sum(a * b for a, b in zip(vec1, vec2))

    norm1 = math.sqrt(sum(a * a for a in vec1))
    norm2 = math.sqrt(sum(b * b for b in vec2))

    return dot_product / (norm1 * norm2)


# -------------------------------
# Step 3: Dataset
# -------------------------------
documents = [
    "AI is transforming technology",
    "Machine learning is a subset of AI",
    "Bananas are yellow and rich in potassium",
    "Cars are used for transportation",
    "Deep learning models require data"
]


# -------------------------------
# Step 4: Precompute embeddings
# -------------------------------
doc_embeddings = [get_embedding(doc) for doc in documents]


# -------------------------------
# Step 5: Search Function
# -------------------------------
def search(query):
    query_embedding = get_embedding(query)

    similarities = []

    for i, doc_embedding in enumerate(doc_embeddings):
        sim = cosine_similarity(query_embedding, doc_embedding)
        similarities.append((documents[i], sim))

    # Sort by similarity (highest first)
    similarities.sort(key=lambda x: x[1], reverse=True)

    return similarities


# -------------------------------
# Step 6: Run Search
# -------------------------------
if __name__ == "__main__":
    query = input("Enter your query: ")

    results = search(query)

    print("\nTop Results:")
    print("-" * 40)

    for doc, score in results:
        print(f"{doc} → {score:.4f}")

    print("-" * 40)

"""
Enter your query: Machine

Top Results:
----------------------------------------
Machine learning is a subset of AI → 0.2762
AI is transforming technology → 0.2665
Deep learning models require data → 0.1974
Cars are used for transportation → 0.1957
Bananas are yellow and rich in potassium → 0.0487
----------------------------------------
"""
