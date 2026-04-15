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
    "AI is transforming technology.",
    "Machine learning is a subset of artificial intelligence.",
    "Bananas are yellow and rich in potassium.",
    "Cars are used for transportation.",
    "Deep learning models require large amounts of data."
]


# -------------------------------
# Step 4: Precompute embeddings
# -------------------------------
doc_embeddings = [get_embedding(doc) for doc in documents]


# -------------------------------
# Step 5: Semantic Search
# -------------------------------
def search(query, top_k=2):
    query_embedding = get_embedding(query)

    similarities = []

    for i, doc_embedding in enumerate(doc_embeddings):
        sim = cosine_similarity(query_embedding, doc_embedding)
        similarities.append((documents[i], sim))

    similarities.sort(key=lambda x: x[1], reverse=True)

    return similarities[:top_k]


# -------------------------------
# Step 6: RAG Answer
# -------------------------------
def rag_answer(query):
    top_docs = search(query)

    context = "\n".join([doc for doc, _ in top_docs])

    prompt = f"""
Use the following context to answer the question.

Context:
{context}

Question:
{query}

Answer clearly and concisely.
"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ]
    )

    return response.choices[0].message.content


# -------------------------------
# Step 7: Main
# -------------------------------
if __name__ == "__main__":
    query = input("Ask a question: ")

    answer = rag_answer(query)

    print("\nAnswer:")
    print("-" * 40)
    print(answer)
    print("-" * 40)
