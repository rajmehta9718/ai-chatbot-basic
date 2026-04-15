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
# Step 3: Real-world dataset
# Company knowledge base
# -------------------------------
documents = [
    "Employees are allowed to work from home up to 2 days per week.",
    "Health insurance includes medical, dental, and vision coverage.",
    "Employees receive 15 days of paid vacation annually.",
    "The company provides free lunch on Fridays.",
    "Performance reviews are conducted every 6 months."
]


# -------------------------------
# Step 4: Precompute embeddings
# -------------------------------
doc_embeddings = [get_embedding(doc) for doc in documents]


# -------------------------------
# Step 5: Semantic Search
# -------------------------------
def search(query, top_k=3):
    query_embedding = get_embedding(query)
    similarities = []

    for i, doc_embedding in enumerate(doc_embeddings):
        sim = cosine_similarity(query_embedding, doc_embedding)
        similarities.append((documents[i], sim))

    similarities.sort(key=lambda x: x[1], reverse=True)
    return similarities[:top_k]


# -------------------------------
# Step 6: Multi-document RAG
# -------------------------------
def rag_answer(query):
    top_docs = search(query, top_k=3)

    context = "\n\n".join(
        [f"Document {i+1}: {doc}" for i, (doc, _) in enumerate(top_docs)]
    )

    prompt = f"""
You are an AI assistant.

Use ONLY the provided context to answer the question.
If the answer is not in the context, say "I don't know".

Context:
{context}

Question:
{query}

Answer clearly and concisely.
"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a helpful company policy assistant."},
            {"role": "user", "content": prompt}
        ]
    )

    return response.choices[0].message.content, top_docs


# -------------------------------
# Step 7: Main
# -------------------------------
if __name__ == "__main__":
    query = input("Ask a company policy question: ")

    answer, top_docs = rag_answer(query)

    print("\nTop Retrieved Documents:")
    print("-" * 50)
    for i, (doc, score) in enumerate(top_docs, start=1):
        print(f"{i}. {doc} (score: {score:.4f})")

    print("\nFinal Answer:")
    print("-" * 50)
    print(answer)
    print("-" * 50)


"""
output:
Ask a company policy question: tell about the wfh and vacation 

Top Retrieved Documents:
--------------------------------------------------
1. Employees receive 15 days of paid vacation annually. (score: 0.4158)
2. Employees are allowed to work from home up to 2 days per week. (score: 0.4079)
3. The company provides free lunch on Fridays. (score: 0.2524)

Final Answer:
--------------------------------------------------
Employees are allowed to work from home up to 2 days per week and receive 15 days of paid vacation annually.
--------------------------------------------------
(ai-env) raj@Rajs-MacBook-Pro ~ % tell about the food benefits
zsh: command not found: tell
(ai-env) raj@Rajs-MacBook-Pro ~ % python multidocrag.py       
Ask a company policy question: tell about the food benefits

Top Retrieved Documents:
--------------------------------------------------
1. The company provides free lunch on Fridays. (score: 0.3015)
2. Health insurance includes medical, dental, and vision coverage. (score: 0.1507)
3. Performance reviews are conducted every 6 months. (score: 0.0799)

Final Answer:
--------------------------------------------------
The company provides free lunch on Fridays.
--------------------------------------------------
(ai-env) raj@Rajs-MacBook-Pro ~ % tell if thry give chocolates or not and what is the median salary of employees
zsh: command not found: tell
(ai-env) raj@Rajs-MacBook-Pro ~ % python multidocrag.py
Ask a company policy question: tell if thry give chocolates or not and what is the median salary of employees

Top Retrieved Documents:
--------------------------------------------------
1. The company provides free lunch on Fridays. (score: 0.2997)
2. Performance reviews are conducted every 6 months. (score: 0.2067)
3. Employees receive 15 days of paid vacation annually. (score: 0.2049)

Final Answer:
--------------------------------------------------
I don't know.
--------------------------------------------------
"""
