import faiss
import numpy as np
from pydantic import BaseModel
from fastapi import FastAPI
from openai import OpenAI
from dotenv import load_dotenv
import math

load_dotenv()
client = OpenAI()

app = FastAPI()

class AskRequest(BaseModel):
    query: str


class Document(BaseModel):
    document: str
    score: float


class AskResponse(BaseModel):
    query: str
    answer: str
    retrieved_documents: list[Document]

# -------------------------------
# Load documents from file
# -------------------------------
def load_documents(filename):
    with open(filename, "r") as file:
        lines = file.readlines()

    return [line.strip() for line in lines if line.strip()]


# -------------------------------
# Embedding generation
# -------------------------------
def get_embedding(text):
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=text
    )
    return response.data[0].embedding


# -------------------------------
# Cosine similarity
# -------------------------------
def cosine_similarity(vec1, vec2):
    dot_product = sum(a * b for a, b in zip(vec1, vec2))
    norm1 = math.sqrt(sum(a * a for a in vec1))
    norm2 = math.sqrt(sum(b * b for b in vec2))
    return dot_product / (norm1 * norm2)


# -------------------------------
# Load dataset and precompute embeddings
# -------------------------------
documents = load_documents("company.txt")
doc_embeddings = [get_embedding(doc) for doc in documents]
# Convert embeddings to numpy
embedding_matrix = np.array(doc_embeddings).astype("float32")

# Create FAISS index
dimension = embedding_matrix.shape[1]
index = faiss.IndexFlatL2(dimension)

# Add vectors to index
index.add(embedding_matrix)

# -------------------------------
# Semantic search
# -------------------------------
def search(query, top_k=3):
    query_embedding = get_embedding(query)
    query_vector = np.array([query_embedding]).astype("float32")

    distances, indices = index.search(query_vector, top_k)

    results = []
    for i, idx in enumerate(indices[0]):
        results.append((documents[idx], distances[0][i]))

    return results

# -------------------------------
# Build context from retrieved docs
# -------------------------------
def build_context(top_docs):
    return "\n\n".join(
        [f"Document {i+1}: {doc}" for i, (doc, _) in enumerate(top_docs)]
    )


# -------------------------------
# Generate answer from context
# -------------------------------
def generate_answer(query, context):
    prompt = f"""
You are an AI assistant for company policy questions.

Rules:
- Use ONLY the provided context
- Do NOT add outside knowledge
- If the answer is not in the context, say "I don't know based on the provided context"
- Keep the answer concise and professional
- If multiple sources are relevant, combine them clearly

Context:
{context}

Question:
{query}

Return a clear answer based only on the context.
"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a careful, grounded company policy assistant."},
            {"role": "user", "content": prompt}
        ]
    )

    return response.choices[0].message.content


# -------------------------------
# Full RAG pipeline
# -------------------------------
def rag_answer(query):
    top_docs = search(query, top_k=3)
    context = build_context(top_docs)
    answer = generate_answer(query, context)

    return {
        "query": query,
        "answer": answer,
        "retrieved_documents": [
            {"document": doc, "score": score} for doc, score in top_docs
        ]
    }


# -------------------------------
# API endpoints
# -------------------------------
@app.get("/health")
def health_check():
    return {"status": "ok"}


@app.post("/ask", response_model=AskResponse)
def ask_question(request: AskRequest):
    result = rag_answer(request.query)
    return result
