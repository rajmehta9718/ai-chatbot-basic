from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from openai import OpenAI
from dotenv import load_dotenv
import faiss
import numpy as np
import logging

load_dotenv()

# -------------------------------
# Config
# -------------------------------
DOCUMENT_FILE = "company.txt"
EMBEDDING_MODEL = "text-embedding-3-small"
CHAT_MODEL = "gpt-4o-mini"
TOP_K = 3

# -------------------------------
# Logging
# -------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# -------------------------------
# App / Client
# -------------------------------
client = OpenAI()
app = FastAPI()


# -------------------------------
# Request / Response Models
# -------------------------------
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
    try:
        with open(filename, "r") as file:
            lines = file.readlines()

        docs = [line.strip() for line in lines if line.strip()]
        logger.info(f"Loaded {len(docs)} documents from {filename}")
        return docs

    except FileNotFoundError:
        logger.error(f"Document file not found: {filename}")
        raise

    except Exception as e:
        logger.error(f"Unexpected error loading documents: {e}")
        raise


# -------------------------------
# Embedding generation
# -------------------------------
def get_embedding(text):
    try:
        response = client.embeddings.create(
            model=EMBEDDING_MODEL,
            input=text
        )
        return response.data[0].embedding

    except Exception as e:
        logger.error(f"Embedding generation failed: {e}")
        raise


# -------------------------------
# Build FAISS index
# -------------------------------
def build_faiss_index(docs):
    try:
        logger.info("Generating embeddings for documents...")
        doc_embeddings = [get_embedding(doc) for doc in docs]
        embedding_matrix = np.array(doc_embeddings).astype("float32")

        dimension = embedding_matrix.shape[1]
        index = faiss.IndexFlatL2(dimension)
        index.add(embedding_matrix)

        logger.info(f"FAISS index created with {len(docs)} documents")
        return index, doc_embeddings

    except Exception as e:
        logger.error(f"Failed to build FAISS index: {e}")
        raise


# -------------------------------
# Startup data
# -------------------------------
documents = load_documents(DOCUMENT_FILE)
index, doc_embeddings = build_faiss_index(documents)


# -------------------------------
# Semantic search using FAISS
# -------------------------------
def search(query, top_k=TOP_K):
    try:
        logger.info(f"Running search for query: {query}")

        query_embedding = get_embedding(query)
        query_vector = np.array([query_embedding]).astype("float32")

        distances, indices = index.search(query_vector, top_k)

        results = []
        for i, idx in enumerate(indices[0]):
            results.append((documents[idx], float(distances[0][i])))

        logger.info(f"Retrieved {len(results)} documents")
        return results

    except Exception as e:
        logger.error(f"Search failed: {e}")
        raise


# -------------------------------
# Build context from retrieved docs
# -------------------------------
def build_context(top_docs):
    return "\n\n".join(
        [f"Source {i+1}: {doc}" for i, (doc, _) in enumerate(top_docs)]
    )


# -------------------------------
# Generate grounded answer
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

    try:
        response = client.chat.completions.create(
            model=CHAT_MODEL,
            messages=[
                {"role": "system", "content": "You are a careful, grounded company policy assistant."},
                {"role": "user", "content": prompt}
            ]
        )

        return response.choices[0].message.content

    except Exception as e:
        logger.error(f"Answer generation failed: {e}")
        raise


# -------------------------------
# Full RAG pipeline
# -------------------------------
def rag_answer(query):
    try:
        top_docs = search(query, top_k=TOP_K)
        context = build_context(top_docs)
        answer = generate_answer(query, context)

        return {
            "query": query,
            "answer": answer,
            "retrieved_documents": [
                {"document": doc, "score": score} for doc, score in top_docs
            ]
        }

    except Exception as e:
        logger.error(f"RAG pipeline failed: {e}")
        raise


# -------------------------------
# API endpoints
# -------------------------------
@app.get("/health")
def health_check():
    return {"status": "ok"}


@app.post("/ask", response_model=AskResponse)
def ask_question(request: AskRequest):
    try:
        logger.info(f"Received /ask request: {request.query}")
        return rag_answer(request.query)

    except Exception as e:
        logger.error(f"/ask endpoint failed: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")
