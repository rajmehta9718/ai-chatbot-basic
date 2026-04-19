import time
from fastapi import Request
import asyncio
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from openai import OpenAI
from dotenv import load_dotenv
import faiss
import numpy as np
import logging

load_dotenv()

# -------------------------------
# Caches
# -------------------------------
embedding_cache = {}
response_cache = {}

# -------------------------------
# Config
# -------------------------------
DOCUMENT_FILE = "company.txt"
EMBEDDING_MODEL = "text-embedding-3-small"
CHAT_MODEL = "gpt-4o-mini"
TOP_K = 4
RATE_LIMIT = 5
WINDOW_SECONDS = 60 

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

client_requests = {}
def is_rate_limited(client_ip):
    current_time = time.time()

    if client_ip not in client_requests:
        client_requests[client_ip] = []

    # remove old requests outside window
    client_requests[client_ip] = [
        t for t in client_requests[client_ip]
        if current_time - t < WINDOW_SECONDS
    ]

    if len(client_requests[client_ip]) >= RATE_LIMIT:
        return True

    client_requests[client_ip].append(current_time)
    return False

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

def chunk_text(text, chunk_size=20, overlap=5):
    words = text.split()
    chunks = []

    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")

    step = chunk_size - overlap

    for i in range(0, len(words), step):
        chunk = " ".join(words[i:i + chunk_size])
        if chunk:
            chunks.append(chunk)

    return chunks

# -------------------------------
# Embedding generation
# -------------------------------
def get_embedding(text):
    if text in embedding_cache:
        logger.info("Embedding cache hit")
        return embedding_cache[text]

    try:
        response = client.embeddings.create(
            model=EMBEDDING_MODEL,
            input=text
        )
        embedding = response.data[0].embedding
        embedding_cache[text] = embedding
        return embedding

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
# Chunking
# -------------------------------
def chunk_text(text, chunk_size=20, overlap=5):
    words = text.split()
    chunks = []

    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")

    step = chunk_size - overlap

    for i in range(0, len(words), step):
        chunk = " ".join(words[i:i + chunk_size])
        if chunk:
            chunks.append(chunk)

    return chunks

# -------------------------------
# Startup data
# -------------------------------
raw_documents = load_documents(DOCUMENT_FILE)
documents = []
for doc in raw_documents:
    chunks = chunk_text(doc, chunk_size=20, overlap=5)
    documents.extend(chunks)
index, doc_embeddings = build_faiss_index(documents)


# -------------------------------
# Semantic search using FAISS
# -------------------------------
async def search(query, top_k=TOP_K):
    try:
        logger.info(f"Running search for query: {query}")

        query_embedding = await get_embedding_async(query)
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
- If the context contains only partial information, mention only what is supported

Context:
{context}

Question:
{query}

Return a direct answer based only on the context.
"""

    response = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=[
            {"role": "system", "content": "You are a careful, grounded company policy assistant."},
            {"role": "user", "content": prompt}
        ]
    )

    return response.choices[0].message.content

async def get_embedding_async(text):
    return await asyncio.to_thread(get_embedding, text)


async def generate_answer_async(query, context):
    return await asyncio.to_thread(generate_answer, query, context)

# -------------------------------
# Full RAG pipeline
# -------------------------------
async def rag_answer(query):
    if query in response_cache:
        logger.info("Response cache hit")
        return response_cache[query]

    try:
        top_docs = await search(query, top_k=TOP_K)
        context = build_context(top_docs)
        answer = await generate_answer_async(query, context)

        result = {
            "query": query,
            "answer": answer,
            "retrieved_documents": [
                {"document": doc, "score": score} for doc, score in top_docs
            ]
        }

        response_cache[query] = result
        return result

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
async def ask_question(request: AskRequest, req: Request):
    client_ip = req.client.host

    if is_rate_limited(client_ip):
        logger.warning(f"Rate limit exceeded for {client_ip}")
        raise HTTPException(status_code=429, detail="Rate limit exceeded")

    try:
        logger.info(f"Received /ask request: {request.query}")
        return await rag_answer(request.query)

    except Exception as e:
        logger.error(f"/ask endpoint failed: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")
