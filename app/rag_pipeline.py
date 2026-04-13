from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

import os
import time
from openai import OpenAI
from dotenv import load_dotenv

# -----------------------------
# Load environment variables
# -----------------------------
load_dotenv()
client = OpenAI()   # reads API key from .env

# -----------------------------
# Load and prepare data (runs once)
# -----------------------------
loader = TextLoader("app/sample.txt")
documents = loader.load()

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=100,
    chunk_overlap=20
)

chunks = text_splitter.split_documents(documents)
chunk_texts = [chunk.page_content for chunk in chunks]

# -----------------------------
# Create embeddings
# -----------------------------
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
embeddings = embedding_model.encode(chunk_texts)

# -----------------------------
# Store in FAISS
# -----------------------------
dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(np.array(embeddings))


# -----------------------------
# MAIN FUNCTION
# -----------------------------
def get_answer(query: str) -> dict:
    start_time = time.time()

    # Convert query to embedding
    query_vector = embedding_model.encode([query])

    # Retrieve top-k chunks
    k = 2
    distances, indices = index.search(np.array(query_vector), k)
    retrieved_chunks = [chunk_texts[i] for i in indices[0]]

    context = " ".join(retrieved_chunks)

    try:
        # -----------------------------
        # OpenAI LLM call (IMPROVED PROMPT)
        # -----------------------------
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {
                    "role": "user",
                    "content": f"""
You are a helpful AI assistant.

Answer clearly in 1-2 sentences.
Be precise and avoid repetition.
Use only the given context.

Context:
{context}

Question: {query}
"""
                }
            ],
            temperature=0.2,
            max_tokens=100
        )

        final_answer = response.choices[0].message.content.strip()

    except Exception as e:
        print("LLM ERROR:", str(e))
        final_answer = "Error generating response from LLM"

    latency = (time.time() - start_time) * 1000

    # -----------------------------
    # FINAL RETURN
    # -----------------------------
    return {
        "answer": final_answer,
        "sources": retrieved_chunks,
        "latency": latency
    }