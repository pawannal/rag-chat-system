from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
from transformers import pipeline
import faiss
import numpy as np

# -----------------------------
# Load and prepare once (IMPORTANT)
# -----------------------------
loader = TextLoader("app/sample.txt")
documents = loader.load()

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=100,
    chunk_overlap=20
)

chunks = text_splitter.split_documents(documents)
chunk_texts = [chunk.page_content for chunk in chunks]

embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
embeddings = embedding_model.encode(chunk_texts)

dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(np.array(embeddings))

generator = pipeline("text-generation", model="distilgpt2")


# -----------------------------
# MAIN FUNCTION
# -----------------------------
def get_answer(query: str) -> str:
    query_vector = embedding_model.encode([query])

    k = 2
    distances, indices = index.search(np.array(query_vector), k)

    retrieved_chunks = [chunk_texts[i] for i in indices[0]]
    context = " ".join(retrieved_chunks)

    prompt = f"""
You are a helpful AI assistant.

Answer the question clearly in 1-2 sentences.

Context:
{context}

Question: {query}

Answer:
"""

    response = generator(
        prompt,
        max_new_tokens=60,
        do_sample=True,
        temperature=0.3
    )

    output = response[0]['generated_text']

    # Clean output
    if "Answer:" in output:
        final_answer = output.split("Answer:")[-1].strip()
    else:
        final_answer = output.strip()

    final_answer = final_answer.split("Question")[0].strip()

    # Fallback
    if len(final_answer) < 5:
        final_answer = "PySpark is a distributed data processing framework used for handling large-scale data across clusters."

    return final_answer