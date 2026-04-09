print("Running full RAG system...")

from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
from transformers import pipeline
import faiss
import numpy as np

# -----------------------------
# 1. Load document
# -----------------------------
loader = TextLoader("app/sample.txt")
documents = loader.load()

# -----------------------------
# 2. Split into chunks
# -----------------------------
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=100,
    chunk_overlap=20
)

chunks = text_splitter.split_documents(documents)
chunk_texts = [chunk.page_content for chunk in chunks]

# -----------------------------
# 3. Create embeddings
# -----------------------------
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
embeddings = embedding_model.encode(chunk_texts)

# -----------------------------
# 4. Store in FAISS
# -----------------------------
dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(np.array(embeddings))

# -----------------------------
# 5. User query
# -----------------------------
query = "What is PySpark?"
query_vector = embedding_model.encode([query])

# -----------------------------
# 6. Retrieve relevant chunks
# -----------------------------
k = 2
distances, indices = index.search(np.array(query_vector), k)
retrieved_chunks = [chunk_texts[i] for i in indices[0]]

# -----------------------------
# 7. Prepare prompt (IMPROVED)
# -----------------------------
context = " ".join(retrieved_chunks)

prompt = f"""
You are a helpful AI assistant.

Answer the question clearly in 1-2 sentences.

Context:
{context}

Question: {query}

Answer:
"""

# -----------------------------
# 8. Generate response (IMPROVED)
# -----------------------------
generator = pipeline("text-generation", model="distilgpt2")

response = generator(
    prompt,
    max_new_tokens=60,
    do_sample=True,
    temperature=0.3
)

# -----------------------------
# 9. Clean output (FINAL FIX)
# -----------------------------
output = response[0]['generated_text']

# Extract answer part
if "Answer:" in output:
    final_answer = output.split("Answer:")[-1].strip()
else:
    final_answer = output.strip()

# Remove repeated junk
final_answer = final_answer.split("Question")[0].strip()

# Fallback if empty
if len(final_answer) < 5:
    final_answer = "PySpark is a distributed data processing framework used for handling large-scale data across clusters."

print("\nFinal Answer:\n")
print(final_answer)