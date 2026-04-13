from fastapi import FastAPI, HTTPException
from app.models import QueryRequest, QueryResponse
from app.rag_pipeline import get_answer
import logging

# -----------------------------
# Logging
# -----------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# -----------------------------
# FastAPI app
# -----------------------------
app = FastAPI(
    title="RAG Chat System API",
    description="Production-ready GenAI API using RAG pipeline",
    version="1.0.0"
)

# -----------------------------
# Health check
# -----------------------------
@app.get("/")
def health_check():
    return {"status": "ok", "message": "RAG API is running"}


# -----------------------------
# Query endpoint
# -----------------------------
@app.post("/query", response_model=QueryResponse)
def query_rag(request: QueryRequest):
    try:
        logger.info(f"Received query: {request.question}")

        if not request.question.strip():
            raise ValueError("Question cannot be empty")

        result = get_answer(request.question)

        return QueryResponse(
            answer=result["answer"],
            source_chunks=result["sources"],
            latency_ms=result["latency"]
        )

    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))

    except Exception as e:
        logger.error(f"Error occurred: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal Server Error")