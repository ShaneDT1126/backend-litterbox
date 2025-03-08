from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
import os
import time
from dotenv import load_dotenv
from typing import Optional

from .models import QueryRequest, QueryResponse
from .rag_engine import RAGEngine
from .conversation_memory import ConversationMemory
from .utils import format_response_for_teams

# Load environment variables
load_dotenv()

# Initialize FastAPI app
app = FastAPI(
    title="Litterbox API",
    description="Backend API for Litterbox, an educational chatbot for Computer Organization Architecture",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust this in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize RAG engine
rag_engine = RAGEngine()


@app.get("/")
async def root():
    """Root endpoint to verify API is running"""
    return {"message": "Welcome to Litterbox API", "status": "operational"}


@app.post("/query", response_model=QueryResponse)
async def process_query(request: QueryRequest):
    """Process a student query using RAG and return a scaffolded response"""
    try:
        # Process query with RAG engine
        response = rag_engine.process_query(
            query=request.query,
            conversation_id=request.conversation_id,
            student_id=request.student_id
        )

        # Format response for Teams display if needed
        formatted_response = format_response_for_teams(response["response"])

        # Return response
        return QueryResponse(
            response=formatted_response,
            sources=response["sources"],
            scaffolding_level=response["scaffolding_level"],
            topic=response["topic"],
            conversation_id=response["conversation_id"]
        )
    except Exception as e:
        import traceback
        print(f"Error processing query: {e}")
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": time.time(),
        "version": "1.0.0"
    }


@app.get("/conversations/{student_id}")
async def get_student_conversations(student_id: str):
    """Get all conversations for a student"""
    try:
        # Use the conversation memory from the RAG engine
        conversations = rag_engine.conversation_memory.get_student_conversations(student_id)
        return {"student_id": student_id, "conversations": conversations}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))