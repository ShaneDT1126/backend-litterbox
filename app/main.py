from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
import os
import time
from dotenv import load_dotenv
from typing import Optional

from .models import QueryRequest, QueryResponse
from .rag_engine import RAGEngine
from .conversation_memory import ConversationMemory
from .utils import generate_conversation_id, format_response_for_teams

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

# Initialize RAG engine and conversation memory
rag_engine = RAGEngine()
conversation_memory = ConversationMemory()

@app.get("/")
async def root():
    """Root endpoint to verify API is running"""
    return {"message": "Welcome to Litterbox API", "status": "operational"}


@app.post("/query", response_model=QueryResponse)
async def process_query(request: QueryRequest):
    """Process a student query using RAG and return a scaffolded response"""
    try:
        # Generate conversation ID if not provided
        conversation_id = request.conversation_id
        if not conversation_id or conversation_id == "new":
            conversation_id = generate_conversation_id()

        # Get conversation history
        conversation_history = conversation_memory.get_conversation(conversation_id)

        # Get current scaffolding level
        current_level = conversation_memory.get_current_scaffolding_level(conversation_id)

        # Process query with RAG multiquery
        response_text, sources, scaffolding_level, detected_topic = rag_engine.process_query(
            request.query,
            conversation_id,
            conversation_history,
            current_level,
            request.metadata
        )

        # Format response for Teams display
        formatted_response = format_response_for_teams(response_text)

        # Update conversation memory
        conversation_memory.add_exchange(
            conversation_id,
            request.query,
            formatted_response,
            detected_topic,
            scaffolding_level,
            request.student_id,
            request.metadata
        )

        # Return response
        return QueryResponse(
            response=formatted_response,
            sources=sources,
            scaffolding_level=scaffolding_level,
            topic=detected_topic,
            conversation_id=conversation_id
        )
    except Exception as e:
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
        conversations = conversation_memory.get_student_conversations(student_id)
        return {"student_id": student_id, "conversations": conversations}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))