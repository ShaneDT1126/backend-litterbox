from pydantic import BaseModel
from typing import List, Optional, Dict, Any

class QueryRequest(BaseModel):
    query: str
    conversation_id: str
    student_id: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None

class Source(BaseModel):
    content: str
    topic: str
    confidence: float
    metadata: Optional[Dict[str, Any]] = None

class QueryResponse(BaseModel):
    response: str
    sources: List[Source]
    scaffolding_level: int
    topic: str
    conversation_id: str

class ConversationExchange(BaseModel):
    student_message: str
    assistant_message: str
    timestamp: float
    metadata: Optional[Dict[str, Any]] = None