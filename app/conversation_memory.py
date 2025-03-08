import time
from typing import List, Dict, Optional, Any, Tuple
from pymongo import MongoClient
import os
import uuid
from dotenv import load_dotenv

load_dotenv()


class ConversationMemory:
    def __init__(self):
        # Initialize MongoDB connection
        mongo_uri = os.getenv("MONGO_URI", "mongodb://localhost:27017")
        self.client = MongoClient(mongo_uri)
        self.db = self.client["litterbox"]
        self.conversations = self.db["conversations"]

        # Indexes for better performance
        self.conversations.create_index("conversation_id")
        self.conversations.create_index("student_id")

    def add_exchange(self,
                     conversation_id: str,
                     student_message: str,
                     assistant_message: str,
                     topic: str,
                     scaffolding_level: int,
                     student_id: Optional[str] = None,
                     metadata: Optional[Dict[str, Any]] = None):
        """Add a new exchange to the conversation"""

        # Create exchange document
        exchange = {
            "timestamp": time.time(),
            "student_message": student_message,
            "assistant_message": assistant_message,
            "topic": topic,
            "scaffolding_level": scaffolding_level
        }

        # Add metadata if provided
        if metadata:
            exchange["metadata"] = metadata

        # Update or create conversation document
        self.conversations.update_one(
            {"conversation_id": conversation_id},
            {
                "$push": {"exchanges": exchange},
                "$set": {
                    "last_updated": time.time(),
                    "student_id": student_id,
                    "last_topic": topic,
                    "current_scaffolding_level": scaffolding_level
                },
                "$setOnInsert": {"created_at": time.time()}
            },
            upsert=True
        )

        return conversation_id

    def get_conversation(self, conversation_id: str, student_id: Optional[str] = None) -> Tuple[str, List[Dict]]:
        """
        Retrieve conversation history or create a new conversation
        Returns: (conversation_id, exchanges)
        """
        # If conversation_id is "new", create a new conversation
        if conversation_id == "new":
            new_id = str(uuid.uuid4())
            # Initialize new conversation with empty exchanges
            self.conversations.insert_one({
                "conversation_id": new_id,
                "student_id": student_id,
                "created_at": time.time(),
                "last_updated": time.time(),
                "exchanges": [],
                "current_scaffolding_level": 1  # Default scaffolding level
            })
            return new_id, []

        # Otherwise, retrieve existing conversation
        conversation = self.conversations.find_one({"conversation_id": conversation_id})

        if conversation and "exchanges" in conversation:
            return conversation_id, conversation["exchanges"]

        # If conversation not found but ID was provided, create a new one with that ID
        self.conversations.insert_one({
            "conversation_id": conversation_id,
            "student_id": student_id,
            "created_at": time.time(),
            "last_updated": time.time(),
            "exchanges": [],
            "current_scaffolding_level": 1  # Default scaffolding level
        })
        return conversation_id, []

    def get_current_scaffolding_level(self, conversation_id: str) -> int:
        """Get the current scaffolding level for a conversation"""
        conversation = self.conversations.find_one(
            {"conversation_id": conversation_id},
            {"current_scaffolding_level": 1}
        )

        if conversation and "current_scaffolding_level" in conversation:
            return conversation["current_scaffolding_level"]

        # Default to level 1 if not found
        return 1

    def get_student_conversations(self, student_id: str) -> List[Dict]:
        """Get all conversations for a student"""
        cursor = self.conversations.find({"student_id": student_id})
        return list(cursor)

    def format_conversation_history(self, exchanges: List[Dict], max_exchanges: int = 5) -> str:
        """Format conversation history for context inclusion"""
        if not exchanges:
            return ""

        # Get the most recent exchanges (limited by max_exchanges)
        recent_exchanges = exchanges[-max_exchanges:] if len(exchanges) > max_exchanges else exchanges

        formatted_history = "Previous conversation:\n"
        for exchange in recent_exchanges:
            formatted_history += f"Student: {exchange['student_message']}\n"
            formatted_history += f"Assistant: {exchange['assistant_message']}\n\n"

        return formatted_history