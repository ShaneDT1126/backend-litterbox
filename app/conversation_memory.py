import time
from typing import List, Dict, Optional, Any
from pymongo import MongoClient
import os
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

    def get_conversation(self, conversation_id: str) -> List[Dict]:
        """Retrieve conversation history"""
        conversation = self.conversations.find_one({"conversation_id": conversation_id})

        if conversation and "exchanges" in conversation:
            return conversation["exchanges"]

        return []

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