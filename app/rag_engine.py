from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.vectorstores import FAISS
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain.prompts import ChatPromptTemplate
from langchain.schema import Document
from typing import List, Dict, Any, Tuple, Optional
import os
import json
import time
from dotenv import load_dotenv

load_dotenv()


class RAGEngine:
    def __init__(self, vector_store_path: str = "faiss_index"):
        # Initialize OpenAI API
        self.api_key = os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY environment variable is not set")

        # Initialize embeddings
        self.embeddings = OpenAIEmbeddings()

        # Load vector store if it exists, otherwise create empty one
        try:
            self.vector_store = FAISS.load_local(vector_store_path, self.embeddings)
            print(f"Loaded vector store from {vector_store_path}")
        except Exception as e:
            print(f"Could not load vector store: {e}")
            print("Creating new vector store")
            # Create an empty vector store
            self.vector_store = FAISS.from_documents([Document(page_content="Litterbox initialization")],
                                                     self.embeddings)

        # Initialize LLM
        self.llm = ChatOpenAI(temperature=0.2)

        # Create multi-query retriever
        self.retriever = MultiQueryRetriever.from_llm(
            retriever=self.vector_store.as_retriever(search_kwargs={"k": 5}),
            llm=self.llm
        )

        # Define topic keywords for detection
        self.topic_keywords = {
            "CPU Architecture": ["cpu", "processor", "alu", "control unit", "registers", "instruction cycle",
                                 "fetch", "decode", "execute", "von neumann"],
            "Cache Memory": ["cache", "memory hierarchy", "hit", "miss", "replacement", "write policy",
                             "direct mapped", "set associative", "fully associative"],
            "Memory Systems": ["ram", "rom", "memory", "dram", "sram", "virtual memory", "paging",
                               "segmentation", "tlb", "page table", "memory management"],
            "Instruction Set Architecture": ["isa", "instruction set", "opcodes", "addressing modes", "cisc", "risc",
                                             "arm", "x86", "mips", "instruction format"],
            "Pipelining": ["pipeline", "stage", "hazard", "forwarding", "stall", "branch prediction",
                           "data hazard", "control hazard", "structural hazard"],
            "I/O Systems": ["input", "output", "i/o", "bus", "interrupt", "dma", "device", "controller",
                            "peripheral", "usb", "pci"],
            "Performance": ["performance", "speedup", "benchmark", "amdahl", "cpi", "mips", "flops",
                            "throughput", "latency", "clock rate"]
        }

        # Define scaffolding templates
        self.scaffolding_templates = {
            1: """You are Litterbox, an educational assistant helping a student who is new to Computer Organization Architecture. 
            Provide structured guidance with clear explanations of basic concepts. Ask simple questions to check understanding. 
            Current topic: {topic}

            Guidelines:
            - Break down complex concepts into simple parts
            - Use analogies to explain difficult concepts
            - Ask basic comprehension questions
            - Provide encouragement and positive reinforcement
            - Never give direct answers to problems
            """,

            2: """You are Litterbox, an educational assistant helping a student with some knowledge of Computer Organization Architecture. 
            Provide hints and partial frameworks rather than complete explanations. Ask probing questions that encourage deeper thinking. 
            Current topic: {topic}

            Guidelines:
            - Provide hints rather than full explanations
            - Ask questions that require application of concepts
            - Encourage the student to make connections between concepts
            - Challenge misconceptions with guiding questions
            - Never give direct answers to problems
            """,

            3: """You are Litterbox, an educational assistant helping a student with good understanding of Computer Organization Architecture. 
            Provide minimal prompting and verification. Ask challenging questions that require application and synthesis of concepts. 
            Current topic: {topic}

            Guidelines:
            - Ask challenging questions that require synthesis of multiple concepts
            - Provide minimal guidance, focusing on verification
            - Encourage the student to evaluate trade-offs and design decisions
            - Prompt for deeper analysis of implications and consequences
            - Never give direct answers to problems
            """
        }

    def detect_topic(self, query: str, retrieved_docs: List[Document] = None) -> str:
        """Detect the topic of the query based on keywords and retrieved documents"""
        query_lower = query.lower()

        # Count keyword matches for each topic
        topic_scores = {topic: 0 for topic in self.topic_keywords}

        # Check query for keywords
        for topic, keywords in self.topic_keywords.items():
            for keyword in keywords:
                if keyword in query_lower:
                    topic_scores[topic] += 1

        # Check retrieved documents for topic metadata
        if retrieved_docs:
            # Count topics in retrieved documents
            doc_topics = {}
            for doc in retrieved_docs:
                if "topic" in doc.metadata:
                    doc_topic = doc.metadata["topic"]
                    if doc_topic in doc_topics:
                        doc_topics[doc_topic] += 1
                    else:
                        doc_topics[doc_topic] = 1

            # Add document topics to scores with higher weight
            for topic, count in doc_topics.items():
                if topic in topic_scores:
                    topic_scores[topic] += count * 2  # Give more weight to document metadata

        # Get topic with highest score
        max_score = 0
        detected_topic = "General Computer Architecture"

        for topic, score in topic_scores.items():
            if score > max_score:
                max_score = score
                detected_topic = topic

        return detected_topic

    def determine_scaffolding_level(self, conversation_history: List[Dict], current_level: int = 1) -> int:
        """Determine appropriate scaffolding level based on conversation history"""
        if not conversation_history:
            return current_level

        # Only consider recent exchanges (last 5)
        recent_exchanges = conversation_history[-5:]

        # Count indicators of understanding
        understanding_indicators = sum(
            1 for exchange in recent_exchanges
            if any(phrase in exchange.get("student_message", "").lower()
                   for phrase in ["i understand", "that makes sense", "got it", "i see", "clear"])
        )

        # Count indicators of confusion
        confusion_indicators = sum(
            1 for exchange in recent_exchanges
            if any(phrase in exchange.get("student_message", "").lower()
                   for phrase in ["confused", "don't understand", "unclear", "lost", "difficult"])
        )

        # Calculate adjustment
        adjustment = understanding_indicators - confusion_indicators

        # Apply adjustment to current level
        new_level = current_level + adjustment

        # Ensure level is between 1 and 3
        return max(1, min(new_level, 3))

    def process_query(self,
                      query: str,
                      conversation_id: str,
                      conversation_history: List[Dict],
                      current_scaffolding_level: int = 1,
                      metadata: Dict = None) -> Tuple[str, List[Dict], int, str]:
        """
        Process a query using RAG multiquery approach
        Returns: (response, sources, scaffolding_level, detected_topic)
        """
        # Retrieve relevant documents using multiquery
        retrieved_docs = self.retriever.get_relevant_documents(query)

        # Detect topic
        detected_topic = self.detect_topic(query, retrieved_docs)

        # Determine scaffolding level
        scaffolding_level = self.determine_scaffolding_level(conversation_history, current_scaffolding_level)

        # Format conversation history for context
        formatted_history = ""
        if conversation_history:
            # Only include last 3 exchanges for context
            recent_history = conversation_history[-3:]
            formatted_history = "\n".join([
                f"Student: {exchange.get('student_message', '')}\nAssistant: {exchange.get('assistant_message', '')}"
                for exchange in recent_history
            ])

        # Create system prompt with appropriate scaffolding
        system_prompt = self.scaffolding_templates[scaffolding_level].format(topic=detected_topic)

        # Extract relevant content from retrieved documents
        relevant_content = "\n\n".join([
            f"DOCUMENT {i + 1}:\n{doc.page_content}"
            for i, doc in enumerate(retrieved_docs[:3])
        ])

        # Create user prompt with query, context, and history
        user_prompt = f"""
        Student question: {query}

        Conversation history:
        {formatted_history}

        Relevant information:
        {relevant_content}

        Remember to use scaffolding techniques appropriate for level {scaffolding_level}. 
        Guide the student to discover the answer rather than providing it directly.
        Ask questions that help the student think through the problem.
        """

        # Create prompt template
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("user", user_prompt)
        ])

        # Generate response
        response = self.llm(prompt.format_messages())

        # Format sources for citation
        sources = [
            {
                "content": doc.page_content[:200] + "...",  # Truncate for brevity
                "topic": doc.metadata.get("topic", detected_topic),
                "confidence": 0.8,  # Placeholder confidence score
                "metadata": doc.metadata
            }
            for doc in retrieved_docs[:3]
        ]

        return response.content, sources, scaffolding_level, detected_topic