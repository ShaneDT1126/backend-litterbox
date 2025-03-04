import uuid
import re
from typing import Dict, Any, List


def generate_conversation_id() -> str:
    """Generates a unique conversation ID"""
    return str(uuid.uuid4())


def extract_concepts(text: str) -> List[str]:
    """Extract key Computer Organization Architecture concepts from PDFs"""
    # List of key COA concepts to look for
    coa_concepts = [
        "CPU", "ALU", "control unit", "register", "cache", "memory hierarchy",
        "pipeline", "instruction set", "addressing mode", "bus", "interrupt",
        "virtual memory", "RISC", "CISC", "branch prediction", "memory management",
        "I/O", "DMA", "clock cycle", "throughput", "latency", "von Neumann",
        "Harvard architecture", "multicore", "superscalar", "out-of-order execution",
        "speculative execution", "cache coherence", "MESI protocol", "TLB",
        "page table", "direct mapping", "set associative", "fully associative"
    ]

    found_concepts = []
    text_lower = text.lower()

    for concept in coa_concepts:
        if concept.lower() in text_lower:
            found_concepts.append(concept)

    return found_concepts


def analyze_question_complexity(question: str) -> int:
    """
    Analyze the complexity of a question to help determine scaffolding level
    Returns: 1 (basic), 2 (intermediate), or 3 (advanced)
    """
    # Keywords indicating complexity
    basic_keywords = ["what is", "define", "describe", "explain", "basic", "introduction"]
    intermediate_keywords = ["how does", "why does", "compare", "contrast", "analyze", "relationship"]
    advanced_keywords = ["evaluate", "design", "optimize", "trade-off", "implement", "synthesize", "critique"]

    question_lower = question.lower()

    # Count keyword matches
    basic_count = sum(1 for word in basic_keywords if word in question_lower)
    intermediate_count = sum(1 for word in intermediate_keywords if word in question_lower)
    advanced_count = sum(1 for word in advanced_keywords if word in question_lower)

    # Determine complexity based on highest count
    if advanced_count > 0:
        return 3
    elif intermediate_count > 0:
        return 2
    else:
        return 1


def format_response_for_teams(response: str) -> str:
    """Format the response for better display in Microsoft Teams"""
    # Convert markdown headers to bold text
    response = re.sub(r'##\s+(.*)', r'**\1**', response)

    # Ensure code blocks are properly formatted
    response = re.sub(r'```(\w+)\n', r'```\1\n', response)

    # Add line breaks for better readability
    response = response.replace('. ', '.\n\n')

    return response