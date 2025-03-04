import os
import glob
from typing import List, Dict, Tuple
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, OpenAI
from langchain.vectorstores import FAISS
from langchain.schema import Document
from dotenv import load_dotenv
import re

# Load environment variables
load_dotenv()

# Define topic keywords for classification
TOPIC_KEYWORDS = {
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


def detect_topic(text: str) -> str:
    """Detect the most likely topic for a chunk of text based on keyword frequency"""
    text_lower = text.lower()

    # Count keyword matches for each topic
    topic_scores = {topic: 0 for topic in TOPIC_KEYWORDS}

    for topic, keywords in TOPIC_KEYWORDS.items():
        for keyword in keywords:
            if keyword in text_lower:
                topic_scores[topic] += 1

    # Get topic with highest score
    max_score = 0
    detected_topic = "General Computer Architecture"

    for topic, score in topic_scores.items():
        if score > max_score:
            max_score = score
            detected_topic = topic

    # If no significant matches, keep as general
    if max_score < 2:
        return "General Computer Architecture"

    return detected_topic


def extract_subtopic_from_content(text: str, topic: str) -> str:
    """Extract a potential subtopic from the content"""
    # Look for section headers or key phrases
    lines = text.split('\n')
    for line in lines:
        line = line.strip()
        # Look for potential headers (short lines with capitalization)
        if 3 < len(line) < 60 and any(c.isupper() for c in line):
            return line

    # If no good header found, use the topic with a generic subtopic
    return f"{topic} Concept"


def process_pdf_file(file_path: str) -> List[Document]:
    """Process a single PDF file and return Documents with metadata"""
    try:
        # Load the PDF
        loader = PyPDFLoader(file_path)
        documents = loader.load()

        print(f"Processing {file_path} - {len(documents)} pages")

        # Add basic metadata to documents
        for i, doc in enumerate(documents):
            doc.metadata["source"] = file_path
            doc.metadata["page"] = i + 1
            # We'll detect topics after chunking for more accurate classification

        return documents
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return []


def process_content_directory(content_dir: str = "content") -> List[Document]:
    """Process all PDF files in content directory"""
    all_documents = []

    # Get all PDF files (not using recursive since we're assuming a flat structure)
    for file_path in glob.glob(f"{content_dir}/*.pdf"):
        documents = process_pdf_file(file_path)
        all_documents.extend(documents)

    return all_documents


def chunk_documents(documents: List[Document], chunk_size: int = 1000, chunk_overlap: int = 200) -> List[Document]:
    """Split documents into chunks for better retrieval and topic detection"""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""]
    )

    chunks = text_splitter.split_documents(documents)
    print(f"Split {len(documents)} documents into {len(chunks)} chunks")

    # Now detect topics for each chunk
    for chunk in chunks:
        # Detect topic based on content
        topic = detect_topic(chunk.page_content)
        chunk.metadata["topic"] = topic

        # Extract potential subtopic
        subtopic = extract_subtopic_from_content(chunk.page_content, topic)
        chunk.metadata["subtopic"] = subtopic

    return chunks


def create_vector_store(documents: List[Document], output_path: str = "faiss_index"):
    """Create and save a FAISS vector store from documents"""
    # Initialize embeddings
    embeddings = OpenAIEmbeddings()

    # Create vector store
    vector_store = FAISS.from_documents(documents, embeddings)

    # Save vector store
    vector_store.save_local(output_path)
    print(f"Vector store saved to {output_path}")

    return vector_store


def analyze_topic_distribution(chunks: List[Document]):
    """Analyze and print the distribution of topics in the processed chunks"""
    topic_counts = {}

    for chunk in chunks:
        topic = chunk.metadata.get("topic", "Unknown")
        if topic in topic_counts:
            topic_counts[topic] += 1
        else:
            topic_counts[topic] = 1

    print("\nTopic Distribution:")
    print("------------------")
    for topic, count in sorted(topic_counts.items(), key=lambda x: x[1], reverse=True):
        percentage = (count / len(chunks)) * 100
        print(f"{topic}: {count} chunks ({percentage:.1f}%)")


def main():
    """Main function to process content and create vector store"""
    print("Starting content indexing process...")

    # Create content directory if it doesn't exist
    os.makedirs("content", exist_ok=True)

    # Process content directory
    documents = process_content_directory()
    print(f"Loaded {len(documents)} document pages")

    if not documents:
        print("No documents found. Please add PDF files to the content directory.")
        return

    # Chunk documents and detect topics
    chunks = chunk_documents(documents)

    # Analyze topic distribution
    analyze_topic_distribution(chunks)

    # Create vector store
    create_vector_store(chunks)

    print("Content indexing complete!")


if __name__ == "__main__":
    main()