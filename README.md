# Litterbox: Educational RAG Chatbot for Computer Organization Architecture

Litterbox is an educational chatbot that uses a scaffolding approach to help Computer Organization Architecture students learn concepts without providing direct answers. It uses Retrieval-Augmented Generation (RAG) with multiquery capabilities and conversation memory.

## Features

- RAG multiquery for accurate information retrieval
- Conversation memory to maintain context
- Adaptive scaffolding based on student understanding
- Topic detection for Computer Organization Architecture concepts
- Integration with Microsoft Teams via Copilot Studio

## Setup Instructions

### Prerequisites

- Python 3.8+
- MongoDB
- OpenAI API key
- PDF content files for Computer Organization Architecture

### Installation

1. Clone the repository:git clone https://github.com/yourusername/litterbox.git
2. python -m venv venv source venv/bin/activate, On Windows: venv\Scripts\activate 
3. Install dependencies: pip install -r requirements.txt
4. Create a `.env` file with your API keys: OPENAI_API_KEY=your_openai_api_key MONGO_URI=mongodb_connection

### Indexing Content
1. Run the content indexing script: python scripts/index_content.py

### Running the Server
1. Start the server: python run_server.py

### Docker Deployment
1. Build and run using Docker Compose: docker-compose up -d

## API Endpoints

- `POST /query`: Process a student query 
- `GET /conversations/{student_id}`: Get all conversations for a student 
- `GET /health`: Health check endpoint 

## Integration with Copilot Studio 
See the documentation for instructions on integrating with Microsoft Copilot Studio.