# Book Companion AI
## Created by Vasundhara Parakh
An intelligent AI system that helps readers engage with books through interactive conversations, audio interactions, and enhanced content analysis.

## 🌟 Features
- RAG-based book analysis and discussion
- Text-to-Speech and Speech-to-Text capabilities
- Enhanced content processing with themes, characters, and discussion topics
- Real-time websocket communication for audio interactions
- Interactive chat system with conversation history

## 🛠️ Components

- **api.py**: FastAPI-based backend that:
  - Implements RAG (Retrieval-Augmented Generation)
  - Manages ChromaDB for data storage
  - Integrates LLaMA model for text processing
  - Handles Deepgram API integration for audio processing

- **AudioTranscriptionServer.py**:
  - Websocket-based server for audio transcription
  - Utilizes Deepgram SDK for accurate speech-to-text conversion

- **enhanced_rag_content_processor.py**:
  - Enhances RAG content with additional context
  - Generates examples, themes, and discussion topics
  - Processes character information and relationships

- **prompts.py**:
  - Contains template prompts for AI interactions
  - Manages conversation flow and context

## 🚀 Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/book-companion-ai.git
cd book-companion-ai
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## 💫 Usage

Start all services with a single command:
```bash
python run_servers.py
```

This command initializes:
- FastAPI server
- Websocket server for audio communication
- RAG system
- Database connections

## 🔌 API Endpoints

### POST `/api/upload`
Upload book content for processing and analysis.

### POST `/api/chat`
Interact with the AI system through text-based conversations.

### GET `/api/chat-history`
Retrieve conversation history and previous interactions.

## 📝 Environment Variables

Create a `.env` file with the following variables:
```
DEEPGRAM_API_KEY=your_key_here
```
