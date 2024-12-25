from fastapi import FastAPI, UploadFile, File, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
import uuid
import wave
from pydantic import BaseModel
from typing import Optional, List, Dict
import asyncio
import os
import io
import tempfile
from chromadb.config import Settings
from langchain_community.vectorstores import Chroma
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_core.output_parsers import StrOutputParser
from deepgram_tts import text_to_speech_buffer
import json
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from prompts import create_system_prompt, create_dynamic_prompt
from enhanced_rag_content_processor import process_document_with_enhancements
import shutil
import threading


AUDIO_DIR = os.path.join(os.path.dirname(__file__), 'audio')
if not os.path.exists(AUDIO_DIR):
    os.makedirs(AUDIO_DIR, exist_ok=True)

db_lock = threading.Lock()

audio_transcription_server = None

app = FastAPI()

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global state class
class RAGState:
    def __init__(self):
        self.vectorstore = None
        self.rag_chain = None
        self.chat_history = []
        self.book_title = None
        self.reader_name = "Lucy"
        # self.tts = AsyncTextToSpeech()
        # self.transcriber = DeepgramTranscriber()
        self.temperature = 0.0

# Initialize global state
state = RAGState()

# Create a persistent directory for the database
PERSIST_DIR = os.path.join(tempfile.gettempdir(), 'book_companion_db')
if not os.path.exists(PERSIST_DIR):
    print("creating persist directory", PERSIST_DIR)
    os.makedirs(PERSIST_DIR, mode=0o777)

CHROMA_SETTINGS = Settings(
    persist_directory=PERSIST_DIR,
    anonymized_telemetry=False,
    allow_reset=True,
    is_persistent=True
)

class ChatMessage(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    message: str
    reader_name: Optional[str] = "Lucy"

def format_chat_history(chat_history: List[Dict], max_length: int = 4) -> str:
    """Format chat history for RAG context."""
    if not chat_history:
        return ""

    formatted_history = []
    for message in chat_history:
        role = message.get('role', 'unknown').title()
        content = message.get('content', '')
        formatted_history.append(f"{role}: {content}")

    return "\n".join(formatted_history)

def get_embeddings_model():
    """Initialize and return the embedding model."""
    return OllamaEmbeddings(
        model='nomic-embed-text',
        base_url="http://localhost:11434"
    )

def cleanup_chroma():
    """Clean up ChromaDB resources."""
    with db_lock:
        try:
            if state.vectorstore is not None:
                # Attempt to reset the client
                state.vectorstore._client.reset()
                # Clear the vectorstore reference
                state.vectorstore = None

            # Remove the persist directory
            if os.path.exists(PERSIST_DIR):
                shutil.rmtree(PERSIST_DIR)
                os.makedirs(PERSIST_DIR)

        except Exception as e:
            print(f"Error during ChromaDB cleanup: {e}")
            state.vectorstore = None

def process_document(file_content: bytes, filename: str) -> List:
    """Process and split the uploaded document."""
    try:
        # Create a temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(filename)[1]) as temp_file:
            temp_file.write(file_content)
            temp_file_path = temp_file.name

        # Create a file-like object with name attribute
        class FileWithName:
            def __init__(self, file_path, original_name, content):
                self.file_path = file_path
                self.name = original_name
                self._content = content
                self.type = "application/pdf" if original_name.endswith('.pdf') else "text/plain"

            def read(self):
                return self._content

            def getvalue(self):
                return self._content

        # Create file-like object with the content
        file_obj = FileWithName(temp_file_path, filename, file_content)

        print('processing document')
        # Process the document
        doc_splits = process_document_with_enhancements(
            file_obj,
            chunk_size=800,
            chunk_overlap=100
        )

        print('doc_splits', len(doc_splits))

        # Clean up
        os.unlink(temp_file_path)

        return doc_splits

    except Exception as e:
        raise Exception(f"Error processing document: {str(e)}")
def create_custom_retriever(vectorstore, embedding_function):
    """Create a custom retriever function."""
    def custom_retriever(query, k=5):
        query_embedding = embedding_function.embed_query(query)
        retriever = vectorstore.as_retriever(search_kwargs={"k": k})
        retrieved_docs = retriever.invoke(query)

        doc_embeddings = [
            embedding_function.embed_query(doc.page_content)
            for doc in retrieved_docs
        ]

        similarities = cosine_similarity([query_embedding], doc_embeddings)[0]

        filtered_docs = [
            doc for doc, sim in zip(retrieved_docs, similarities)
            if sim > 0.3
        ]

        return filtered_docs

    return custom_retriever

@app.get("/")
def api_home():
    return {'detail': 'Welcome to Book Companion API'}

@app.post("/upload")
async def upload_document(file: UploadFile = File(...), reader_name: str = "Lucy"):
    try:
        # Clean up existing ChromaDB and state
        cleanup_chroma()

        state.chat_history = []
        state.rag_chain = None
        state.book_title = None

        # Ensure the persist directory exists and is writable
        if not os.path.exists(PERSIST_DIR):
            os.makedirs(PERSIST_DIR, mode=0o777)

        # Read file content
        file_content = await file.read()

        # Process document
        doc_splits = process_document(file_content, file.filename)

        # Get book title
        state.book_title = doc_splits[0].page_content.split('\n')[0].strip()
        state.reader_name = reader_name

        print('book title', state.book_title)
        print('reader name', state.reader_name)

        print('initializing embedding model')
        # Initialize embedding model
        embedding_function = get_embeddings_model()

        print('creating vectorstore')
        # Create vectorstore
        state.vectorstore = Chroma.from_documents(
            documents=doc_splits,
            embedding=embedding_function,
            collection_metadata={"hnsw:space": "cosine"},
            client_settings=CHROMA_SETTINGS
        )

        print('custom retriever')
        # Create custom retriever
        custom_retriever = create_custom_retriever(state.vectorstore, embedding_function)

        print('initializing LLM')
        # Initialize LLM
        model_local = ChatOllama(
            model="llama3.2",
            base_url="http://localhost:11434",
            temperature=state.temperature,
            num_predict=150,
            top_k=10,
            top_p=0.1,
        )

        print('creating system message')
        # Create system message
        system_message = create_system_prompt(reader_name, state.book_title)
        print('system message', system_message)
        state.chat_history = [{"role": "system", "content": system_message}]

        print('creating RAG chain')
        # Create RAG chain
        state.rag_chain = (
            {
                "context": lambda x: "\n\n".join([
                    doc.page_content for doc in custom_retriever(
                        x["question"] if isinstance(x, dict) else x,
                        k=5
                    )
                ]),
                "chat_history": lambda x: format_chat_history(
                    x.get("chat_history", []) if isinstance(x, dict) else []
                ),
                "question": lambda x: x["question"] if isinstance(x, dict) else x,
                "reader_name": lambda x: reader_name,
                "book_title": lambda x: state.book_title
            }
            | create_dynamic_prompt()
            | model_local
            | StrOutputParser()
        )


        # Generate welcome message
        welcome_prompt = f"""Hi there {reader_name}! I'm so excited to talk about {state.book_title} with you!"""
        # welcome_prompt = f"""My name is {reader_name}! I'm so excited to talk about {state.book_title} with you! Let's start talking about one (and only one) of the most important themes of the book. Ask me anything!"""
        print('welcome_prompt', welcome_prompt)
        welcome_response = state.rag_chain.invoke({
            "question": welcome_prompt,
            "chat_history": state.chat_history,
            "reader_name": reader_name,
            "book_title": state.book_title
        })

        print('addding welcome message to chat history')
        # Add welcome message to chat history
        state.chat_history.append({"role": "assistant", "content": welcome_response})

        print('generating audio for welcome message')
        # Generate audio for welcome message
        # audio_data = state.tts.generate(welcome_response)
        audio_data = text_to_speech_buffer(welcome_response)

        # import base64
        # audio_base64 = base64.b64encode(audio_data).decode() if audio_data else None

        if audio_data:
            # Generate unique filename
            audio_filename = f"{uuid.uuid4()}.wav"
            audio_path = os.path.join(AUDIO_DIR, audio_filename)

            # Save audio file
            with wave.open(audio_path, 'wb') as wave_file:
                wave_file.setnchannels(1)  # mono
                wave_file.setsampwidth(2)  # 2 bytes per sample (16-bit)
                wave_file.setframerate(48000)  # sample rate
                wave_file.writeframes(audio_data)

            audio_array = np.frombuffer(audio_data, dtype=np.int16)
            audio_list = audio_array.tolist()

            return {
                "status": "success",
                "book_title": state.book_title,
                "welcome_message": welcome_response,
                "audio_url": f"/audio/{audio_filename}",
                "audio": audio_list
            }
        else:
            return {
                "status": "success",
                "book_title": state.book_title,
                "welcome_message": welcome_response,
                "audio_url": None,
                "audio": None
            }

    except Exception as e:
        print(f"Upload error: {str(e)}")
        # Clean up in case of error
        cleanup_chroma()
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/chat")
async def chat(request: ChatRequest):
    if not state.rag_chain:
        raise HTTPException(status_code=400, detail="Please upload a document first")

    try:
        print('adding user message to history', request.message)
        # Add user message to history
        state.chat_history.append({"role": "user", "content": request.message})

        print('generating response invoking rag_chain')
        # Generate response
        response = state.rag_chain.invoke({
            "question": request.message,
            "chat_history": state.chat_history,
            "reader_name": request.reader_name,
            "book_title": state.book_title
        })

        print('adding response to history', response)
        # Add response to history
        state.chat_history.append({"role": "assistant", "content": response})

        # Generate audio
        # audio_data = None; # state.tts.generate(response)

        audio_data = text_to_speech_buffer(response)

        if audio_data:
            # Generate unique filename
            audio_filename = f"{uuid.uuid4()}.wav"
            audio_path = os.path.join(AUDIO_DIR, audio_filename)

            # Save audio file
            with wave.open(audio_path, 'wb') as wave_file:
                wave_file.setnchannels(1)  # mono
                wave_file.setsampwidth(2)  # 2 bytes per sample (16-bit)
                wave_file.setframerate(48000)  # sample rate
                wave_file.writeframes(audio_data)



            audio_array = np.frombuffer(audio_data, dtype=np.int16)
            audio_list = audio_array.tolist()

            return {
                "message": response,
                "audio_url": f"/audio/{audio_filename}",
                "audio": audio_list
            }
        else:
            return {
                "message": response,
                "audio_url": None,
                "audio": None
            }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/chat-history")
async def get_chat_history():
    return {"history": state.chat_history}

@app.post("/clear")
async def clear_database():
    with db_lock:
        try:
            cleanup_chroma()
            state.chat_history = []
            return {"status": "success", "message": "Database cleared"}
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, ws_max_size=1024*1024)
