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
import sys
from pathlib import Path

from chromadb.config import Settings
from chromadb import Client
from chromadb.utils import embedding_functions

from langchain_chroma import Chroma
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.output_parsers import StrOutputParser
from deepgram_tts import text_to_speech_buffer
import json
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from prompts import create_system_prompt, create_dynamic_prompt
from enhanced_rag_content_processor import process_document_with_enhancements
import shutil
import threading
from dotenv import load_dotenv
from AudioTranscriptionServer import AudioTranscriptionServer
from datetime import datetime

os.environ["TOKENIZERS_PARALLELISM"] = "false"

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")


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
        self.current_collection = None

# Initialize global state
state = RAGState()

# Create a persistent directory for the database
HOME = str(Path.home())
PERSIST_DIR = os.path.join(HOME, '.book_companion_db')
def ensure_directory_permissions():
    """Ensure the persistence directory exists and has correct permissions"""
    try:
        # Create parent directory if it doesn't exist
        os.makedirs(PERSIST_DIR, exist_ok=True)

        # Set permissions for the directory
        os.chmod(PERSIST_DIR, 0o777)

        # Set permissions for all existing contents
        for root, dirs, files in os.walk(PERSIST_DIR):
            for d in dirs:
                os.chmod(os.path.join(root, d), 0o777)
            for f in files:
                os.chmod(os.path.join(root, f), 0o777)
    except Exception as e:
        print(f"Error setting directory permissions: {e}")

# Call this function at startup
ensure_directory_permissions()

CHROMA_SETTINGS = Settings(
    persist_directory=PERSIST_DIR,
    anonymized_telemetry=False,
    allow_reset=True,
    is_persistent=True
)

class Book(BaseModel):
    title: str
    filename: str
    upload_date: datetime
    collection_name: str

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
    return HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2",
        model_kwargs={'device': 'cpu'}
    )

def cleanup_chroma():
    """Clean up ChromaDB resources."""
    with db_lock:
        try:
            if state.vectorstore is not None:
                try:
                    # Attempt to reset the client
                    state.vectorstore._client.reset()
                except Exception as e:
                    print(f"Error resetting client: {e}")
                # Clear the vectorstore reference
                state.vectorstore = None

            # Ensure the directory exists with proper permissions
            ensure_directory_permissions()

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

def get_books_collection():
    """Get or create the books metadata collection"""
    try:
        # Create ChromaDB client
        client = Client(Settings(
            persist_directory=PERSIST_DIR,
            anonymized_telemetry=False,
            allow_reset=True,
            is_persistent=True
        ))

        # Get or create collection
        collection = client.get_or_create_collection(
            name="books_metadata",
            embedding_function=embedding_functions.DefaultEmbeddingFunction()
        )
        return collection
    except Exception as e:
        print(f"Error getting books collection: {e}")
        return None

def save_book_metadata(title: str, filename: str, collection_name: str):
    """Save book metadata to the books collection"""
    book = Book(
        title=title,
        filename=filename,
        upload_date=datetime.now(),
        collection_name=collection_name
    )

    metadata = {
        "title": book.title,
        "filename": book.filename,
        "upload_date": book.upload_date.isoformat(),
        "collection_name": book.collection_name
    }
    print('metadata', metadata)

    try:
        collection = get_books_collection()
        if collection:
            collection.add(
                documents=[book.title],
                metadatas=[metadata],
                ids=[collection_name]
            )
    except Exception as e:
        print(f"Error adding book to collection: {e}")

def get_available_books() -> List[Dict]:
    """Get list of available books"""
    collection = get_books_collection()
    if not collection:
        return []

    try:
        results = collection.get()
        books = []
        for i, metadata in enumerate(results['metadatas']):
            if metadata:  # Check if metadata exists
                books.append({
                    "title": metadata["title"],
                    "filename": metadata["filename"],
                    "upload_date": metadata["upload_date"],
                    "collection_name": metadata["collection_name"]
                })
        return books
    except Exception as e:
        print(f"Error getting books: {e}")
        return []

def load_book_collection(collection_name: str):
    """Load a book's vector collection"""
    embedding_function = get_embeddings_model()
    return Chroma(
        collection_name=collection_name,
        embedding_function=embedding_function,
        client_settings=CHROMA_SETTINGS
    )

@app.get("/")
def api_home():
    return {'detail': 'Welcome to Book Companion API'}

@app.post("/upload")
async def upload_document(file: UploadFile = File(...), reader_name: str = "Lucy"):
    try:
        # Clean up existing ChromaDB and state
        cleanup_chroma()

        # Ensure directory permissions
        ensure_directory_permissions()

        state.chat_history = []
        state.rag_chain = None
        state.book_title = None

        # Read file content
        try:
            file_content = await file.read()
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Error reading file: {str(e)}")

        # Process document
        try:
            doc_splits = process_document(file_content, file.filename)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Error processing document: {str(e)}")

        if not doc_splits:
            raise HTTPException(status_code=400, detail="No content could be extracted from the document")

        # Get book title
        try:
            state.book_title = doc_splits[0].page_content.split('\n')[0].strip()
            state.reader_name = reader_name
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Error extracting book title: {str(e)}")

        print('book title', state.book_title)
        print('reader name', state.reader_name)

        print('initializing embedding model')
        # Initialize embedding model
        try:
            embedding_function = get_embeddings_model()
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error initializing embedding model: {str(e)}")

        print('creating vectorstore')
        # Create vectorstore
        collection_name = f"book_{uuid.uuid4().hex}"
        print('collection name', collection_name)

        try:
            # First attempt with standard initialization
            state.vectorstore = Chroma.from_documents(
                documents=doc_splits,
                embedding=embedding_function,
                collection_name=collection_name,
                collection_metadata={"hnsw:space": "cosine"},
                client_settings=CHROMA_SETTINGS,
            )
        except Exception as e:
            print(f"First attempt at creating vectorstore failed: {e}")
            try:
                # Second attempt with explicit client creation
                client = Client(CHROMA_SETTINGS)
                state.vectorstore = Chroma.from_documents(
                    documents=doc_splits,
                    embedding=embedding_function,
                    collection_name=collection_name,
                    collection_metadata={"hnsw:space": "cosine"},
                    client=client,
                )
            except Exception as e2:
                raise HTTPException(
                    status_code=500,
                    detail=f"Error creating vector store: {str(e2)}"
                )

        try:
            print('saving book metadata', state.book_title, file.filename, collection_name)
            save_book_metadata(state.book_title, file.filename, collection_name)
            print('adding collection name to state', collection_name)
            state.current_collection = collection_name
        except Exception as e:
            print(f"Error saving book metadata: {e}")
            # Continue even if metadata saving fails

        print('custom retriever')
        # Create custom retriever
        custom_retriever = create_custom_retriever(state.vectorstore, embedding_function)

        print('initializing LLM')
        # Initialize LLM
        model_local = ChatGroq(
            temperature=state.temperature,
            groq_api_key=GROQ_API_KEY,
            model_name="llama-3.3-70b-versatile",
            max_tokens=150,
            model_kwargs={
                "top_p": 0.1
            }
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
        print('welcome_prompt', welcome_prompt)
        welcome_response = state.rag_chain.invoke({
            "question": welcome_prompt,
            "chat_history": state.chat_history,
            "reader_name": reader_name,
            "book_title": state.book_title
        })

        print('adding welcome message to chat history')
        # Add welcome message to chat history
        state.chat_history.append({"role": "assistant", "content": welcome_response})

        print('generating audio for welcome message')
        # Generate audio for welcome message
        audio_data = text_to_speech_buffer(welcome_response)

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

    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    except Exception as e:
        print(f"Unexpected error in upload: {str(e)}")
        import traceback
        traceback.print_exc()
        # Clean up in case of error
        cleanup_chroma()
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")
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

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    print("New WebSocket connection attempt")
    transcription_server = AudioTranscriptionServer()
    try:
        await transcription_server.handle_fastapi_websocket(websocket)
    except WebSocketDisconnect:
        print("Client disconnected normally")
    except Exception as e:
        print(f"WebSocket error: {str(e)}")
        import traceback
        traceback.print_exc()
    finally:
        print("WebSocket connection closed")

@app.get("/books")
async def list_books():
    """Get list of available books"""
    try:
        books = get_available_books()
        return {"books": books}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/books/{collection_name}/load")
async def load_book(collection_name: str, reader_name: str = "Lucy"):
    """Load a specific book"""
    try:
        # Get book metadata
        collection = get_books_collection()
        print('collection', collection)
        result = collection.get(ids=[collection_name])
        print('result', result)
        if not result['metadatas'] or not result['metadatas'][0]:
            raise HTTPException(status_code=404, detail="Book not found")

        metadata = result['metadatas'][0]

        # Load the book's vector collection
        state.vectorstore = load_book_collection(collection_name)
        state.current_collection = collection_name
        state.book_title = metadata["title"]
        state.reader_name = reader_name
        state.chat_history = []

        print('vectorstore', state.vectorstore)

        # Initialize LLM and RAG chain
        embedding_function = get_embeddings_model()
        custom_retriever = create_custom_retriever(state.vectorstore, embedding_function)

        print('embedding_function', embedding_function)
        print('custom_retriever', custom_retriever)

        print('creating model')
        model_local = ChatGroq(
            temperature=state.temperature,
            groq_api_key=GROQ_API_KEY,
            model_name="llama-3.3-70b-versatile",
            max_tokens=150,
            model_kwargs={"top_p": 0.1}
        )

        # Create system message and initialize chat
        system_message = create_system_prompt(reader_name, state.book_title)
        state.chat_history = [{"role": "system", "content": system_message}]

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
        welcome_response = state.rag_chain.invoke({
            "question": welcome_prompt,
            "chat_history": state.chat_history,
            "reader_name": reader_name,
            "book_title": state.book_title
        })

        state.chat_history.append({"role": "assistant", "content": welcome_response})

        return {
            "status": "success",
            "book_title": state.book_title,
            "welcome_message": welcome_response
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.on_event("startup")
async def startup_event():
    ensure_directory_permissions()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, ws_max_size=1024*1024)
