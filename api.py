from fastapi import FastAPI, UploadFile, File, Form, HTTPException, WebSocket, WebSocketDisconnect
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
import subprocess

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
from uuid import uuid4
from models import RAGState
from datetime import datetime, timedelta

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

class ClientRequest(BaseModel):
    client_id: str
    reader_name: Optional[str] = "Lucy"

class ChatRequest(BaseModel):
    message: str
    client_id: str  # Add client_id to chat requests
    reader_name: Optional[str] = "Lucy"

class LoadBookRequest(BaseModel):
    client_id: str  # Add client_id to load book requests
    reader_name: Optional[str] = "Lucy"

class ConnectionManager:
    def __init__(self):
        self.active_connections: Dict[str, WebSocket] = {}
        self.connection_states: Dict[str, RAGState] = {}
        self.transcription_servers: Dict[str, AudioTranscriptionServer] = {}
        self.last_activity: Dict[str, datetime] = {}

    async def connect(self, websocket: WebSocket, client_id: str):
        await websocket.accept()
        self.active_connections[client_id] = websocket
        self.connection_states[client_id] = RAGState()
        self.transcription_servers[client_id] = AudioTranscriptionServer(
            client_id=client_id,
            connection_manager=self
        )
    def disconnect(self, client_id: str):
        if client_id in self.active_connections:
            del self.active_connections[client_id]
        if client_id in self.connection_states:
            del self.connection_states[client_id]
        if client_id in self.transcription_servers:
            del self.transcription_servers[client_id]

    def get_state(self, client_id: str) -> RAGState:
        return self.connection_states.get(client_id)

    def get_transcription_server(self, client_id: str) -> AudioTranscriptionServer:
        return self.transcription_servers.get(client_id)

    async def cleanup_inactive_clients(self):
        """Remove clients that haven't been active for more than 1 hour"""
        now = datetime.now()
        inactive_clients = [
            client_id for client_id, last_active in self.last_activity.items()
            if now - last_active > timedelta(hours=1)
        ]
        for client_id in inactive_clients:
            self.disconnect(client_id)

    def update_activity(self, client_id: str):
        """Update the last activity time for a client"""
        self.last_activity[client_id] = datetime.now()

# Initialize global state
state = RAGState()

# Create a persistent directory for the database
HOME = str(Path.home())
PERSIST_DIR = os.path.join(HOME, '.book_companion_db')

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
            # Reset state
            if state.vectorstore is not None:
                state.vectorstore = None

            # Reinitialize the database directory
            initialize_database()

        except Exception as e:
            print(f"Error during ChromaDB cleanup: {e}")
        finally:
            state.vectorstore = None
            state.current_collection = None

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

def get_chroma_client():
    """Get a fresh ChromaDB client with proper settings"""
    initialize_database()  # Ensure directory exists with proper permissions
    return Client(Settings(
        persist_directory=PERSIST_DIR,
        anonymized_telemetry=False,
        allow_reset=True,
        is_persistent=True
    ))

@app.post("/upload")
async def upload_document(file: UploadFile = File(...), client_id: str = Form(...), reader_name: str = "Lucy"):
    try:
        # Clean up existing ChromaDB and state
        # initialize_database()
        # Get or create client state
        client_state = manager.connection_states.get(client_id)
        if not client_state:
            client_state = RAGState()
            manager.connection_states[client_id] = client_state

        client_state.chat_history = []
        client_state.rag_chain = None
        client_state.book_title = None

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
            client_state.book_title = doc_splits[0].page_content.split('\n')[0].strip()
            client_state.reader_name = reader_name
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Error extracting book title: {str(e)}")

        print('book title', client_state.book_title)
        print('reader name', client_state.reader_name)

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
            client = get_chroma_client()
            print("Created ChromaDB client")
            state.vectorstore = Chroma.from_documents(
                documents=doc_splits,
                embedding=embedding_function,
                collection_name=collection_name,
                collection_metadata={"hnsw:space": "cosine"},
                client=client,
            )
            print("Created vectorstore successfully")
        except Exception as e:
            print(f"Error creating vectorstore: {e}")
            subprocess.run(['ls', '-la', PERSIST_DIR], check=True)
            raise HTTPException(
                status_code=500,
                detail=f"Error creating vector store: {str(e)}"
            )

        try:
            print('saving book metadata', client_state.book_title, file.filename, collection_name)
            save_book_metadata(client_state.book_title, file.filename, collection_name)
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
        system_message = create_system_prompt(reader_name, client_state.book_title)
        print('system message', system_message)
        client_state.chat_history = [{"role": "system", "content": system_message}]

        print('creating RAG chain')
        # Create RAG chain
        client_state.rag_chain = (
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
        welcome_prompt = f"""Hi there {reader_name}! I'm so excited to talk about {client_state.book_title} with you!"""
        print('welcome_prompt', welcome_prompt)
        welcome_response = client_state.rag_chain.invoke({
            "question": welcome_prompt,
            "chat_history": client_state.chat_history,
            "reader_name": reader_name,
            "book_title": client_state.book_title
        })

        print('adding welcome message to chat history')
        # Add welcome message to chat history
        client_state.chat_history.append({"role": "assistant", "content": welcome_response})

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
                "book_title": client_state.book_title,
                "welcome_message": welcome_response,
                "audio_url": f"/audio/{audio_filename}",
                "audio": audio_list
            }
        else:
            return {
                "status": "success",
                "book_title": client_state.book_title,
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
    client_state = manager.connection_states.get(request.client_id)
    if not client_state or not client_state.rag_chain:
        raise HTTPException(status_code=400, detail="Please upload a document first")

    try:
        print('adding user message to history', request.message)
        # Add user message to history
        client_state.chat_history.append({"role": "user", "content": request.message})

        print('generating response invoking rag_chain')
        # Generate response
        response = client_state.rag_chain.invoke({
            "question": request.message,
            "chat_history": client_state.chat_history,
            "reader_name": request.reader_name,
            "book_title": client_state.book_title
        })

        print('adding response to history', response)
        # Add response to history
        client_state.chat_history.append({"role": "assistant", "content": response})

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
    client_state = manager.connection_states.get(request.client_id)
    return {"history": client_state.chat_history}

@app.post("/clear")
async def clear_database():
    """Clear all collections and reset the database"""
    try:
        cleanup_chroma()
        state.chat_history = []
        state.book_title = None
        return {"status": "success", "message": "Database cleared"}
    except Exception as e:
        print(f"Error clearing database: {e}")
        raise HTTPException(status_code=500, detail=str(e))

manager = ConnectionManager()

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    client_id = str(uuid4())  # Generate unique client ID
    print(f"New WebSocket connection attempt from client {client_id}")

    try:
        # Connect and store client-specific state
        await manager.connect(websocket, client_id)

        transcription_server = manager.get_transcription_server(client_id)
        if not transcription_server:
            raise Exception(f"Failed to initialize transcription server for client {client_id}")

        # Handle the WebSocket connection with client-specific state
        await transcription_server.handle_fastapi_websocket(
            websocket
        )
    except WebSocketDisconnect:
        print(f"Client {client_id} disconnected normally")
    except Exception as e:
        print(f"WebSocket error for client {client_id}: {str(e)}")
        import traceback
        traceback.print_exc()
    finally:
        print(f"WebSocket connection closed for client {client_id}")
        manager.disconnect(client_id)

@app.get("/books")
async def list_books():
    """Get list of available books"""
    try:
        books = get_available_books()
        return {"books": books}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/books/{collection_name}/load")
async def load_book(collection_name: str, request: LoadBookRequest):
    """Load a specific book"""
    try:
        client_state = manager.connection_states.get(request.client_id)
        if not client_state:
            client_state = RAGState()
            manager.connection_states[request.client_id] = client_state
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
        client_state.current_collection = collection_name
        client_state.book_title = metadata["title"]
        client_state.reader_name = request.reader_name
        client_state.chat_history = []

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
        system_message = create_system_prompt(request.reader_name, client_state.book_title)
        client_state.chat_history = [{"role": "system", "content": system_message}]
        print(system_message)
        client_state.rag_chain = (
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
                "reader_name": lambda x: request.reader_name,
                "book_title": lambda x: client_state.book_title
            }
            | create_dynamic_prompt()
            | model_local
            | StrOutputParser()
        )

        # Generate welcome message
        welcome_prompt = f"""Hi there {request.reader_name}! I'm so excited to talk about {client_state.book_title} with you!"""
        welcome_response = client_state.rag_chain.invoke({
            "question": welcome_prompt,
            "chat_history": client_state.chat_history,
            "reader_name": request.reader_name,
            "book_title": client_state.book_title
        })

        client_state.chat_history.append({"role": "assistant", "content": welcome_response})

        # Generate audio for welcome message
        print('generating audio for welcome message')
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
                "book_title": client_state.book_title,
                "welcome_message": welcome_response,
                "audio_url": f"/audio/{audio_filename}",
                "audio": audio_list
            }
        else:
            return {
                "status": "success",
                "book_title": client_state.book_title,
                "welcome_message": welcome_response,
                "audio_url": None,
                "audio": None
            }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

def initialize_database():
    """Initialize the database directory with correct permissions"""
    try:
        # Create base directory if it doesn't exist
        os.makedirs(PERSIST_DIR, exist_ok=True)

        # Set permissions for the entire directory tree
        subprocess.run(['chmod', '-R', '777', PERSIST_DIR], check=True)

        print(f"Initialized database directory: {PERSIST_DIR}")
        # Print current permissions for debugging
        subprocess.run(['ls', '-la', PERSIST_DIR], check=True)

    except Exception as e:
        print(f"Error initializing database: {e}")
        raise e

async def verify_database_state():
    """Verify the database state and log available books"""
    try:
        books = await list_books()
        print(f"Found {len(books.get('books', []))} books in database")
        for book in books.get('books', []):
            print(f"  - {book['title']}")
    except Exception as e:
        print(f"Error verifying database state: {e}")

@app.get("/debug")
async def debug_permissions():
    try:
        subprocess.run(['ls', '-la', PERSIST_DIR], check=True)
        subprocess.run(['whoami'], check=True)
        return {"message": "Debug info printed to logs"}
    except Exception as e:
        return {"error": str(e)}

async def periodic_cleanup():
    while True:
        await asyncio.sleep(3600)  # Run every hour
        await manager.cleanup_inactive_clients()

@app.on_event("startup")
async def startup_event():
    try:
        print("Initializing database on startup...")
        initialize_database()
        await verify_database_state()

        # Start periodic cleanup
        asyncio.create_task(periodic_cleanup())
    except Exception as e:
        print(f"Error during startup: {e}")

def delete_book_collection(collection_name: str):
    """Delete a specific book collection from ChromaDB"""
    try:
        client = get_chroma_client()
        # Delete the vector collection
        client.delete_collection(name=collection_name)

        # Remove from books metadata collection
        books_collection = get_books_collection()
        if books_collection:
            books_collection.delete(ids=[collection_name])

        print(f"Deleted collection: {collection_name}")
        return True
    except Exception as e:
        print(f"Error deleting collection {collection_name}: {e}")
        return False

def delete_all_collections():
    """Delete all book collections and reset the database"""
    try:
        client = get_chroma_client()

        # Get list of all collections
        collections = client.list_collections()

        # Delete each collection
        for collection in collections:
            if collection.name != "books_metadata":  # Preserve the metadata collection
                client.delete_collection(name=collection.name)

        # Reset the books metadata collection
        books_collection = get_books_collection()
        if books_collection:
            # Get all document IDs first
            results = books_collection.get()
            if results and results['ids']:
                # Delete all documents by their IDs
                books_collection.delete(ids=results['ids'])

        return True
    except Exception as e:
        print(f"Error deleting all collections: {e}")
        return False

@app.delete("/books/{collection_name}")
async def delete_book(collection_name: str):
    """Delete a specific book"""
    try:
        success = delete_book_collection(collection_name)
        if success:
            return {"status": "success", "message": f"Book {collection_name} deleted successfully"}
        else:
            raise HTTPException(status_code=500, detail="Failed to delete book")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/books")
async def delete_all_books():
    """Delete all books"""
    try:
        success = delete_all_collections()
        if success:
            return {"status": "success", "message": "All books deleted successfully"}
        else:
            raise HTTPException(status_code=500, detail="Failed to delete all books")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, ws_max_size=1024*1024)
