import streamlit as st
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_community.vectorstores import Chroma
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import Docx2txtLoader
import tempfile
import os
import docx2txt
import chromadb
from RealtimeTTS import TextToAudioStream, SystemEngine, ElevenlabsEngine, CoquiEngine
from dotenv import load_dotenv
from deepgram_tts import TextToSpeech
from micro import listen

from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

import logging
from typing import List, Dict

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

uploaded_file = None

load_dotenv()

# DEBUGGING: Print all session state keys
def print_session_state_keys():
    print("Current Session State Keys:")
    for key in st.session_state.keys():
        print(f"{key}: {type(st.session_state[key])}")

# Comprehensive and explicit session state initialization
def ensure_session_state():
    """Ensure all required session state variables exist with explicit initialization."""
    # List of all required session state keys with their default values
    default_states = {
        'rag_chain': None,
        'chat_history': [],
        'vectorstore': None,
        'temperature': 0.0,
        'prompt_type': "SIMPLE",
        'quiz_started': False,
        'tts_enabled': True,
        'book_title': None,
        'reader_name': "Lucy",
        'first_run': True,
        'uploaded_file': None
    }

    # Explicitly set each state if not already present
    for key, default_value in default_states.items():
        if key not in st.session_state:
            print(f"INITIALIZING: {key}")
            st.session_state[key] = default_value

    # DEBUGGING: Print all keys after initialization
    print_session_state_keys()

# Call state initialization at the beginning
ensure_session_state()

def debug_session_state():
    """Additional debugging function to thoroughly inspect session state."""
    st.write("### Session State Debug Info ###")
    st.write(f"RAG Chain: {'Initialized' if st.session_state.get('rag_chain') is not None else 'Not Initialized'}")
    st.write(f"Vectorstore: {'Initialized' if st.session_state.get('vectorstore') is not None else 'Not Initialized'}")
    st.write(f"Chat History Length: {len(st.session_state.get('chat_history', []))}")

def create_rag(uploaded_file, chunk_size, chunk_overlap):
    try:
        # Process documents
        doc_splits = process_document(uploaded_file, chunk_size, chunk_overlap)

        # Get the title from the first line of the document
        book_title = doc_splits[0].page_content.split('\n')[0].strip()
        st.session_state['book_title'] = book_title

        # Get embedding model
        embedding_function = get_embeddings_model()

        # CRITICAL: Use dictionary-style session state update
        st.session_state['vectorstore'] = Chroma.from_documents(
            documents=doc_splits,
            embedding=embedding_function,
            persist_directory=PERSIST_DIR
        )

        # Rest of your existing create_rag function remains the same...
        retriever = st.session_state['vectorstore'].as_retriever(
            search_kwargs={"k": 3}
        )

        # ... (rest of the function stays the same)

        return rag_chain

    except Exception as e:
        st.error(f"Error in create_rag: {e}")
        print(f"Detailed Error: {e}")
        raise

def crunchPrompt(prompt, debug=True):
    # EXTENSIVE DEBUGGING
    print("\n===== PROMPT PROCESSING DEBUG =====")
    print(f"Attempting to process prompt: {prompt}")
    print_session_state_keys()

    # CRITICAL: Explicit check and initialization
    if 'vectorstore' not in st.session_state or st.session_state['vectorstore'] is None:
        st.error("Vectorstore not initialized. Please upload a document and create RAG system.")
        return

    # Rest of the function remains mostly the same
    st.session_state['chat_history'].append({"role": "user", "content": prompt})

    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner('Thinking...'):
            try:
                # DEBUGGING: Print retrieval details
                print("Invoking RAG Chain...")

                response = st.session_state['rag_chain'].invoke({
                    "question": prompt,
                    "chat_history": st.session_state['chat_history']
                })

                response = response.strip()
                print(f"RAG Response: {response}")

                st.session_state['chat_history'].append({"role": "assistant", "content": response})
                st.markdown(response)

            except Exception as e:
                response = f"Processing error: {str(e)}"
                st.error(response)
                print(f"RAG Processing Error: {e}")
                # Print extended error details
                import traceback
                traceback.print_exc()

# Modify your file uploader section to include debugging
if uploaded_file is not None:
    st.info("After changing the prompt type, click 'Create RAG' to rebuild the system.")
    if st.button('Create RAG'):
        with st.spinner('Creating RAG system...'):
            try:
                # Ensure clean state
                if 'vectorstore' in st.session_state:
                    st.session_state['vectorstore'] = None

                # Create RAG
                st.session_state['rag_chain'] = create_rag(
                    uploaded_file,
                    chunk_size,
                    chunk_overlap
                )
                st.success('RAG system created successfully!')

                # OPTIONAL: Show debug information
                debug_session_state()

            except Exception as e:
                st.error(f'Error creating RAG system: {str(e)}')
                print(f"RAG Creation Error: {e}")
                import traceback
                traceback.print_exc()
                st.session_state['vectorstore'] = None
                st.session_state['rag_chain'] = None

# Streamlit UI
if st.session_state.book_title:
    st.title(f"📚 {st.session_state.book_title} 🎯")
else:
    st.title("📚")

# Sidebar
with st.sidebar:
    st.session_state.reader_name = st.text_input(
            "Reader's Name",
            value=st.session_state.reader_name,
            help="Enter the name of the reader"
        )

    prompt_type = st.radio(
            "Select Chat Mode",
            options=["SIMPLE", "QUIZ"],
            index=0 if st.session_state.prompt_type == "SIMPLE" else 1,
            help="Choose between a friendly chat or an interactive quiz about the book"
        )
    st.session_state.prompt_type = prompt_type

    st.markdown("### Speech Settings")
    if st.checkbox(
        "🔊 Enable Text-to-Speech",
        value=st.session_state.tts_enabled,
        key="tts_toggle",
        help="Turn text-to-speech on/off"
    ):
        tts = TextToSpeech()
    st.session_state.tts_enabled = st.session_state.tts_toggle

    st.header("📖 Story Upload")
    uploaded_file = st.file_uploader(
        "Choose your book",
        type=['pdf', 'txt'],
        key="book_uploader"
    )

    # Track uploaded file in session state
    if uploaded_file is not None:
        st.session_state.uploaded_file = uploaded_file
        st.info(f"File uploaded: {uploaded_file.name}")

    # Text splitting parameters
    st.markdown("### Chunk Settings")
    chunk_size = st.slider("Chunk Size", 100, 2000, 1000)
    chunk_overlap = st.slider("Chunk Overlap", 0, 500, 100)

    if uploaded_file is not None:
        st.info("After changing the prompt type, click 'Create RAG' to rebuild the system with the new prompt.")
        if st.button('Create RAG'):
            with st.spinner('Creating RAG system...'):
                try:
                    if st.session_state.vectorstore is not None:
                        cleanup_chroma()

                    st.session_state.rag_chain = create_rag(
                        uploaded_file,
                        chunk_size,
                        chunk_overlap
                    )
                    st.success('RAG system created successfully!')
                    st.rerun()
                except Exception as e:
                    st.error(f'Error creating RAG system: {str(e)}')
                    # Ensure vectorstore is reset on error
                    st.session_state.vectorstore = None
                    st.session_state.rag_chain = None

    st.session_state.temperature = st.slider(
        "Temperature",
        min_value=0.0,
        max_value=1.0,
        value=st.session_state.temperature,
        step=0.1
    )

    # Clear database button
    if st.button('Clear Database'):
        if os.path.exists(PERSIST_DIR):
            import shutil
            shutil.rmtree(PERSIST_DIR)
            os.makedirs(PERSIST_DIR)
        cleanup_chroma()
        st.session_state.vectorstore = None
        st.session_state.rag_chain = None
        st.session_state.chat_history = []
        st.session_state.quiz_started = False
        st.session_state.book_title = None  # Reset book title
        st.session_state.reader_name = "Lucy"
        st.session_state.first_run = True
        st.cache_data.clear()
        st.cache_resource.clear()
        st.success('Database and caches cleared!')

# Chat interface
for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        # Play audio for assistant messages
        if message["role"] == "assistant" and st.session_state.tts_enabled and st.session_state.first_run:
            # play(message["content"])
            tts.play(message["content"])
            st.session_state.first_run = False

if prompt := st.chat_input("Ask a question"):
    if st.session_state.rag_chain is None:
        st.error("Please upload a document and create RAG system first!")
    else:
        crunchPrompt(prompt, debug=True)
elif st.button("Use Microphone"):
    if st.session_state.rag_chain is None:
        st.error("Please upload a document and create RAG system first!")
    else:
        # Call the listening function when the button is clicked
        st.write("Listening... please speak now...")
        prompt = listen(stop_keyword="full stop")
        crunchPrompt(prompt, debug=True)
        # st.write(f"You said: {prompt}")
