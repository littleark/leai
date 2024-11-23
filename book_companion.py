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
# from RealtimeTTS import TextToAudioStream, SystemEngine, ElevenlabsEngine, CoquiEngine
from dotenv import load_dotenv
# from tts_utils import initialize_tts, play
from deepgram_tts import TextToSpeech
from micro import listen
from typing import List, Dict
import numpy as np
from langchain_core.documents import Document
from sklearn.metrics.pairwise import cosine_similarity
import logging


# Configure logging
# logging.basicConfig(level=logging.DEBUG)
# logger = logging.getLogger(__name__)

load_dotenv()

# elevenlabs_api_key = os.getenv("ELEVENLABS_API_KEY")

MAX_HISTORY_LENGTH = 4

# Initialize session state variables
if 'rag_created' not in st.session_state:
    st.session_state.rag_created = False

if 'rag_chain' not in st.session_state:
    st.session_state.rag_chain = None

if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

if 'vectorstore' not in st.session_state:
    st.session_state.vectorstore = None

if 'temperature' not in st.session_state:
    st.session_state.temperature = 0.0

if 'prompt_type' not in st.session_state:
    st.session_state.prompt_type = "SIMPLE"

if 'tts_enabled' not in st.session_state:
    st.session_state.tts_enabled = True

if 'book_title' not in st.session_state:
    st.session_state.book_title = None

if 'reader_name' not in st.session_state:
    st.session_state.reader_name = "Lucy"

if 'first_run' not in st.session_state:
    st.session_state.first_run = True

# Create a persistent directory for the database
PERSIST_DIR = os.path.join(os.getcwd(), 'db')
if not os.path.exists(PERSIST_DIR):
    os.makedirs(PERSIST_DIR)

@st.cache_data
def process_document(uploaded_file, chunk_size=800, chunk_overlap=100):
    """Process and split the uploaded document using RecursiveCharacterTextSplitter."""
    print(f"Processing file: {uploaded_file.name} of type: {uploaded_file.type}")
    print(f"Chunk size: {chunk_size}, chunk overlap: {chunk_overlap}")

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_file_path = os.path.join(temp_dir, uploaded_file.name)

        with open(temp_file_path, 'wb') as f:
            f.write(uploaded_file.getvalue())

        try:
            # Load document based on file type
            if uploaded_file.type == "application/pdf":
                loader = PyPDFLoader(temp_file_path)
                docs = loader.load()
                # Configure text splitter for narrative text
                text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=chunk_size,
                    chunk_overlap=chunk_overlap,
                    length_function=len,
                    # Add separators specific to book content
                    separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""],
                    # Keep sentences together where possible
                    keep_separator=True
                )
                doc_splits = text_splitter.split_documents(docs)

            elif uploaded_file.type in ["text/plain"]:
                with open(temp_file_path, 'r') as f:
                    text = f.read()
                text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=chunk_size,
                    chunk_overlap=chunk_overlap,
                    length_function=len,
                    separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""],
                    keep_separator=True
                )
                doc_splits = text_splitter.create_documents([text])

            else:
                raise ValueError(f"Unsupported file type: {uploaded_file.type}")

            # Add debug information
            print(f"Document split into {len(doc_splits)} chunks")
            print(f"Average chunk size: {sum(len(chunk.page_content) for chunk in doc_splits) / len(doc_splits):.0f} characters")
            # Print a sample chunk for verification
            if doc_splits:
                print("\nSample chunk:")
                print("-" * 50)
                print(doc_splits[0].page_content[:200] + "...")
                print("-" * 50)

            return doc_splits

        except Exception as e:
            print(f"Error processing document: {str(e)}")
            raise e

@st.cache_resource
def get_embeddings_model():
    """Initialize and return the embedding model."""
    return OllamaEmbeddings(
        model='nomic-embed-text',
        base_url="http://localhost:11434"
    )

# Enhanced prompt template with strict instructions
def create_chat_prompt():
    template = """CRITICAL CONTEXT GUIDELINES:
    - You are talking to an 8-year-old about a specific book
    - ONLY use information DIRECTLY from the provided context
    - If a question CANNOT be answered using the context, respond: "I don't know that from the book."
    - Be concise and child-friendly
    - NEVER invent or guess information
    - After the answer ask a follow-up question

    Context: {context}

    Chat History: {chat_history}

    Current Question: {question}

    RESPONSE RULES:
    - Include the name of the kid {reader_name} in your response but NEVER start with "Hi {reader_name}" or any greeting
    - Maximum 2 sentences for the answer
    - After the answer ask a simple follow-up question
    - If unsure, admit you don't know
    - Do not use greetings or phrases like "Hey", "Hello", "Hi there"
    - Get straight to answering the question

    Answer: """
    return ChatPromptTemplate.from_template(template)

def cleanup_chroma():
    print("""Clean up ChromaDB resources.""")
    try:
        if st.session_state.vectorstore is not None:
            # Attempt to reset the client
            st.session_state.vectorstore._client.reset()
            # Clear the vectorstore reference
            st.session_state.vectorstore = None
    except Exception as e:
        print(f"Error during ChromaDB cleanup: {e}")
        st.session_state.vectorstore = None

def format_chat_history(chat_history, max_length=4):
    """
    Format chat history for RAG context with optional length limitation.

    Args:
    - chat_history (list): List of chat messages
    - max_length (int): Maximum number of recent messages to include

    Returns:
    - Formatted string representation of chat history
    """
    # If chat history is empty, return empty string
    if not chat_history:
        return ""

    # Limit to most recent messages
    recent_history = chat_history[-max_length * 2:]

    # Format history
    formatted_history = []
    for message in recent_history:
        # Ensure the message has 'role' and 'content' keys
        role = message.get('role', 'unknown').title()
        content = message.get('content', '')

        # Truncate very long messages
        if len(content) > 200:
            content = content[:200] + "..."

        formatted_history.append(f"{role}: {content}")

    # Join formatted messages
    return "\n".join(formatted_history)

def create_rag(uploaded_file):
    try:
        chunk_size = 800  # Fixed optimal chunk size
        chunk_overlap = 100  # Fixed optimal overlap

        # Capture session state values at creation time
        current_reader_name = st.session_state.get('reader_name', 'Lucy')

        # Process documents
        doc_splits = process_document(uploaded_file, chunk_size, chunk_overlap)

        # Get the title from the first line of the document
        book_title = doc_splits[0].page_content.split('\n')[0].strip()
        st.session_state.book_title = book_title

        # Get embedding model
        embedding_function = get_embeddings_model()

        # Create vector store with enhanced retrieval
        vectorstore = Chroma.from_documents(
            documents=doc_splits,
            embedding=embedding_function,
            persist_directory=PERSIST_DIR
        )

        # Store in session state after creation
        st.session_state.vectorstore = vectorstore

        # Closure over vectorstore instance
        def custom_retriever(query, k=5):
            """
            Custom retriever with multiple relevance checks
            1. Semantic similarity
            2. Keyword matching
            3. Similarity score filtering
            """
            print('custom_retriever', query)
            # Get embeddings for query and documents
            query_embedding = embedding_function.embed_query(query)

            # Use captured vectorstore instance instead of accessing from session state
            retriever = vectorstore.as_retriever(
                search_kwargs={"k": k}
            )
            retrieved_docs = retriever.invoke(query)
            print('retrieved_docs', retrieved_docs)

            # Calculate semantic similarity
            doc_embeddings = [
                embedding_function.embed_query(doc.page_content)
                for doc in retrieved_docs
            ]

            # Compute cosine similarities
            similarities = cosine_similarity(
                [query_embedding],
                doc_embeddings
            )[0]

            # Filter documents based on similarity threshold
            filtered_docs = [
                doc for doc, sim in zip(retrieved_docs, similarities)
                if sim > 0.3  # Adjust this threshold as needed
            ]

            # Logging for debugging
            print("Document Retrieval Debug:")
            for doc, sim in zip(retrieved_docs, similarities):
                print(f"Similarity: {sim:.4f} - Content Preview: {doc.page_content[:200]}...")

            return filtered_docs

        # Get LLM model
        model_local = ChatOllama(
            model="llama3.2",
            base_url="http://localhost:11434",
            temperature=0.0,
            num_predict=150,
            top_k=10,
            top_p=0.1,
        )

        def format_docs(docs):
            return "\n\n".join(doc.page_content for doc in docs)

        # Modified RAG chain using captured vectorstore
        rag_chain = (
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
                "reader_name": lambda x: current_reader_name
            }
            | create_chat_prompt()
            | model_local
            | StrOutputParser()
        )

        # Store the chain in session state
        st.session_state.rag_chain = rag_chain
        st.session_state.rag_created = True

        # Generate welcome message with an interesting fact
        welcome_prompt = f"Tell me an interesting fact from this book that would excite an 8-year-old reader."
        welcome_response = rag_chain.invoke({
            "question": welcome_prompt,
            "chat_history": [],
            "reader_name": current_reader_name
        })

        # Store welcome message in chat history
        st.session_state.chat_history.append({"role": "assistant", "content": welcome_response})

        return rag_chain

    except Exception as e:
        print(f"Error creating RAG system: {str(e)}")
        raise e

def enhanced_custom_retriever(query: str, vectorstore, embedding_function, k: int = 5, debug: bool = True) -> List[Document]:
    """
    Enhanced retriever with detailed debugging and verification
    """
    # Get query embedding
    query_embedding = embedding_function.embed_query(query)

    # Get retrieved documents
    retriever = vectorstore.as_retriever(search_kwargs={"k": k})
    retrieved_docs = retriever.invoke(query)

    # Calculate semantic similarity scores
    doc_embeddings = [embedding_function.embed_query(doc.page_content) for doc in retrieved_docs]
    similarities = cosine_similarity([query_embedding], doc_embeddings)[0]

    # Create detailed retrieval report
    retrieval_report = {
        "query": query,
        "num_docs_retrieved": len(retrieved_docs),
        "similarity_scores": similarities,
        "retrieved_contexts": [
            {
                "content": doc.page_content[:200] + "...",
                "similarity": sim,
                "metadata": doc.metadata
            }
            for doc, sim in zip(retrieved_docs, similarities)
        ]
    }

    if debug:
        st.sidebar.expander("🔍 RAG Retrieval Debug").write(retrieval_report)

    # Filter documents based on similarity threshold
    filtered_docs = [
        doc for doc, sim in zip(retrieved_docs, similarities)
        if sim > 0.3
    ]

    return filtered_docs, retrieval_report

def verify_rag_response(response: str, retrieval_report: Dict, debug: bool = True) -> Dict:
    """Enhanced verification using multiple methods"""
    verification_results = {
        "response_length": len(response),
        "num_contexts_used": len(retrieval_report["retrieved_contexts"]),
        "avg_similarity": np.mean(retrieval_report["similarity_scores"]),
        "verification_metrics": {}
    }

    # 1. Direct text overlap
    combined_contexts = " ".join([ctx["content"] for ctx in retrieval_report["retrieved_contexts"]])
    verification_results["verification_metrics"]["exact_overlap"] = any(
        phrase in combined_contexts
        for phrase in response.split(". ")
    )

    # 2. Key term overlap
    def extract_key_terms(text):
        # Simple key term extraction (could be enhanced with NLP)
        return set(word.lower() for word in text.split()
                  if len(word) > 3 and word.isalnum())

    response_terms = extract_key_terms(response)
    context_terms = extract_key_terms(combined_contexts)
    term_overlap_ratio = len(response_terms.intersection(context_terms)) / len(response_terms)

    verification_results["verification_metrics"]["term_overlap_ratio"] = term_overlap_ratio

    # 3. High similarity check
    verification_results["verification_metrics"]["high_similarity_context"] = \
        any(sim > 0.7 for sim in retrieval_report["similarity_scores"])

    # Combined confidence score
    verification_results["rag_confidence"] = (
        (term_overlap_ratio * 0.4) +
        (float(verification_results["verification_metrics"]["exact_overlap"]) * 0.3) +
        (float(verification_results["verification_metrics"]["high_similarity_context"]) * 0.3)
    )

    if debug:
        st.sidebar.expander("🎯 RAG Verification").write(verification_results)

        # Visual confidence indicator
        confidence = verification_results["rag_confidence"]
        if confidence > 0.7:
            st.sidebar.success(f"🎯 High RAG confidence: {confidence:.2f}")
        elif confidence > 0.4:
            st.sidebar.warning(f"⚠️ Medium RAG confidence: {confidence:.2f}")
        else:
            st.sidebar.error(f"❌ Low RAG confidence: {confidence:.2f}")

    return verification_results

def crunchPrompt(prompt: str, debug: bool = True):
    # Add user message to history
    st.session_state.chat_history.append({"role": "user", "content": prompt})

    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner('Thinking...'):
            try:
                # Get retrieved documents and debug info
                retrieved_docs, retrieval_report = enhanced_custom_retriever(
                    prompt,
                    st.session_state.vectorstore,
                    get_embeddings_model(),
                    debug=debug
                )

                # Generate response
                response = st.session_state.rag_chain.invoke({
                    "question": prompt,
                    "chat_history": st.session_state.chat_history
                })

                # Verify response
                verification = verify_rag_response(response, retrieval_report, debug=debug)

                # Display warning if confidence is low
                if verification["rag_confidence"] < 0.4:
                    st.warning("⚠️ This response might not be fully grounded in the document context.")

                st.markdown(response)
                st.session_state.chat_history.append({"role": "assistant", "content": response})

                if st.session_state.tts_enabled:
                    tts.play(response)

            except Exception as e:
                st.error(f"Error processing question: {str(e)}")

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

    if prompt := st.button("Use Microphone"):
        if st.session_state.rag_chain is None:
            st.error("Please upload a document and create RAG system first!")
        else:
            st.write("Listening... please speak now...")
            prompt = listen(stop_keyword="full stop")
            crunchPrompt(prompt, debug=True)

    st.markdown("### Speech Settings")
    if st.checkbox(
        "🔊 Enable Text-to-Speech",
        value=st.session_state.tts_enabled,
        key="tts_toggle",
        help="Turn text-to-speech on/off"
    ):
        tts = TextToSpeech()
    st.session_state.tts_enabled = st.session_state.tts_toggle

    if st.session_state.rag_created:
        st.success('RAG system created successfully!')

    # Only show file upload and Create RAG if RAG hasn't been created yet
    if not st.session_state.rag_created:
        st.header("📖 Story Upload")
        uploaded_file = st.file_uploader("Choose your book", type=['pdf', 'txt'])

        if uploaded_file is not None:
            if st.button('Create RAG'):
                with st.spinner('Creating RAG system...'):
                    try:
                        if st.session_state.vectorstore is not None:
                            print('Because vectorstore is not None, cleaning up ChromaDB resources.')
                            cleanup_chroma()

                        print('Creating RAG system with rag_chain!!!!')
                        rag_chain = create_rag(uploaded_file)

                        # Display the welcome message right after creation
                        if st.session_state.chat_history:
                            with st.chat_message("assistant"):
                                st.markdown(st.session_state.chat_history[-1]["content"])
                                if st.session_state.tts_enabled:
                                    tts.play(st.session_state.chat_history[-1]["content"])

                        st.rerun()

                    except Exception as e:
                        st.error(f'Error creating RAG system: {str(e)}')
                        st.session_state.vectorstore = None
                        st.session_state.rag_chain = None
                        st.session_state.rag_created = False

if prompt := st.chat_input("Ask a question"):
    if st.session_state.rag_chain is None:
        st.error("Please upload a document and create RAG system first!")
    else:
        crunchPrompt(prompt, debug=True)
