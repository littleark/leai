import streamlit as st
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_community.vectorstores import Chroma
from langchain_community import embeddings
from langchain_community.chat_models import ChatOllama
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.document_loaders import Docx2txtLoader
from langchain_core.documents import Document
import tempfile
import os
import docx2txt
import chromadb
from sentence_transformers import SentenceTransformer, util
import numpy as np
import pprint

# Initialize session state variables
if 'rag_chain' not in st.session_state:
    st.session_state.rag_chain = None

if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

if 'vectorstore' not in st.session_state:
    st.session_state.vectorstore = None

if 'temperature' not in st.session_state:
    st.session_state.temperature = 0.2

# Create a persistent directory for the database
PERSIST_DIR = os.path.join(os.getcwd(), 'db')
if not os.path.exists(PERSIST_DIR):
    os.makedirs(PERSIST_DIR)

def clear_all():
    """Clear all caches, database, and session states."""
    # Clear ChromaDB
    if os.path.exists(PERSIST_DIR):
        import shutil
        shutil.rmtree(PERSIST_DIR)
        os.makedirs(PERSIST_DIR)
    cleanup_chroma()

    # Clear session states
    st.session_state.vectorstore = None
    st.session_state.rag_chain = None
    st.session_state.chat_history = []

    # Clear all caches
    st.cache_data.clear()
    st.cache_resource.clear()

@st.cache_resource
def load_sbert_model():
    """Load and cache SBERT model."""
    return SentenceTransformer("all-MiniLM-L6-v2")

def split_to_sentences(text):
    """Split text into sentences."""
    return [s.strip() + "." for s in text.split(".") if s.strip()]

@st.cache_data
def split_semantically(text, model, similarity_threshold=0.75, max_chunk_size=7500):
    """Split text into semantic chunks using SBERT embeddings."""
    sentences = split_to_sentences(text)

    if not sentences:
        return []

    # Generate embeddings for all sentences
    embeddings = model.encode(sentences)

    # Initialize chunks
    chunks = []
    current_chunk = [sentences[0]]
    current_length = len(sentences[0])

    for i in range(1, len(sentences)):
        # Check semantic similarity
        sim = util.cos_sim(embeddings[i-1], embeddings[i]).item()
        next_sentence_length = len(sentences[i])

        # Conditions for starting a new chunk:
        # 1. Low semantic similarity OR
        # 2. Chunk would exceed max size
        if sim < similarity_threshold or (current_length + next_sentence_length) > max_chunk_size:
            chunks.append(" ".join(current_chunk))
            current_chunk = [sentences[i]]
            current_length = next_sentence_length
        else:
            current_chunk.append(sentences[i])
            current_length += next_sentence_length

    # Add the last chunk if it exists
    if current_chunk:
        chunks.append(" ".join(current_chunk))

    pprint.pp(chunks)
    return chunks

@st.cache_resource
def get_embeddings_model():
    """Initialize and return the embedding model."""
    return embeddings.ollama.OllamaEmbeddings(model='nomic-embed-text')

@st.cache_resource
def get_llm_model(_temperature):
    """Initialize and return the LLM model."""
    return ChatOllama(
        model="llama3.1",
        base_url="http://localhost:11434",
        temperature=_temperature
    )

@st.cache_data
def create_rag_prompt():
    """Create and return a minimal RAG prompt template."""
    template = """Based on this context:

    {context}

    Answer this question only, no extra information. Be concise: {question}

    If the information isn't in the context, say "I cannot find this information in the document."
    Keep your answer focused and concise.
    """
    return ChatPromptTemplate.from_template(template)

@st.cache_data
def create_rag_prompt2():
    """Create and return the RAG prompt template for legal document analysis."""
    template = """You are a legal expert assistant analyzing legal documents talking with other lawyers. Use the following context to answer the question:

    Context:
    {context}

    Guidelines:
    1. Do not include any technical metadata or document structure information in your response
    2. Base your answers strictly on the content provided in the context
    3. If the information isn't explicitly stated in the context, respond with "I cannot find this information in the document"
    4. When referencing specific parts of the document, cite them directly
    5. If legal terms are used, do not explain them in clear, simple language. Leave them as they are.
    6. If there are any conditions or exceptions mentioned in the text, make sure to include them
    7. For questions about dates, deadlines, or time periods, be very specific
    8. If asked about parties' obligations or rights, list them clearly and completely
    9. When discussing procedures or processes, present them in a step-by-step format
    10. If there are any ambiguities in the text, acknowledge them explicitly
    11. For numerical values (amounts, percentages, etc.), quote them exactly as they appear

    Question: {question}

    Please provide a clear, structured response that:
    - Directly addresses the question
    - Keep it concise
    - Cites relevant sections when applicable
    - Explains any legal terminology used
    - Highlights important conditions or exceptions
    - Maintains accuracy to the source document
    - Do not explain legal terms as it will be read by other lawyers
    """
    return ChatPromptTemplate.from_template(template)

def cleanup_chroma():
    """Clean up ChromaDB resources."""
    if st.session_state.vectorstore is not None:
        try:
            st.session_state.vectorstore._collection.delete()
            st.session_state.vectorstore = None
        except Exception as e:
            print(f"Error cleaning up ChromaDB: {str(e)}")
            pass

@st.cache_data
def process_document(uploaded_file, similarity_threshold, max_chunk_size):
    """Process and split the uploaded document using SBERT-based semantic splitting."""
    print(f"Processing file: {uploaded_file.name} of type: {uploaded_file.type}")

    # Load SBERT model
    model = load_sbert_model()

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_file_path = os.path.join(temp_dir, uploaded_file.name)

        with open(temp_file_path, 'wb') as f:
            f.write(uploaded_file.getvalue())

        try:
            # Load document based on file type
            if uploaded_file.type == "application/pdf":
                loader = PyPDFLoader(temp_file_path)
                docs = loader.load()
                # Combine all pages into one text
                full_text = " ".join([doc.page_content for doc in docs])
            elif uploaded_file.type in ["application/vnd.openxmlformats-officedocument.wordprocessingml.document"]:
                full_text = docx2txt.process(temp_file_path)
            elif uploaded_file.type == "application/msword":
                print("Loading DOC document...")
                st.warning("Warning: .doc files might not be fully supported. Consider converting to .docx")
                full_text = docx2txt.process(temp_file_path)
            elif uploaded_file.type == "text/plain":
                with open(temp_file_path, 'r') as f:
                    full_text = f.read()
            else:
                raise ValueError(f"Unsupported file type: {uploaded_file.type}")

            # Split text semantically using SBERT
            chunks = split_semantically(full_text, model,
                                     similarity_threshold=similarity_threshold,
                                     max_chunk_size=max_chunk_size)

            # Convert chunks to Document objects
            doc_splits = [
                Document(
                    page_content=chunk,
                    metadata={"source": uploaded_file.name, "chunk_id": i}
                )
                for i, chunk in enumerate(chunks)
            ]

            print(f"Document split into {len(doc_splits)} semantic chunks")

            return doc_splits

        except Exception as e:
            print(f"Error processing document: {str(e)}")
            raise e

@st.cache_data
def format_chat_history(chat_history):
    """Format chat history for download."""
    return "\n\n".join([
        f"{msg['role'].upper()}: {msg['content']}"
        for msg in chat_history
    ])

def format_docs(docs):
    """Format documents to remove metadata and return only the content."""
    return "\n\n".join(doc.page_content for doc in docs)

def create_rag(uploaded_file, similarity_threshold, max_chunk_size):
    """Create the RAG system with semantic document splitting."""
    try:
        # Process documents using cached function with semantic splitting
        doc_splits = process_document(uploaded_file, similarity_threshold, max_chunk_size)

        # Get embedding model using cached function
        embedding_function = get_embeddings_model()

        # Create vector store with persistence
        st.session_state.vectorstore = Chroma.from_documents(
            documents=doc_splits,
            embedding=embedding_function,
            persist_directory=PERSIST_DIR,
            client_settings=chromadb.config.Settings(
                anonymized_telemetry=False,
                is_persistent=True
            )
        )
        st.session_state.vectorstore.persist()

        # Setup retriever and model
        retriever = st.session_state.vectorstore.as_retriever(
            search_kwargs={"k": 3}
        )
        model_local = get_llm_model(st.session_state.temperature)

        # Create RAG prompt using cached function
        rag_prompt = create_rag_prompt()

        # Create RAG chain with document formatting
        rag_chain = (
            {
                "context": retriever | format_docs,  # Add formatting step here
                "question": RunnablePassthrough()
            }
            | rag_prompt
            | model_local
            | StrOutputParser()
        )

        return rag_chain

    except Exception as e:
        print(f"Error creating RAG system: {str(e)}")
        raise e

# Add title at the top of the app
st.title("Document Chat Assistant")

# Sidebar
with st.sidebar:
    st.header("Document Upload")

    # Keep track of the previous file
    if 'previous_file_name' not in st.session_state:
        st.session_state.previous_file_name = None

    uploaded_file = st.file_uploader("Choose a document", type=['pdf', 'txt', 'doc', 'docx'])

    # Check if a new file is uploaded
    if uploaded_file is not None and (
        st.session_state.previous_file_name is None or
        uploaded_file.name != st.session_state.previous_file_name
    ):
        # Clear everything when a new file is uploaded
        clear_all()
        st.session_state.previous_file_name = uploaded_file.name
        st.warning('New file detected. Database and caches cleared!')

    # Rest of the sidebar code...
    # Semantic Splitting Settings
    st.markdown("### Semantic Splitting Settings")
    similarity_threshold = st.slider(
        "Semantic Similarity Threshold",
        min_value=0.0,
        max_value=1.0,
        value=0.75,
        step=0.05,
        help="Higher values create more granular chunks"
    )

    max_chunk_size = st.number_input(
        "Maximum Chunk Size (characters)",
        min_value=500,
        max_value=10000,
        value=7500,
        step=500,
        help="Maximum number of characters per chunk"
    )

    if uploaded_file is not None:
        if st.button('Create RAG'):
            with st.spinner('Creating RAG system...'):
                try:
                    st.session_state.rag_chain = create_rag(
                        uploaded_file,
                        similarity_threshold,
                        max_chunk_size
                    )
                    st.success('RAG system created successfully!')
                except Exception as e:
                    st.error(f'Error creating RAG system: {str(e)}')

    # Add manual clear button
    if st.button('Clear Database'):
        clear_all()
        st.success('Database and caches cleared!')

    # Download chat history button
    if st.session_state.chat_history:
        chat_text = format_chat_history(st.session_state.chat_history)
        st.download_button(
            label="Download Chat History",
            data=chat_text,
            file_name="chat_history.txt",
            mime="text/plain"
        )

# Main chat interface
# Display chat history
for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
if prompt := st.chat_input("Ask a question about your document"):
    if st.session_state.rag_chain is None:
        st.error("Please upload a document and create RAG system first!")
    else:
        # Add user message to chat history
        st.session_state.chat_history.append({"role": "user", "content": prompt})

        with st.chat_message("user"):
            st.markdown(prompt)

        # Get assistant response
        with st.chat_message("assistant"):
            with st.spinner('Thinking...'):
                response = st.session_state.rag_chain.invoke(prompt)
                st.session_state.chat_history.append({"role": "assistant", "content": response})
                st.markdown(response)
