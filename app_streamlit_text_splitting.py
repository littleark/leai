import streamlit as st
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_community.vectorstores import Chroma
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import Docx2txtLoader
from langchain_core.documents import Document
import tempfile
import os
import docx2txt
import chromadb
import spacy
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Initialize session state variables
if 'rag_chain' not in st.session_state:
    st.session_state.rag_chain = None

if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

if 'vectorstore' not in st.session_state:
    st.session_state.vectorstore = None

if 'temperature' not in st.session_state:
    st.session_state.temperature = 0.25

# Create a persistent directory for the database
PERSIST_DIR = os.path.join(os.getcwd(), 'db')
if not os.path.exists(PERSIST_DIR):
    os.makedirs(PERSIST_DIR)

# Add the split_with_transformer function
def split_with_transformer(text, nlp, sentence_transformer, similarity_threshold=0.75, max_chunk_size=2000):
    """Split text using SpaCy and Sentence Transformer."""
    doc = nlp(text)
    sentences = [sent.text.strip() for sent in doc.sents if sent.text.strip()]

    if not sentences:
        return []

    embeddings = sentence_transformer.encode(sentences)

    chunks = []
    current_chunk = [sentences[0]]
    current_length = len(sentences[0])

    for i in range(1, len(sentences)):
        sim = cosine_similarity([embeddings[i - 1]], [embeddings[i]])[0][0]
        next_sentence_length = len(sentences[i])

        if sim < similarity_threshold or (current_length + next_sentence_length) > max_chunk_size:
            chunks.append(" ".join(current_chunk))
            current_chunk = [sentences[i]]
            current_length = next_sentence_length
        else:
            current_chunk.append(sentences[i])
            current_length += next_sentence_length

    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks

@st.cache_resource
def load_nlp_models():
    """Load and cache SpaCy and Sentence Transformer models."""
    nlp = spacy.load("en_core_web_sm")
    sentence_transformer = SentenceTransformer("all-MiniLM-L6-v2")
    return nlp, sentence_transformer

@st.cache_data
def process_document(uploaded_file, similarity_threshold):
    """Process and split the uploaded document using transformer-based splitting."""
    print(f"Processing file: {uploaded_file.name} of type: {uploaded_file.type}")

    # Load NLP models
    nlp, sentence_transformer = load_nlp_models()

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_file_path = os.path.join(temp_dir, uploaded_file.name)

        with open(temp_file_path, 'wb') as f:
            f.write(uploaded_file.getvalue())

        try:
            # Load document based on file type
            if uploaded_file.type == "application/pdf":
                print("Loading PDF document...")
                loader = PyPDFLoader(temp_file_path)
                docs = loader.load()
                full_text = " ".join([doc.page_content for doc in docs])
            elif uploaded_file.type in ["application/vnd.openxmlformats-officedocument.wordprocessingml.document"]:
                print("Loading DOCX document...")
                full_text = docx2txt.process(temp_file_path)
            elif uploaded_file.type == "application/msword":
                print("Loading DOC document...")
                st.warning("Warning: .doc files might not be fully supported. Consider converting to .docx")
                full_text = docx2txt.process(temp_file_path)
            elif uploaded_file.type == "text/plain":
                print("Loading text document...")
                with open(temp_file_path, 'r') as f:
                    full_text = f.read()
            else:
                raise ValueError(f"Unsupported file type: {uploaded_file.type}")

            # Split text using transformer-based splitting
            chunks = split_with_transformer(
                full_text,
                nlp,
                sentence_transformer,
                similarity_threshold=similarity_threshold
            )

            # Convert chunks to Document objects
            doc_splits = [
                Document(
                    page_content=chunk,
                    metadata={
                        "source": uploaded_file.name,
                        "chunk_id": i,
                        "similarity_threshold": similarity_threshold  # Add this for reference
                    }
                )
                for i, chunk in enumerate(chunks)
            ]

            print(f"Document split into {len(doc_splits)} semantic chunks")
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
@st.cache_resource
def get_llm_model(_temperature):
    """Initialize and return the LLM model."""
    return ChatOllama(
        model="llama3.1",
        base_url="http://localhost:11434",
        temperature=_temperature,
        timeout=120,    # keep timeout if needed
    )

@st.cache_data
def create_rag_prompt2():
    """Create and return the RAG prompt template for legal document analysis."""
    template = """You are an expert legal analyst. Using only the following context, provide a clear and detailed response to the question.

    CONTEXT:
    {context}

    QUESTION: {question}

    INSTRUCTIONS:
    1. Answer based solely on the provided context
    2. If the information isn't in the context, say "I cannot find this information in the document"
    3. Use clear, professional language
    4. Quote relevant parts of the document when appropriate
    5. Explain any legal terms you reference
    6. Present information in a structured, easy-to-read format

    RESPONSE:"""
    return ChatPromptTemplate.from_template(template)

@st.cache_data
def create_rag_prompt():
    """Create and return the RAG prompt template for legal document analysis."""
    template = """You are an expert legal analyst with extensive experience in interpreting legal documents. Analyze the following context to answer the question:

    DOCUMENT CONTEXT:
    {context}

    ANALYTICAL FRAMEWORK:
    1. Legal Interpretation:
       - Interpret terms according to their legal definition
       - Consider the document's jurisdiction and applicable legal framework
       - Identify key legal provisions and their implications

    2. Document Structure:
       - Reference specific sections, clauses, or paragraphs
       - Maintain the hierarchical structure of provisions
       - Connect related clauses when relevant

    3. Rights and Obligations:
       - Clearly distinguish between mandatory and optional provisions
       - Identify the parties involved and their respective roles
       - Specify conditions and prerequisites for obligations

    4. Temporal Aspects:
       - Note effective dates, deadlines, and time periods
       - Identify sequential requirements or procedures
       - Highlight any temporal conditions or limitations

    5. Compliance Requirements:
       - List specific compliance obligations
       - Note any reporting or documentation requirements
       - Identify consequences of non-compliance

    USER QUESTION: {question}

    RESPONSE GUIDELINES:
    - Begin with a direct answer to the question
    - Support your answer with specific references from the document
    - Explain technical or legal terms in plain language
    - Highlight any relevant conditions, exceptions, or limitations
    - If information is not found in the document, clearly state this
    - Where applicable, provide structured or step-by-step explanations
    - Maintain objectivity and accuracy to the source material

    Answer: """
    return ChatPromptTemplate.from_template(template)


def cleanup_chroma():
    """Clean up ChromaDB resources."""
    if st.session_state.vectorstore is not None:
        try:
            st.session_state.vectorstore._client.reset()
            st.session_state.vectorstore._client = None
        except:
            pass

def create_rag(uploaded_file, similarity_threshold):
    """Create the RAG system."""
    try:
        # Process documents using cached function with similarity threshold
        doc_splits = process_document(uploaded_file, similarity_threshold)

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

        # Setup retriever with only supported parameters
        retriever = st.session_state.vectorstore.as_retriever(
            search_kwargs={
                "k": 3  # Number of documents to retrieve
            }
        )
        model_local = get_llm_model(st.session_state.temperature)

        # Create RAG prompt using cached function
        rag_prompt = create_rag_prompt()

        # Create RAG chain with formatting
        def format_docs(docs):
            return "\n\n".join(doc.page_content for doc in docs)

        rag_chain = (
            {
                "context": lambda x: format_docs(retriever.get_relevant_documents(x)),
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

@st.cache_data
def format_chat_history(chat_history):
    """Format chat history for download."""
    return "\n\n".join([
        f"{msg['role'].upper()}: {msg['content']}"
        for msg in chat_history
    ])

# Add title at the top of the app
st.title("Legal Chat Assistant")

# Sidebar
with st.sidebar:
    st.header("Document Upload")
    uploaded_file = st.file_uploader("Choose a document", type=['pdf', 'txt', 'doc', 'docx'])

    # Add semantic splitting controls
    st.markdown("### Semantic Splitting Settings")
    similarity_threshold = st.slider(
        "Semantic Similarity Threshold",
        min_value=0.0,
        max_value=1.0,
        value=0.75,
        step=0.05,
        help="Higher values create more granular chunks"
    )

    if uploaded_file is not None:
        if st.button('Create RAG'):
            with st.spinner('Creating RAG system...'):
                try:
                    cleanup_chroma()
                    st.session_state.rag_chain = create_rag(
                        uploaded_file,
                        similarity_threshold
                    )
                    st.success('RAG system created successfully!')
                except Exception as e:
                    st.error(f'Error creating RAG system: {str(e)}')

    st.markdown("### Settings")
    st.session_state.temperature = st.slider(
        "LLM Temperature",
        min_value=0.0,
        max_value=1.0,
        value=st.session_state.temperature,
        step=0.1,
        help="Higher values make the output more creative but less focused"
    )

    if st.button('Clear Database'):
        if os.path.exists(PERSIST_DIR):
            import shutil
            shutil.rmtree(PERSIST_DIR)
            os.makedirs(PERSIST_DIR)
            cleanup_chroma()
        st.session_state.vectorstore = None
        st.session_state.rag_chain = None
        st.session_state.chat_history = []
        st.cache_data.clear()
        st.cache_resource.clear()
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
                # Clean up response if needed
                response = response.strip()
                if response.startswith('"') and response.endswith('"'):
                    response = response[1:-1]
                st.session_state.chat_history.append({"role": "assistant", "content": response})
                st.markdown(response)
