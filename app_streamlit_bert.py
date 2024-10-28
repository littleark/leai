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
from transformers import AutoTokenizer, AutoModel
import torch
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
    st.session_state.temperature = 0.2

# Create a persistent directory for the database
PERSIST_DIR = os.path.join(os.getcwd(), 'db')
if not os.path.exists(PERSIST_DIR):
    os.makedirs(PERSIST_DIR)

@st.cache_resource
def load_bert_model():
    """Load and cache BERT model and tokenizer."""
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    model = AutoModel.from_pretrained("bert-base-uncased")
    return tokenizer, model

def embed_text(text, tokenizer, model):
    """Generate BERT embeddings for text."""
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    with torch.no_grad():  # Disable gradient calculations for inference
        outputs = model(**inputs)
    embeddings = outputs.last_hidden_state[:, 0, :]
    return embeddings.numpy()

@st.cache_data
def split_semantically(text, _tokenizer, _model, similarity_threshold=0.5, max_chunk_size=7500):
    """Split text into semantic chunks using BERT embeddings."""
    # First, split into rough sentences
    sentences = [s.strip() + "." for s in text.split(".") if s.strip()]

    if not sentences:
        return []

    # Generate embeddings for all sentences
    embeddings = [embed_text(sentence, _tokenizer, _model) for sentence in sentences]

    # Initialize chunks
    chunks = []
    current_chunk = [sentences[0]]
    current_length = len(sentences[0])

    for i in range(1, len(sentences)):
        # Check semantic similarity
        sim = cosine_similarity(embeddings[i-1], embeddings[i])[0][0]
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

    print(chunks)

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
def create_rag_prompt2():
    """Create and return the RAG prompt template for legal document analysis."""
    template = """You are a legal expert assistant analyzing legal documents. Use the following context to answer the question:

    Context:
    {context}

    Guidelines:
    1. Base your answers strictly on the content provided in the context
    2. If the information isn't explicitly stated in the context, respond with "I cannot find this information in the document"
    3. When referencing specific parts of the document, cite them directly
    4. If legal terms are used, explain them in clear, simple language
    5. If there are any conditions or exceptions mentioned in the text, make sure to include them
    6. For questions about dates, deadlines, or time periods, be very specific
    7. If asked about parties' obligations or rights, list them clearly and completely
    8. When discussing procedures or processes, present them in a step-by-step format
    9. If there are any ambiguities in the text, acknowledge them explicitly
    10. For numerical values (amounts, percentages, etc.), quote them exactly as they appear

    Question: {question}

    Please provide a clear, structured response that:
    - Directly addresses the question
    - Cites relevant sections when applicable
    - Explains any legal terminology used
    - Highlights important conditions or exceptions
    - Maintains accuracy to the source document
    """
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

@st.cache_data
def process_document(uploaded_file, similarity_threshold, max_chunk_size):
    """Process and split the uploaded document using BERT-based semantic splitting."""
    print(f"Processing file: {uploaded_file.name} of type: {uploaded_file.type}")

    # Load BERT model and tokenizer
    tokenizer, model = load_bert_model()

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

            # Split text semantically
            chunks = split_semantically(full_text, tokenizer, model,
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
            search_kwargs={"k": 3}  # Adjust the number of chunks to retrieve
        )
        model_local = get_llm_model(st.session_state.temperature)

        # Create RAG prompt using cached function
        rag_prompt = create_rag_prompt()

        # Create RAG chain
        rag_chain = (
            {"context": retriever, "question": RunnablePassthrough()}
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
    uploaded_file = st.file_uploader("Choose a document", type=['pdf', 'txt', 'doc', 'docx'])

    # Semantic Splitting Settings
    st.markdown("### Semantic Splitting Settings")
    similarity_threshold = st.slider(
        "Semantic Similarity Threshold",
        min_value=0.0,
        max_value=1.0,
        value=0.8,
        step=0.1,
        help="Higher values create more granular chunks"
    )

    max_chunk_size = st.number_input(
        "Maximum Chunk Size (characters)",
        min_value=500,
        max_value=10000,
        value=500,
        step=500,
        help="Maximum number of characters per chunk"
    )

    if uploaded_file is not None:
        if st.button('Create RAG'):
            with st.spinner('Creating RAG system...'):
                try:
                    cleanup_chroma()
                    st.session_state.rag_chain = create_rag(
                        uploaded_file,
                        similarity_threshold,
                        max_chunk_size
                    )
                    st.success('RAG system created successfully!')
                except Exception as e:
                    st.error(f'Error creating RAG system: {str(e)}')

    st.markdown("### Model Settings")
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
                st.session_state.chat_history.append({"role": "assistant", "content": response})
                st.markdown(response)
