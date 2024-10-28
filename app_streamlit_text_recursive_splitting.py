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

# Initialize session state variables
if 'rag_chain' not in st.session_state:
    st.session_state.rag_chain = None

if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

if 'vectorstore' not in st.session_state:
    st.session_state.vectorstore = None

if 'temperature' not in st.session_state:
    st.session_state.temperature = 0.25

if 'prompt_type' not in st.session_state:
    st.session_state.prompt_type = "SIMPLE"

# Create a persistent directory for the database
PERSIST_DIR = os.path.join(os.getcwd(), 'db')
if not os.path.exists(PERSIST_DIR):
    os.makedirs(PERSIST_DIR)

@st.cache_data
def process_document(uploaded_file, chunk_size, chunk_overlap):
    """Process and split the uploaded document using RecursiveCharacterTextSplitter."""
    print(f"Processing file: {uploaded_file.name} of type: {uploaded_file.type}")

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_file_path = os.path.join(temp_dir, uploaded_file.name)

        with open(temp_file_path, 'wb') as f:
            f.write(uploaded_file.getvalue())

        try:
            # Load document based on file type
            if uploaded_file.type == "application/pdf":
                loader = PyPDFLoader(temp_file_path)
                docs = loader.load()
                text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=chunk_size,
                    chunk_overlap=chunk_overlap
                )
                doc_splits = text_splitter.split_documents(docs)

            elif uploaded_file.type in ["text/plain"]:
                with open(temp_file_path, 'r') as f:
                    text = f.read()
                text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=chunk_size,
                    chunk_overlap=chunk_overlap
                )
                doc_splits = text_splitter.create_documents([text])

            else:
                raise ValueError(f"Unsupported file type: {uploaded_file.type}")

            print(f"Document split into {len(doc_splits)} chunks")
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
        temperature=_temperature
    )

def create_rag_prompt():
    """Create and return the RAG prompt template."""
    template = """Based on the following context, please answer the question.

    Context: {context}

    Question: {question}

    Answer: """
    return ChatPromptTemplate.from_template(template)

@st.cache_data
def create_legal_rag_prompt():
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

def create_rag(uploaded_file, chunk_size, chunk_overlap):
    """Create the RAG system."""
    try:
        # Process documents
        doc_splits = process_document(uploaded_file, chunk_size, chunk_overlap)

        # Get embedding model
        embedding_function = get_embeddings_model()

        # Create vector store
        st.session_state.vectorstore = Chroma.from_documents(
            documents=doc_splits,
            embedding=embedding_function,
            persist_directory=PERSIST_DIR
        )

        # Setup retriever
        retriever = st.session_state.vectorstore.as_retriever(
            search_kwargs={"k": 3}
        )

        # Get LLM model
        model_local = get_llm_model(st.session_state.temperature)

        # Create RAG prompt based on selection
        if st.session_state.prompt_type == "SIMPLE":
            rag_prompt = create_rag_prompt()
        else:
            rag_prompt = create_legal_rag_prompt()

        # Create RAG chain
        def format_docs(docs):
            return "\n\n".join(doc.page_content for doc in docs)

        rag_chain = (
            {"context": lambda x: format_docs(retriever.get_relevant_documents(x)),
             "question": RunnablePassthrough()}
            | rag_prompt
            | model_local
            | StrOutputParser()
        )

        return rag_chain

    except Exception as e:
        print(f"Error creating RAG system: {str(e)}")
        raise e

# Streamlit UI
st.title("Legal Document Assistant")

# Sidebar
with st.sidebar:
    st.header("Document Upload")
    uploaded_file = st.file_uploader("Choose a document", type=['pdf', 'txt'])

    # Text splitting parameters
    st.markdown("### Chunk Settings")
    chunk_size = st.slider("Chunk Size", 100, 2000, 1000)
    chunk_overlap = st.slider("Chunk Overlap", 0, 500, 100)

    if uploaded_file is not None:
        st.info("After changing the prompt type, click 'Create RAG' to rebuild the system with the new prompt.")
        if st.button('Create RAG'):
            with st.spinner('Creating RAG system...'):
                try:
                    cleanup_chroma()
                    st.session_state.rag_chain = create_rag(
                        uploaded_file,
                        chunk_size,
                        chunk_overlap
                    )
                    st.success('RAG system created successfully!')
                except Exception as e:
                    st.error(f'Error creating RAG system: {str(e)}')

    # Temperature setting
    st.markdown("### Model Settings")
    prompt_type = st.radio(
        "Select Prompt Type",
        options=["SIMPLE", "LEGAL"],
        index=0 if st.session_state.prompt_type == "SIMPLE" else 1,
        help="Choose between a simple or detailed legal analysis prompt"
    )
    st.session_state.prompt_type = prompt_type

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
        st.cache_data.clear()
        st.cache_resource.clear()
        st.success('Database and caches cleared!')

# Chat interface
for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Ask a question about your document"):
    if st.session_state.rag_chain is None:
        st.error("Please upload a document and create RAG system first!")
    else:
        st.session_state.chat_history.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner('Thinking...'):
                response = st.session_state.rag_chain.invoke(prompt)
                response = response.strip()
                st.session_state.chat_history.append({"role": "assistant", "content": response})
                st.markdown(response)
