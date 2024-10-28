import chromadb
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

def cleanup_chroma():
    if st.session_state.vectorstore is not None:
        try:
            st.session_state.vectorstore._client.reset()
            st.session_state.vectorstore._client = None
        except:
            pass

def create_rag(uploaded_file):
    print(f"Processing file: {uploaded_file.name} of type: {uploaded_file.type}")

    # Create a temporary directory to store the file
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_file_path = os.path.join(temp_dir, uploaded_file.name)

        # Save the uploaded file
        with open(temp_file_path, 'wb') as f:
            f.write(uploaded_file.getvalue())

        print(f"Temporary file created at: {temp_file_path}")

        try:
            # Load document based on file type
            if uploaded_file.type == "application/pdf":
                print("Loading PDF document...")
                loader = PyPDFLoader(temp_file_path)
                docs = loader.load()
            elif uploaded_file.type in ["application/vnd.openxmlformats-officedocument.wordprocessingml.document"]:
                print("Loading DOCX document...")
                text = docx2txt.process(temp_file_path)
                docs = [Document(page_content=text, metadata={"source": uploaded_file.name})]
            elif uploaded_file.type == "application/msword":
                print("Loading DOC document...")
                st.warning("Warning: .doc files might not be fully supported. Consider converting to .docx")
                text = docx2txt.process(temp_file_path)
                docs = [Document(page_content=text, metadata={"source": uploaded_file.name})]
            elif uploaded_file.type == "text/plain":
                print("Loading text document...")
                loader = TextLoader(temp_file_path)
                docs = loader.load()
            else:
                raise ValueError(f"Unsupported file type: {uploaded_file.type}")

            print(f"Document loaded successfully with {len(docs)} pages/sections")

            # Split documents
            text_splitter = CharacterTextSplitter.from_tiktoken_encoder(
                chunk_size=7500,
                chunk_overlap=100
            )
            doc_splits = text_splitter.split_documents(docs)
            print(f"Document split into {len(doc_splits)} chunks")

            # Initialize embedding function
            embedding_function = embeddings.ollama.OllamaEmbeddings(model='nomic-embed-text')

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
            # Persist the database
            st.session_state.vectorstore.persist()

            # Setup retriever and model
            retriever = st.session_state.vectorstore.as_retriever()
            model_local = ChatOllama(
                model="llama3.1",
                base_url="http://localhost:11434",
                temperature=st.session_state.temperature
            )

            context = (
                    f"You are a legal expert assisting with the analysis of a legal document. "
                    f"The document is a legal dispute from India "
                    f". Please provide clear and concise answers based on the content of the document. Your responses should strictly adhere to the content of this document."
                )

            # Create RAG prompt
            # final_prompt = f"{context}\n\n{user_intent}"
            rag_template = """{context}
            The user has asked: '{question}'.

            Please clarify the relevant sections or implications of the document related to their question.
            If the answer to their question is not found in the document, respond with 'I don't know'
            instead of making assumptions or fabricating information.
            """
            rag_prompt = ChatPromptTemplate.from_template(rag_template)

            # Create RAG chain
            rag_chain = (
                {"context": retriever, "question": RunnablePassthrough()}
                | rag_prompt
                | model_local
                | StrOutputParser()
            )

            return rag_chain

        except Exception as e:
            print(f"Error processing document: {str(e)}")
            raise e

# Add title at the top of the app
st.title("Document Chat Assistant")

# Sidebar
with st.sidebar:
    st.header("Document Upload")
    uploaded_file = st.file_uploader("Choose a document", type=['pdf', 'txt', 'doc', 'docx'])

    if uploaded_file is not None:
        if st.button('Create RAG'):
            with st.spinner('Creating RAG system...'):
                try:
                    cleanup_chroma()
                    st.session_state.rag_chain = create_rag(uploaded_file)
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
            # Clear the Chroma client
            if st.session_state.vectorstore is not None:
                st.session_state.vectorstore._client = None
                st.session_state.vectorstore = None
        st.session_state.rag_chain = None
        st.session_state.chat_history = []
        st.success('Database cleared!')

    # Download chat history button
    if st.session_state.chat_history:
        # Convert chat history to downloadable format
        chat_text = "\n\n".join([
            f"{msg['role'].upper()}: {msg['content']}"
            for msg in st.session_state.chat_history
        ])
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
