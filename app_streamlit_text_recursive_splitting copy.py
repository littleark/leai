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
from tts_utils import initialize_tts, play
from micro import listen

load_dotenv()

# elevenlabs_api_key = os.getenv("ELEVENLABS_API_KEY")



MAX_HISTORY_LENGTH = 4

# Initialize session state variables
if 'rag_chain' not in st.session_state:
    st.session_state.rag_chain = None

if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

if 'vectorstore' not in st.session_state:
    st.session_state.vectorstore = None

if 'temperature' not in st.session_state:
    st.session_state.temperature = 0.1

if 'prompt_type' not in st.session_state:
    st.session_state.prompt_type = "SIMPLE"

if 'quiz_started' not in st.session_state:
    st.session_state.quiz_started = False

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
        model="llama3.2",
        base_url="http://localhost:11434",
        temperature=_temperature,
    )


def create_welcome_chat_prompt():
    """Create and return the RAG prompt template for the initial welcome in chat mode."""
    template = """You are talking to an 8 year old kid named {reader_name} who has just read the book.
    This is your first interaction. Give a warm, friendly welcome mentioning the book's title,
    and ask an engaging opening question about the book (like their favorite part or character).

    Context: {context}

    Keep your response:
    - Friendly and appropriate for an 8-year-old
    - Short and engaging
    - Ending with an open question about the book

    Answer: """
    return ChatPromptTemplate.from_template(template)


def create_chat_prompt():
    """Create and return the RAG prompt template for ongoing chat."""
    template = """You are talking to an 8 year old kid named {reader_name} who has just read the book.
    Based on the following context and chat history, continue the conversation about the book.

    Context: {context}

    Chat History:
    {chat_history}

    Current Question: {question}

    Guidelines:
    - If the question asks to impersonate a character, use first person when answering
    - Keep the discussion flowing naturally without greetings or acknowledgements
    - End your response with a follow-up question to keep the conversation going
    - Keep responses friendly and appropriate for an 8-year-old
    - Make specific references to events and characters from the book

    Answer: """
    return ChatPromptTemplate.from_template(template)


def create_welcome_quiz_prompt():
    """Create and return the RAG prompt template for the initial welcome in quiz mode."""
    template = """You are an enthusiastic teacher starting a fun quiz about the book.
    This is your first interaction. Give a short, exciting welcome and ask the first quiz question.

    Context: {context}

    Keep your response:
    - Short and exciting
    - Ending with a specific question about the book's content
    - Making the question clear and appropriate for an 8-year-old

    Answer: """
    return ChatPromptTemplate.from_template(template)


def create_quiz_prompt():
    """Create and return the RAG prompt template for ongoing quiz."""
    template = """You are an enthusiastic teacher running a fun quiz about the book.
    Continue the quiz based on the following context and chat history.

    Context: {context}

    Chat History:
    {chat_history}

    Current Question: {question}

    Guidelines:
    - Evaluate the user's previous answer to your last question
    - Give specific feedback about whether they were correct or incorrect
    - For correct answers: award a star (⭐) and give a brief explanation why it's correct
    - For incorrect answers: give an apple (🍎), provide the correct answer, and a brief explanation
    - Then ask a new question about a different aspect of the book
    - Keep responses concise and engaging
    - Questions should be specific and based on the book's content
    - If the user asks a question instead of answering:
        * Provide a brief answer
        * Then steer back to the quiz with a new question

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

        # Get the title from the first line of the document
        book_title = doc_splits[0].page_content.split('\n')[0].strip()
        st.session_state.book_title = book_title

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

        # Create RAG prompt based on selection and conversation state
        if not st.session_state.quiz_started:
            # Select welcome prompt based on mode
            if st.session_state.prompt_type == "QUIZ":
                rag_prompt = create_welcome_quiz_prompt()
            else:
                rag_prompt = create_welcome_chat_prompt()
        else:
            # Select ongoing conversation prompt based on mode
            if st.session_state.prompt_type == "QUIZ":
                rag_prompt = create_quiz_prompt()
            else:
                rag_prompt = create_chat_prompt()

        def format_docs(docs):
            return "\n\n".join(doc.page_content for doc in docs)

        def format_chat_history(chat_history):
            formatted_history = ""
            for message in chat_history:
                formatted_history += f"{message['role'].title()}: {message['content']}\n"
            return formatted_history

        reader_name = st.session_state.reader_name

        rag_chain = (
            {
                "context": lambda x: format_docs(retriever.get_relevant_documents(x["question"])),
                "chat_history": lambda x: format_chat_history(x["chat_history"]),
                "question": lambda x: x["question"],
                "reader_name": lambda x: reader_name
            }
            | rag_prompt
            | model_local
            | StrOutputParser()
        )

        # Initiate conversation for both modes
        if not st.session_state.quiz_started:
            start_message = f"start conversation about {book_title}"
            chain_input = {
                "question": start_message,
                "chat_history": [],
                "reader_name": reader_name
            }
            response = rag_chain.invoke(chain_input)
            st.session_state.chat_history = [
                {"role": "assistant", "content": response.strip()}
            ]
            st.session_state.quiz_started = True

            # Create a new chain with the appropriate ongoing prompt
            if st.session_state.prompt_type == "QUIZ":
                rag_prompt = create_quiz_prompt()
            else:
                rag_prompt = create_chat_prompt()

            # Create the final chain with the ongoing prompt
            rag_chain = (
                {
                    "context": lambda x: format_docs(retriever.get_relevant_documents(x["question"])),
                    "chat_history": lambda x: format_chat_history(x["chat_history"]),
                    "question": lambda x: x["question"],
                    "reader_name": lambda x: reader_name  #
                }
                | rag_prompt
                | model_local
                | StrOutputParser()
            )

        return rag_chain

    except Exception as e:
        print(f"Error creating RAG system: {str(e)}")
        raise e

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
        initialize_tts()
    st.session_state.tts_enabled = st.session_state.tts_toggle

    st.header("📖 Story Upload")
    uploaded_file = st.file_uploader("Choose your book", type=['pdf', 'txt'])



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
                    st.rerun()
                except Exception as e:
                    st.error(f'Error creating RAG system: {str(e)}')

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





def crunchPrompt(prompt):
        # Maintain limited history
        if len(st.session_state.chat_history) >= MAX_HISTORY_LENGTH * 2:
            # Remove oldest messages if we exceed the limit
            st.session_state.chat_history = st.session_state.chat_history[-(MAX_HISTORY_LENGTH * 2 - 1):]

        st.session_state.chat_history.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner('Thinking...'):
                # Create input dictionary with question and limited chat history
                chain_input = {
                    "question": prompt,
                    "chat_history": st.session_state.chat_history[-(MAX_HISTORY_LENGTH * 2 - 1):-1],
                    "reader_name": st.session_state.reader_name
                }
                response = st.session_state.rag_chain.invoke(chain_input)
                response = response.strip()
                st.session_state.chat_history.append({"role": "assistant", "content": response})
                st.markdown(response)
                # Play audio for new assistant response
                if st.session_state.tts_enabled:
                    play(response)

# Create columns for the chat input and microphone button
col1, col2 = st.columns([4, 1])

# Chat input and microphone button at the bottom
with col1:
    prompt = st.chat_input("Ask a question about your document")

with col2:
    mic_button = st.button("🎤", key="mic_button")

# Container for the chat history
if st.session_state.chat_history:
    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

# Check if microphone button or chat input was used
if mic_button:
    if st.session_state.rag_chain is None:
        st.error("Please upload a document and create RAG system first!")
    else:
        st.write("Listening... please speak now...")
        prompt = listen_for_long_break()  # Use the microphone function when button is clicked
        if prompt:
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

elif prompt:
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

# Add chat input and microphone button next to each other
# col1, col2 = st.columns([4, 1])  # Create two columns with different widths

# prompt = None
# with col1:
#     prompt = st.chat_input("Ask a question")

# with col2:
#     mic_button = st.button("🎤", key="mic_button")

# # Check if microphone button or chat input was used
# if mic_button:
#     if st.session_state.rag_chain is None:
#         st.error("Please upload a document and create RAG system first!")
#     else:
#         st.write("Listening... please speak now...")
#         prompt = listen_for_long_break()  # Only call listen_for_long_break when mic_button is clicked
#         if prompt:
#             crunchPrompt(prompt)
# elif prompt:
#     if st.session_state.rag_chain is None:
#         st.error("Please upload a document and create RAG system first!")
#     else:
#         crunchPrompt(prompt)
