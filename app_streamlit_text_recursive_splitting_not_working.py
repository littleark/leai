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
# from tts_utils import initialize_tts, play
from deepgram_tts import TextToSpeech
from micro import listen

from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

import logging
from typing import List, Dict

# Configure logging
# logging.basicConfig(level=logging.DEBUG)
# logger = logging.getLogger(__name__)

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
    st.session_state.temperature = 0.0

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
        model="llama3.1",
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

    CRITICAL INSTRUCTIONS:
    - ONLY use information from the provided CONTEXT
    - If the question cannot be answered using CONTEXT, respond with "I'm not sure about that."
    - Do NOT make up any information not present in the CONTEXT

    Guidelines:
    - Your answer should be maximum two lines long
    - If the question asks to impersonate a character, use first person when answering
    - Keep the discussion flowing naturally without greetings or acknowledgements
    - End your response with a follow-up question to keep the conversation going
    - Keep responses friendly and appropriate for an 8-year-old
    - Make specific references to events and characters from the book

    Context: {context}

    Chat History:
    {chat_history}

    Current Question: {question}

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

# Enhanced prompt template with strict instructions
def create_strict_chat_prompt():
    template = """CRITICAL CONTEXT GUIDELINES:
    - You are talking to an 8-year-old about a specific book
    - ONLY use information DIRECTLY from the provided context
    - If a question CANNOT be answered using the context, respond: "I don't know that from the book."
    - Be concise and child-friendly
    - NEVER invent or guess information

    Context: {context}

    Chat History: {chat_history}

    Current Question: {question}

    RESPONSE RULES:
    - Maximum 2 sentences
    - Use simple language
    - Refer to specific book details
    - If unsure, admit you don't know

    Answer: """
    return ChatPromptTemplate.from_template(template)

def create_rag(uploaded_file, chunk_size, chunk_overlap):
    try:
        # Process documents
        doc_splits = process_document(uploaded_file, chunk_size, chunk_overlap)

        # Get the title from the first line of the document
        book_title = doc_splits[0].page_content.split('\n')[0].strip()
        st.session_state.book_title = book_title

        # Get embedding model
        embedding_function = get_embeddings_model()

        # Create vector store with enhanced retrieval
        st.session_state.vectorstore = Chroma.from_documents(
            documents=doc_splits,
            embedding=embedding_function,
            persist_directory=PERSIST_DIR
        )

        print('vectorstore', st.session_state.vectorstore)
        print('##############################################################')

        retriever = st.session_state.vectorstore.as_retriever(
            search_kwargs={"k": 3}
        )

        # Advanced retriever with multiple strategies
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

            # Retrieve initial set of documents
            print('create retriever')
            print('vectorstore', st.session_state.vectorstore)
            print('------------------------------------------------------------------------------')
            retriever = st.session_state.vectorstore.as_retriever(
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

        # Get LLM model with even more conservative settings
        model_local = ChatOllama(
            model="llama3.2",
            base_url="http://localhost:11434",
            temperature=0.0,  # Lowest temperature
            num_predict=150,  # Limit response length
            top_k=10,  # Limit token selection
            top_p=0.1,  # Very conservative sampling
        )

        # Modify the RAG chain to use custom retriever
        # rag_chain = (
        #     {
        #         "context": lambda x: "\n\n".join([
        #             doc.page_content for doc in custom_retriever(
        #                 x["question"] if isinstance(x, dict) else x,
        #                 k=5  # Retrieve up to 5 most relevant documents
        #             )
        #         ]),
        #         "chat_history": lambda x: format_chat_history(
        #             x.get("chat_history", []) if isinstance(x, dict) else []
        #         ),
        #         "question": lambda x: x["question"] if isinstance(x, dict) else x,
        #         "reader_name": lambda x: st.session_state.reader_name
        #     }
        #     | create_strict_chat_prompt()
        #     | model_local
        #     | StrOutputParser()
        # )

        def format_docs(docs):
                    return "\n\n".join(doc.page_content for doc in docs)

        # rag_chain = (
        #     {"context": lambda x: format_docs(retriever.get_relevant_documents(x)),
        #         "question": RunnablePassthrough()}
        #     | create_strict_chat_prompt()
        #     | model_local
        #     | StrOutputParser()
        # )

        # rag_chain = (
        #     {"context": lambda x: format_docs(retriever.get_relevant_documents(x)),
        #         "question": RunnablePassthrough()}
        #     | create_chat_prompt()
        #     | model_local
        #     | StrOutputParser()
        # )

        rag_chain = (
            {
                "context": lambda x: "\n\n".join([
                    doc.page_content for doc in custom_retriever(
                        x["question"] if isinstance(x, dict) else x,
                        k=5  # Retrieve up to 5 most relevant documents
                    )
                ]),
                "chat_history": lambda x: format_chat_history(
                    x.get("chat_history", []) if isinstance(x, dict) else []
                ),
                "question": lambda x: x["question"] if isinstance(x, dict) else x,
                "reader_name": lambda x: st.session_state.reader_name
            }
            | create_strict_chat_prompt()
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
        tts = TextToSpeech()
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
                    if st.session_state.vectorstore is not None:
                        print('Because vectorstore is not None, cleaning up ChromaDB resources.')
                        cleanup_chroma()

                    print('Creating RAG system with rag_chaing')
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



def crunchPrompt(prompt, debug=True):
    # Add extensive logging
    if debug:
        print("\n===== PROMPT PROCESSING DEBUG =====")
        print(f"Input Prompt: {prompt}")

    # Maintain limited history
    if len(st.session_state.chat_history) >= MAX_HISTORY_LENGTH * 2:
        st.session_state.chat_history = st.session_state.chat_history[-(MAX_HISTORY_LENGTH * 2 - 1):]

    st.session_state.chat_history.append({"role": "user", "content": prompt})

    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner('Thinking...'):
            # response = st.session_state.rag_chain.invoke(
            #     question=prompt,
            #     chat_history=st.session_state.chat_history,
            #     reader_name=st.session_state.reader_name,
            # )
            response = st.session_state.rag_chain.invoke({
                "question": prompt,
                "chat_history": st.session_state.chat_history
            })
            response = response.strip()
            st.session_state.chat_history.append({"role": "assistant", "content": response})
            st.markdown(response)
    # with st.chat_message("assistant"):
    #     with st.spinner('Thinking...'):
    #         try:
    #             # Invoke the RAG chain
    #             print('invoking rag chain --->', prompt)
    #             response = st.session_state.rag_chain.invoke({
    #                 "question": prompt,
    #                 "chat_history": st.session_state.chat_history
    #             })

    #             # Clean and validate response
    #             response = response.strip()

    #             # Optional: Add a post-processing check
    #             if debug:
    #                 print("RAW RESPONSE:", response)

    #             st.session_state.chat_history.append({"role": "assistant", "content": response})
    #             st.markdown(response)

    #             # Play audio for new assistant response
    #             # if st.session_state.tts_enabled:
    #             #     tts.play(response)

    #         except Exception as e:
    #             response = f"I'm sorry, I couldn't process that question. Error: {str(e)}"
    #             st.error(response)
    #             print(f"dRAG Processing Error: {e}")

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
