class RAGState:
    def __init__(self):
        self.vectorstore = None
        self.rag_chain = None
        self.chat_history = []
        self.book_title = None
        self.reader_name = "Lucy"
        self.temperature = 0.0
        self.current_collection = None
