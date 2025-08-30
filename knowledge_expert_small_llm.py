import os
from dotenv import load_dotenv
import gradio as gr
from langchain_community.document_loaders import DirectoryLoader, TextLoader, PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.llms import Ollama
from langchain_community.vectorstores import FAISS
import pickle

load_dotenv(override=True)

# Ollama configuration for local models
EMBEDDING_MODEL = "nomic-embed-text"
LLM_MODEL = "llama3.2:3b"
VECTOR_DB_PATH = "vector_db_small_llm"  # New database name for local models

conversation_history = []

def initialize_system():
    # Import required modules at the top
    import faiss
    import pickle
    from langchain_community.vectorstores.faiss import FAISS
    from langchain_community.docstore.in_memory import InMemoryDocstore
    
    # Check for both possible file structures
    faiss_file_path = f"{VECTOR_DB_PATH}.faiss"
    pkl_file_path = f"{VECTOR_DB_PATH}.pkl"
    
    if os.path.exists(faiss_file_path) and os.path.exists(pkl_file_path):
        print("Loading existing vector database...")
        try:
            embeddings = OllamaEmbeddings(
                model=EMBEDDING_MODEL,
                base_url="http://localhost:11434"
            )
            
            # Load the FAISS index directly
            index = faiss.read_index(faiss_file_path)
            
            # Load the docstore from pickle
            with open(pkl_file_path, 'rb') as f:
                docstore_data = pickle.load(f)
            
            # Handle different pickle formats
            if isinstance(docstore_data, dict):
                if 'docstore' in docstore_data and 'index_to_docstore_id' in docstore_data:
                    docstore = docstore_data['docstore']
                    index_to_docstore_id = docstore_data['index_to_docstore_id']
                elif '_dict' in docstore_data:
                    # Handle InMemoryDocstore format
                    docstore = InMemoryDocstore(docstore_data['_dict'])
                    index_to_docstore_id = {i: str(i) for i in range(index.ntotal)}
                else:
                    # Direct dictionary format
                    docstore = InMemoryDocstore(docstore_data)
                    index_to_docstore_id = {i: str(i) for i in range(index.ntotal)}
            else:
                # Fallback for other formats
                docstore = InMemoryDocstore({})
                index_to_docstore_id = {}
            
            vectorstore = FAISS(embeddings, index, docstore, index_to_docstore_id)
            
            total_vectors = vectorstore.index.ntotal
            dimensions = vectorstore.index.d
            print(f"Loaded {total_vectors} vectors with {dimensions:,} dimensions")
            return vectorstore
            
        except Exception as e:
            print(f"Error loading existing database: {e}")
            print("Creating new vector database...")
            # Fall through to create new database
    
    # If no existing database or loading failed, create new one
    else:
        print("Creating new vector database from MillTech knowledge base...")
        
        text_loader_kwargs = {'encoding': 'utf-8'}
        
        documents = []
        milltech_path = "knowledge-base/MillTech"
        
        # Determine which path exists
        active_path = None
        if os.path.exists(milltech_path):
            active_path = milltech_path
        elif os.path.exists("MillTech"):
            active_path = "MillTech"
        
        if active_path:
            # Load markdown files
            md_loader = DirectoryLoader(active_path, glob="**/*.md", loader_cls=TextLoader, loader_kwargs=text_loader_kwargs)
            try:
                md_docs = md_loader.load()
                for doc in md_docs:
                    doc.metadata["doc_type"] = "MillTech"
                    doc.metadata["file_type"] = "markdown"
                    documents.append(doc)
                print(f"Loaded {len(md_docs)} markdown files")
            except Exception as e:
                print(f"No markdown files found or error loading: {e}")
            
            # Load PDF files
            import glob as glob_module
            pdf_files = glob_module.glob(os.path.join(active_path, "**/*.pdf"), recursive=True)
            for pdf_file in pdf_files:
                try:
                    pdf_loader = PyPDFLoader(pdf_file)
                    pdf_docs = pdf_loader.load()
                    for doc in pdf_docs:
                        doc.metadata["doc_type"] = "MillTech"
                        doc.metadata["file_type"] = "pdf"
                        doc.metadata["source"] = pdf_file
                        documents.append(doc)
                    print(f"Loaded PDF: {os.path.basename(pdf_file)}")
                except Exception as e:
                    print(f"Error loading PDF {pdf_file}: {e}")
        
        if not documents:
            print("No documents found! Please ensure MillTech folder exists with .md or .pdf files")
            return None
        
        print(f"Total documents loaded: {len(documents)}")
        
        text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = text_splitter.split_documents(documents)
        
        embeddings = OllamaEmbeddings(
            model=EMBEDDING_MODEL,
            base_url="http://localhost:11434"
        )
        vectorstore = FAISS.from_documents(chunks, embedding=embeddings)
        
        # Save using the same format as our loading logic expects
        # Save FAISS index
        faiss.write_index(vectorstore.index, f"{VECTOR_DB_PATH}.faiss")
        
        # Save docstore and mapping
        docstore_data = {
            'docstore': vectorstore.docstore,
            'index_to_docstore_id': vectorstore.index_to_docstore_id
        }
        with open(f"{VECTOR_DB_PATH}.pkl", 'wb') as f:
            pickle.dump(docstore_data, f)
        
        total_vectors = vectorstore.index.ntotal
        dimensions = vectorstore.index.d
        print(f"Created {len(chunks)} chunks with {dimensions:,} dimensions in the vector store")
        
        return vectorstore

def query(message, vectorstore, llm):
    global conversation_history
    
    try:
        retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
        relevant_docs = retriever.invoke(message)
        
        context = "\n\n".join([doc.page_content for doc in relevant_docs])
        
        conversation_context = ""
        if conversation_history:
            recent_history = conversation_history[-6:]
            conversation_context = "\n".join([
                f"Human: {h['human']}\nAssistant: {h['assistant']}" 
                for h in recent_history
            ])
        
        prompt = f"""You are a knowledgeable assistant for MillTech, a company specializing in FX & Cash Management automation. Use the following context to answer questions accurately and helpfully.

Context from MillTech knowledge base:
{context}

Previous conversation:
{conversation_context}

Current question: {message}

Please provide a helpful response based on the MillTech information provided. If the information isn't in the context, say so politely and offer to help with what you do know about MillTech."""

        # Ollama returns string directly, not a response object
        answer = llm.invoke(prompt)
        
        conversation_history.append({
            "human": message,
            "assistant": answer
        })
        
        if len(conversation_history) > 10:
            conversation_history = conversation_history[-10:]
        
        return answer
        
    except Exception as e:
        print(f"Error processing query: {e}")
        return f"I apologize, but I encountered an error processing your question: {str(e)}"

def create_knowledge_expert_interface():
    vectorstore = initialize_system()
    if not vectorstore:
        print("Failed to initialize vector database!")
        return None
    
    llm = Ollama(
        model=LLM_MODEL,
        temperature=0.7,
        base_url="http://localhost:11434"
    )
    
    def chat_wrapper(message, _history):
        return query(message, vectorstore, llm)
    
    # Professional MillTech UI with improved contrast and readability
    css = """
    * {
        font-family: 'Segoe UI', 'Roboto', 'Helvetica Neue', Arial, sans-serif;
    }
    
    .gradio-container {
        background: linear-gradient(135deg, #f8fafc 0%, #e2e8f0 100%) !important;
        min-height: 100vh;
    }
    
    .main-header {
        text-align: center;
        padding: 30px 20px;
        margin-bottom: 20px;
        background: linear-gradient(135deg, #1e3a5f 0%, #2c5282 100%);
        border-radius: 0 0 20px 20px;
        box-shadow: 0 4px 20px rgba(30, 58, 95, 0.3);
    }
    
    .logo-container {
        display: flex;
        justify-content: center;
        align-items: center;
        margin-bottom: 20px;
    }
    
    .logo-container img {
        max-height: 70px;
        width: auto;
        filter: drop-shadow(0 2px 10px rgba(0,0,0,0.2));
    }
    
    .company-title {
        color: white;
        font-size: 32px;
        font-weight: 700;
        margin: 15px 0 8px 0;
        text-shadow: 0 2px 10px rgba(0,0,0,0.3);
        letter-spacing: -0.5px;
    }
    
    .company-subtitle {
        color: #cbd5e0;
        font-size: 18px;
        font-weight: 400;
        margin: 0;
        opacity: 0.9;
    }
    
    .chat-container {
        background: white !important;
        border-radius: 20px !important;
        border: 1px solid #e2e8f0 !important;
        padding: 30px !important;
        margin: 20px auto !important;
        max-width: 1200px !important;
        min-height: 600px !important;
        box-shadow: 0 10px 40px rgba(30, 58, 95, 0.1) !important;
    }
    
    /* Make the chatbot interface larger */
    .gr-chatinterface {
        height: 700px !important;
        min-height: 700px !important;
    }
    
    .gr-chatbot {
        height: 500px !important;
        min-height: 500px !important;
        max-height: 500px !important;
    }
    
    /* Chat Messages Styling */
    .message.user, .message.user p, .message.user div {
        background: linear-gradient(135deg, #1e3a5f 0%, #2c5282 100%) !important;
        color: white !important;
        border-radius: 18px 18px 4px 18px !important;
        padding: 12px 16px !important;
        margin: 8px 0 !important;
        border: none !important;
        box-shadow: 0 3px 12px rgba(30, 58, 95, 0.3) !important;
        font-weight: 500 !important;
        line-height: 1.5 !important;
        max-width: none !important;
        width: auto !important;
        height: auto !important;
        min-width: 150px !important;
        font-size: 15px !important;
        display: inline-block !important;
        word-break: keep-all !important;
        white-space: nowrap !important;
        overflow: visible !important;
        text-align: left !important;
    }
    
    .message.user * {
        color: white !important;
        background: transparent !important;
    }
    
    /* Force white text for all user message content */
    .message.user p, .message.user span, .message.user div, .message.user code, .message.user pre {
        color: white !important;
    }
    
    /* Gradio specific user message styling */
    .gr-chatbot .message.user, 
    .gr-chatbot .user,
    .gr-chatbot .message[data-testid*="user"],
    .chat-message.user,
    div[data-testid*="user"],
    .gr-chatbot .message:first-child,
    .gradio-chatbot .message:first-child {
        background: linear-gradient(135deg, #1e3a5f 0%, #2c5282 100%) !important;
        color: white !important;
        max-width: none !important;
        width: auto !important;
        height: auto !important;
        min-width: 150px !important;
        padding: 12px 16px !important;
        margin: 8px 0 !important;
        font-size: 15px !important;
        line-height: 1.5 !important;
        border-radius: 18px 18px 4px 18px !important;
        display: inline-block !important;
        word-break: keep-all !important;
        white-space: nowrap !important;
        overflow: visible !important;
        text-align: left !important;
        box-sizing: border-box !important;
    }
    
    .gr-chatbot .message.user *,
    .gr-chatbot .user *,
    .gr-chatbot .message[data-testid*="user"] *,
    .chat-message.user *,
    div[data-testid*="user"] *,
    .gr-chatbot .message:first-child *,
    .gradio-chatbot .message:first-child * {
        color: white !important;
        max-width: none !important;
        width: auto !important;
        height: auto !important;
        padding: 0 !important;
        margin: 0 !important;
    }
    
    .message.bot {
        background: #f7fafc !important;
        color: #2d3748 !important;
        border-radius: 18px 18px 18px 4px !important;
        padding: 14px 18px !important;
        margin: 12px 0 !important;
        border: 1px solid #e2e8f0 !important;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.05) !important;
        line-height: 1.6 !important;
    }
    
    /* Input Styling */
    .input-container, textarea, input[type="text"] {
        background: #f7fafc !important;
        border-radius: 12px !important;
        border: 2px solid #e2e8f0 !important;
        padding: 12px 16px !important;
        color: #2d3748 !important;
        font-size: 16px !important;
        transition: all 0.3s ease !important;
        min-height: 50px !important;
        max-height: 120px !important;
        resize: vertical !important;
    }
    
    /* Specific textarea sizing - More aggressive targeting */
    .gr-chatbot textarea, 
    .gr-textbox textarea,
    .gr-chatinterface textarea,
    .gradio-container textarea,
    textarea[data-testid*="textbox"],
    .chat-input textarea,
    .message-input textarea {
        min-height: 50px !important;
        max-height: 100px !important;
        height: 50px !important;
        resize: vertical !important;
        font-size: 16px !important;
        line-height: 1.4 !important;
    }
    
    /* Force smaller input containers */
    .gr-textbox,
    .gr-chatinterface .gr-textbox,
    .gradio-container .gr-textbox,
    div[data-testid*="textbox"] {
        min-height: 50px !important;
        max-height: 100px !important;
        height: auto !important;
    }
    
    /* Override any large default heights */
    .gr-form .gr-textbox,
    .gr-interface .gr-textbox {
        height: 50px !important;
        min-height: 50px !important;
    }
    
    /* Nuclear option - force all textareas to be smaller */
    * textarea {
        min-height: 50px !important;
        height: 50px !important;
        max-height: 120px !important;
    }
    
    /* ChatInterface specific fixes */
    .gradio-chatinterface textarea,
    .gradio-chatinterface .gr-textbox textarea {
        height: 50px !important;
        min-height: 50px !important;
        rows: 2 !important;
    }
    
    .input-container:focus-within, textarea:focus, input[type="text"]:focus {
        border-color: #2c5282 !important;
        box-shadow: 0 0 0 3px rgba(44, 82, 130, 0.1) !important;
        outline: none !important;
    }
    
    /* Button Styling */
    button.primary, .gr-button-primary {
        background: linear-gradient(135deg, #2c5282 0%, #1e3a5f 100%) !important;
        border: none !important;
        border-radius: 12px !important;
        color: white !important;
        font-weight: 600 !important;
        padding: 12px 24px !important;
        transition: all 0.3s ease !important;
        font-size: 15px !important;
        cursor: pointer !important;
    }
    
    button.primary:hover, .gr-button-primary:hover {
        transform: translateY(-1px) !important;
        box-shadow: 0 6px 20px rgba(30, 58, 95, 0.3) !important;
        background: linear-gradient(135deg, #1e3a5f 0%, #2c5282 100%) !important;
    }
    
    /* Examples Styling - Improved sample questions */
    .gr-examples {
        margin-top: 25px !important;
        margin-bottom: 20px !important;
    }
    
    .gr-examples .gr-button, 
    .gradio-button,
    .gr-chatinterface .gr-examples button,
    button[data-testid*="example"] {
        background: linear-gradient(135deg, #f8fafc 0%, #e2e8f0 100%) !important;
        border: 2px solid #cbd5e0 !important;
        border-radius: 25px !important;
        color: #2d3748 !important;
        padding: 10px 16px !important;
        margin: 6px !important;
        transition: all 0.3s ease !important;
        font-size: 14px !important;
        font-weight: 500 !important;
        height: auto !important;
        min-height: 40px !important;
        max-height: 65px !important;
        width: auto !important;
        max-width: 280px !important;
        display: inline-block !important;
        text-align: center !important;
        line-height: 1.3 !important;
        overflow: hidden !important;
        white-space: normal !important;
        word-wrap: break-word !important;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05) !important;
    }
    
    .gr-examples .gr-button:hover,
    .gradio-button:hover,
    .gr-chatinterface .gr-examples button:hover,
    button[data-testid*="example"]:hover {
        border-color: #2c5282 !important;
        background: linear-gradient(135deg, #2c5282 0%, #1e3a5f 100%) !important;
        color: white !important;
        transform: translateY(-2px) !important;
        box-shadow: 0 4px 12px rgba(30, 58, 95, 0.2) !important;
    }
    
    /* Force example buttons to be compact */
    .gr-chatinterface .gr-examples,
    .gradio-container .gr-examples {
        display: flex !important;
        flex-wrap: wrap !important;
        gap: 8px !important;
        justify-content: center !important;
    }
    
    /* Override any large button dimensions */
    .gr-chatinterface button {
        height: auto !important;
        min-height: 36px !important;
        max-height: 60px !important;
        padding: 8px 12px !important;
        font-size: 14px !important;
    }
    
    /* Scrollbar Styling */
    ::-webkit-scrollbar {
        width: 8px;
    }
    
    ::-webkit-scrollbar-track {
        background: #f1f1f1;
        border-radius: 4px;
    }
    
    ::-webkit-scrollbar-thumb {
        background: #2c5282;
        border-radius: 4px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: #1e3a5f;
    }
    
    .footer-info {
        text-align: center;
        color: #718096;
        font-size: 14px;
        margin-top: 30px;
        padding: 20px;
        font-weight: 500;
    }
    
    /* Description Text */
    .gr-markdown {
        color: #4a5568 !important;
        font-size: 16px !important;
        margin-bottom: 20px !important;
        text-align: center !important;
    }
    
    /* Loading Animation */
    .loading {
        display: inline-block;
        width: 20px;
        height: 20px;
        border: 3px solid #f3f3f3;
        border-radius: 50%;
        border-top-color: #2c5282;
        animation: spin 1s ease-in-out infinite;
    }
    
    @keyframes spin {
        to { transform: rotate(360deg); }
    }
    """
    
    # Get the logo path
    logo_path = "milltech_brand.png" if os.path.exists("milltech_brand.png") else None
    
    with gr.Blocks(css=css, title="MillTech Knowledge Expert") as interface:
        with gr.Column(elem_classes=["main-header"]):
            if logo_path:
                gr.Image(logo_path, height=60, width=200, elem_classes=["logo-container"], show_label=False, show_download_button=False, container=False)
            
            gr.HTML("""
                <h1 class="company-title">MillTech Knowledge Expert</h1>
                <p class="company-subtitle">FX & Cash Management Automation Assistant</p>
            """)
        
        with gr.Column(elem_classes=["chat-container"]):
            gr.HTML("""
                <div style="text-align: center; padding: 0 0 20px 0; border-bottom: 1px solid #e2e8f0; margin-bottom: 20px;">
                    <h3 style="color: #2d3748; font-size: 20px; margin: 0 0 10px 0; font-weight: 600;">
                        🚀 Welcome to MillTech Knowledge Expert
                    </h3>
                    <p style="color: #718096; font-size: 16px; margin: 0; line-height: 1.5;">
                        Your AI-powered assistant for FX & Cash Management solutions. Get instant answers about our products, services, and how we can help optimize your financial operations.
                    </p>
                </div>
            """)
            
            gr.ChatInterface(
                chat_wrapper,
                type="messages",
                title="",
                description="💡 **Get started with these sample questions or ask anything about MillTech:**",
                examples=[
                    "🏢 What does MillTech do?",
                    "💱 Tell me about FX Management solutions",
                    "💰 How does Cash Management work?",
                    "🤖 What is Co-Pilot and how does it help?",
                    "📊 What are the benefits of using MillTech?",
                    "🔄 How does Trade Workflow Automation work?",
                    "📈 Tell me about Transaction Cost Analysis",
                    "🏦 Who are MillTech's typical clients?"
                ],
                cache_examples=False
            )
        
        gr.HTML("""
            <div class="footer-info">
                <p>Powered by advanced AI and MillTech's comprehensive knowledge base</p>
            </div>
        """)
    
    return interface

if __name__ == "__main__":
    interface = create_knowledge_expert_interface()
    if interface:
        # Auto-select available port
        import socket
        def find_free_port(start_port=7860, max_attempts=100):
            for i in range(max_attempts):
                port = start_port + i
                try:
                    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                        s.bind(('127.0.0.1', port))
                        return port
                except OSError:
                    continue
            return None

        port = find_free_port()
        if port:
            print(f"Starting MillTech Knowledge Expert on port {port}")
            interface.launch(server_port=port, inbrowser=True)
        else:
            print("Could not find an available port")
            interface.launch(inbrowser=True)