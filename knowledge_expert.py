import os
import glob
from dotenv import load_dotenv
import gradio as gr
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.schema import Document
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
import pickle
import base64

load_dotenv(override=True)
os.environ['OPENAI_API_KEY'] = os.getenv('OPENAI_API_KEY', 'your-key-if-not-using-env')

MODEL = "gpt-4o-mini"
VECTOR_DB_PATH = "milltech_vector_db"

conversation_history = []

def initialize_system():
    if os.path.exists(f"{VECTOR_DB_PATH}.faiss") and os.path.exists(f"{VECTOR_DB_PATH}.pkl"):
        print("Loading existing vector database...")
        embeddings = OpenAIEmbeddings()
        vectorstore = FAISS.load_local(VECTOR_DB_PATH, embeddings, allow_dangerous_deserialization=True)
        total_vectors = vectorstore.index.ntotal
        dimensions = vectorstore.index.d
        print(f"Loaded {total_vectors} vectors with {dimensions:,} dimensions")
        return vectorstore
    else:
        print("Creating new vector database from MillTech knowledge base...")
        
        text_loader_kwargs = {'encoding': 'utf-8'}
        
        documents = []
        if os.path.exists("MillTech"):
            loader = DirectoryLoader("MillTech", glob="**/*.md", loader_cls=TextLoader, loader_kwargs=text_loader_kwargs)
            folder_docs = loader.load()
            for doc in folder_docs:
                doc.metadata["doc_type"] = "MillTech"
                documents.append(doc)
        
        if not documents:
            print("No documents found! Please ensure MillTech folder exists with .md files")
            return None
        
        text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = text_splitter.split_documents(documents)
        
        embeddings = OpenAIEmbeddings()
        vectorstore = FAISS.from_documents(chunks, embedding=embeddings)
        
        vectorstore.save_local(VECTOR_DB_PATH)
        
        total_vectors = vectorstore.index.ntotal
        dimensions = vectorstore.index.d
        print(f"Created {len(chunks)} chunks with {dimensions:,} dimensions in the vector store")
        
        return vectorstore

def query(message, history, vectorstore, llm):
    global conversation_history
    
    try:
        retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
        relevant_docs = retriever.get_relevant_documents(message)
        
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

        response = llm.invoke(prompt)
        answer = response.content
        
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
    
    llm = ChatOpenAI(temperature=0.7, model_name=MODEL)
    
    def chat_wrapper(message, history):
        return query(message, history, vectorstore, llm)
    
    # MillTech brand colors from logo analysis: #1e3a5f (dark teal), #2c5282 (lighter teal)
    css = """
    .gradio-container {
        background: linear-gradient(135deg, #1e3a5f 0%, #2c5282 100%) !important;
        min-height: 100vh;
    }
    
    .main-header {
        text-align: center;
        padding: 20px;
        margin-bottom: 20px;
    }
    
    .logo-container {
        display: flex;
        justify-content: center;
        align-items: center;
        margin-bottom: 15px;
    }
    
    .logo-container img {
        max-height: 60px;
        width: auto;
    }
    
    .company-title {
        color: white;
        font-size: 28px;
        font-weight: 600;
        margin: 10px 0 5px 0;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
    }
    
    .company-subtitle {
        color: #e2e8f0;
        font-size: 16px;
        font-weight: 400;
        margin: 0;
    }
    
    .chat-container {
        background: rgba(255, 255, 255, 0.1) !important;
        backdrop-filter: blur(10px);
        border-radius: 15px !important;
        border: 1px solid rgba(255, 255, 255, 0.2) !important;
        padding: 20px !important;
        margin: 20px auto !important;
        max-width: 1000px !important;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.2) !important;
    }
    
    .message.user {
        background: rgba(255, 255, 255, 0.95) !important;
        color: #2d3748 !important;
        border-radius: 12px !important;
        padding: 12px 16px !important;
        margin: 8px 0 !important;
        border: none !important;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1) !important;
    }
    
    .message.bot {
        background: rgba(226, 232, 240, 0.95) !important;
        color: #2d3748 !important;
        border-radius: 12px !important;
        padding: 12px 16px !important;
        margin: 8px 0 !important;
        border: none !important;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1) !important;
    }
    
    .input-container {
        background: rgba(255, 255, 255, 0.9) !important;
        border-radius: 25px !important;
        border: 2px solid rgba(255, 255, 255, 0.3) !important;
        padding: 8px 16px !important;
    }
    
    .input-container:focus-within {
        border-color: rgba(255, 255, 255, 0.6) !important;
        box-shadow: 0 0 0 3px rgba(255, 255, 255, 0.1) !important;
    }
    
    button.primary {
        background: linear-gradient(135deg, #2c5282 0%, #1e3a5f 100%) !important;
        border: none !important;
        border-radius: 20px !important;
        color: white !important;
        font-weight: 600 !important;
        padding: 10px 20px !important;
        transition: all 0.3s ease !important;
    }
    
    button.primary:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.2) !important;
    }
    
    .footer-info {
        text-align: center;
        color: #e2e8f0;
        font-size: 14px;
        margin-top: 20px;
        padding: 20px;
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
            chatbot = gr.ChatInterface(
                chat_wrapper,
                type="messages",
                title="",
                description="Ask me anything about MillTech's products, services, and capabilities.",
                examples=[
                    "What does MillTech do?",
                    "Tell me about MillTech's FX management solutions",
                    "How does MillTech help with cash management?",
                    "What is Co-Pilot?"
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