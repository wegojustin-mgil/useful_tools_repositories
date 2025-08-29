"""
Multi-Model Gradio Chatbot

A sleek dark-mode Gradio interface that allows users to chat with multiple AI models:
- OpenAI GPT models
- Anthropic Claude models  
- Google Gemini models
- DeepSeek models

Features streaming responses and easy model switching via dropdown menu.
All API keys are loaded from .env file.
"""

import os
import gradio as gr
from dotenv import load_dotenv
from openai import OpenAI
import anthropic
import google.generativeai
from typing import Generator, Optional


class MultiModelChatbot:
    """
    A chatbot class that handles multiple AI model providers.
    """
    
    def __init__(self):
        """Initialize all model clients."""
        # Load environment variables
        load_dotenv(override=True)
        
        # Initialize clients
        self.openai_client = None
        self.claude_client = None
        self.gemini_client = None
        self.deepseek_client = None
        
        # Setup OpenAI
        openai_key = os.getenv('OPENAI_API_KEY')
        if openai_key:
            self.openai_client = OpenAI(api_key=openai_key)
            print("✓ OpenAI client initialized")
        
        # Setup Claude
        anthropic_key = os.getenv('ANTHROPIC_API_KEY')
        if anthropic_key:
            self.claude_client = anthropic.Anthropic(api_key=anthropic_key)
            print("✓ Claude client initialized")
        
        # Setup Gemini
        google_key = os.getenv('GOOGLE_API_KEY')
        if google_key:
            google.generativeai.configure(api_key=google_key)
            self.gemini_client = google.generativeai.GenerativeModel('gemini-2.0-flash')
            print("✓ Gemini client initialized")
        
        # Setup DeepSeek
        deepseek_key = os.getenv('DEEPSEEK_API_KEY')
        if deepseek_key:
            self.deepseek_client = OpenAI(
                api_key=deepseek_key,
                base_url="https://api.deepseek.com"
            )
            print("✓ DeepSeek client initialized")
        
        # System message for all models
        self.system_message = "You are a helpful assistant that responds in markdown format."
    
    def stream_gpt(self, prompt: str, model: str = "gpt-4.1-mini") -> Generator[str, None, None]:
        """
        Stream response from OpenAI GPT model.
        
        Args:
            prompt: User input
            model: GPT model to use
            
        Yields:
            Partial responses as they arrive
        """
        if not self.openai_client:
            yield "❌ OpenAI not configured. Please check your API key."
            return
        
        try:
            messages = [
                {"role": "system", "content": self.system_message},
                {"role": "user", "content": prompt}
            ]
            
            stream = self.openai_client.chat.completions.create(
                model=model,
                messages=messages,
                stream=True,
                temperature=0.7
            )
            
            result = ""
            for chunk in stream:
                content = chunk.choices[0].delta.content or ""
                result += content
                yield result
                
        except Exception as e:
            yield f"❌ Error with OpenAI: {str(e)}"
    
    def stream_claude(self, prompt: str, model: str = "claude-3-5-sonnet-20241022") -> Generator[str, None, None]:
        """
        Stream response from Anthropic Claude model.
        
        Args:
            prompt: User input  
            model: Claude model to use
            
        Yields:
            Partial responses as they arrive
        """
        if not self.claude_client:
            yield "❌ Claude not configured. Please check your API key."
            return
        
        try:
            result = self.claude_client.messages.stream(
                model=model,
                max_tokens=2000,
                temperature=0.7,
                system=self.system_message,
                messages=[{"role": "user", "content": prompt}]
            )
            
            response = ""
            with result as stream:
                for text in stream.text_stream:
                    response += text or ""
                    yield response
                    
        except Exception as e:
            yield f"❌ Error with Claude: {str(e)}"
    
    def stream_gemini(self, prompt: str) -> Generator[str, None, None]:
        """
        Stream response from Google Gemini model.
        
        Args:
            prompt: User input
            
        Yields:
            Partial responses as they arrive
        """
        if not self.gemini_client:
            yield "❌ Gemini not configured. Please check your API key."
            return
        
        try:
            # Create model with system instruction
            model = google.generativeai.GenerativeModel(
                model_name='gemini-2.0-flash',
                system_instruction=self.system_message
            )
            
            response = model.generate_content(prompt, stream=True)
            
            result = ""
            for chunk in response:
                if hasattr(chunk, 'text') and chunk.text:
                    result += chunk.text
                    yield result
                    
        except Exception as e:
            yield f"❌ Error with Gemini: {str(e)}"
    
    def stream_deepseek(self, prompt: str, model: str = "deepseek-chat") -> Generator[str, None, None]:
        """
        Stream response from DeepSeek model.
        
        Args:
            prompt: User input
            model: DeepSeek model to use
            
        Yields:
            Partial responses as they arrive
        """
        if not self.deepseek_client:
            yield "❌ DeepSeek not configured. Please check your API key."
            return
        
        try:
            messages = [
                {"role": "system", "content": self.system_message},
                {"role": "user", "content": prompt}
            ]
            
            stream = self.deepseek_client.chat.completions.create(
                model=model,
                messages=messages,
                stream=True,
                temperature=0.7
            )
            
            result = ""
            for chunk in stream:
                content = chunk.choices[0].delta.content or ""
                result += content
                yield result
                
        except Exception as e:
            yield f"❌ Error with DeepSeek: {str(e)}"
    
    def chat_with_model(self, prompt: str, model_name: str) -> Generator[str, None, None]:
        """
        Route the prompt to the selected model and stream the response.
        
        Args:
            prompt: User input
            model_name: Selected model name
            
        Yields:
            Streaming response from the selected model
        """
        if not prompt or not prompt.strip():
            yield "⚠️ Please enter a message."
            return
        
        # Add model indicator at the beginning
        model_indicator = f"**🤖 {model_name} is thinking...**\n\n"
        yield model_indicator
        
        if model_name == "GPT-4.1 Mini":
            yield from self.stream_gpt(prompt, "gpt-4.1-mini")
        elif model_name == "GPT-4o Mini":
            yield from self.stream_gpt(prompt, "gpt-4o-mini")
        elif model_name == "Claude 3.5 Sonnet":
            yield from self.stream_claude(prompt, "claude-3-5-sonnet-20241022")
        elif model_name == "Claude 3.5 Haiku":
            yield from self.stream_claude(prompt, "claude-3-5-haiku-latest")
        elif model_name == "Gemini 2.0 Flash":
            yield from self.stream_gemini(prompt)
        elif model_name == "DeepSeek Chat":
            yield from self.stream_deepseek(prompt, "deepseek-chat")
        elif model_name == "DeepSeek Reasoner":
            yield from self.stream_deepseek(prompt, "deepseek-reasoner")
        else:
            yield f"❌ Unknown model: {model_name}"


def create_chatbot_interface():
    """
    Create and configure the Gradio interface.
    
    Returns:
        Configured Gradio interface
    """
    # Initialize chatbot
    chatbot = MultiModelChatbot()
    
    # JavaScript for forcing dark mode
    force_dark_mode = """
    function refresh() {
        const url = new URL(window.location);
        if (url.searchParams.get('__theme') !== 'dark') {
            url.searchParams.set('__theme', 'dark');
            window.location.href = url.href;
        }
    }
    """
    
    # Available models
    model_options = [
        "GPT-4.1 Mini",
        "GPT-4o Mini", 
        "Claude 3.5 Sonnet",
        "Claude 3.5 Haiku",
        "Gemini 2.0 Flash",
        "DeepSeek Chat",
        "DeepSeek Reasoner"
    ]
    
    # Create interface
    interface = gr.Interface(
        fn=chatbot.chat_with_model,
        inputs=[
            gr.Textbox(
                label="💬 Your Message",
                placeholder="Ask me anything...",
                lines=4,
                max_lines=10
            ),
            gr.Dropdown(
                choices=model_options,
                label="🤖 Select AI Model",
                value="GPT-4.1 Mini",
                interactive=True
            )
        ],
        outputs=[
            gr.Markdown(
                label="🎯 AI Response",
                show_copy_button=True
            )
        ],
        title="🌟 Multi-Model AI Chatbot",
        description="""
        Chat with different AI models and compare their responses! 
        
        **Available Models:**
        - **GPT-4.1 Mini & GPT-4o Mini**: OpenAI's efficient models
        - **Claude 3.5 Sonnet & Haiku**: Anthropic's reasoning models  
        - **Gemini 2.0 Flash**: Google's latest multimodal model
        - **DeepSeek Chat & Reasoner**: Advanced reasoning capabilities
        
        💡 *Tip: Try asking the same question to different models to see how they compare!*
        """,
        theme=gr.themes.Base(primary_hue="blue", secondary_hue="gray"),
        flagging_mode="never",
        js=force_dark_mode,
        examples=[
            ["Explain quantum computing in simple terms", "GPT-4.1 Mini"],
            ["Write a haiku about artificial intelligence", "Claude 3.5 Sonnet"],
            ["What are the pros and cons of renewable energy?", "Gemini 2.0 Flash"],
            ["How do neural networks learn?", "DeepSeek Chat"]
        ]
    )
    
    return interface


def launch_chatbot(share: bool = False, debug: bool = False, 
                  server_name: str = "127.0.0.1", server_port: int = None):
    """
    Launch the chatbot interface.
    
    Args:
        share: Whether to create a public shareable link
        debug: Enable debug mode
        server_name: Server hostname
        server_port: Server port (None for auto-selection)
    """
    print("🚀 Starting Multi-Model AI Chatbot...")
    print("🌙 Dark mode enabled by default")
    
    interface = create_chatbot_interface()
    
    # Auto-select available port if none specified
    launch_kwargs = {
        'share': share,
        'debug': debug,
        'server_name': server_name,
        'inbrowser': True,
        'show_error': True
    }
    
    # Only specify port if provided
    if server_port is not None:
        launch_kwargs['server_port'] = server_port
    
    interface.launch(**launch_kwargs)


if __name__ == "__main__":
    # Configuration
    SHARE_PUBLIC = False  # Set to True to create public link
    DEBUG_MODE = False    # Set to True for debugging
    
    try:
        print("🎯 All AI models loaded successfully!")
        print("🌐 Starting web interface...")
        
        launch_chatbot(
            share=SHARE_PUBLIC,
            debug=DEBUG_MODE
        )
    except KeyboardInterrupt:
        print("\n👋 Chatbot stopped by user")
    except Exception as e:
        print(f"❌ Error launching chatbot: {e}")
        import traceback
        traceback.print_exc()