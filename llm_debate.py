"""
Multi-LLM Debate Module

This module orchestrates debates between different Large Language Models:
- OpenAI GPT models
- Anthropic Claude models  
- Google Gemini models
- DeepSeek as the debate moderator and summarizer

Each model presents their perspective on a given topic across multiple rounds,
with DeepSeek providing final analysis and conclusions.
"""

import os
import time
from typing import List, Dict, Optional
from dotenv import load_dotenv
from openai import OpenAI
import anthropic
import google.generativeai


class LLMDebater:
    """
    A class to manage debates between multiple LLM models.
    """
    
    def __init__(self):
        """Initialize all LLM clients and check API keys."""
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
        else:
            print("✗ OpenAI API key not found")
        
        # Setup Claude
        anthropic_key = os.getenv('ANTHROPIC_API_KEY')
        if anthropic_key:
            self.claude_client = anthropic.Anthropic(api_key=anthropic_key)
            print("✓ Claude client initialized")
        else:
            print("✗ Anthropic API key not found")
        
        # Setup Gemini
        google_key = os.getenv('GOOGLE_API_KEY')
        if google_key:
            google.generativeai.configure(api_key=google_key)
            self.gemini_client = google.generativeai.GenerativeModel('gemini-2.0-flash')
            print("✓ Gemini client initialized")
        else:
            print("✗ Google API key not found")
        
        # Setup DeepSeek
        deepseek_key = os.getenv('DEEPSEEK_API_KEY')
        if deepseek_key:
            self.deepseek_client = OpenAI(
                api_key=deepseek_key,
                base_url="https://api.deepseek.com"
            )
            print("✓ DeepSeek client initialized")
        else:
            print("✗ DeepSeek API key not found")
    
    def call_gpt(self, messages: List[Dict], model: str = "gpt-4.1-mini", 
                 temperature: float = 0.7) -> Optional[str]:
        """
        Call OpenAI GPT model.
        
        Args:
            messages: List of message dictionaries
            model: GPT model to use
            temperature: Sampling temperature
            
        Returns:
            Model response or None if error
        """
        if not self.openai_client:
            return None
            
        try:
            response = self.openai_client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"Error calling GPT: {e}")
            return None
    
    def call_claude(self, messages: List[Dict], system_prompt: str = "", 
                   model: str = "claude-3-5-sonnet-20241022", 
                   temperature: float = 0.7) -> Optional[str]:
        """
        Call Anthropic Claude model.
        
        Args:
            messages: List of message dictionaries (user/assistant only)
            system_prompt: System message
            model: Claude model to use
            temperature: Sampling temperature
            
        Returns:
            Model response or None if error
        """
        if not self.claude_client:
            return None
            
        try:
            # Filter out system messages and format for Claude
            claude_messages = []
            for msg in messages:
                if msg['role'] != 'system':
                    claude_messages.append(msg)
            
            response = self.claude_client.messages.create(
                model=model,
                max_tokens=1000,
                temperature=temperature,
                system=system_prompt,
                messages=claude_messages
            )
            return response.content[0].text
        except Exception as e:
            print(f"Error calling Claude: {e}")
            return None
    
    def call_gemini(self, messages: List[Dict], system_prompt: str = "", 
                   temperature: float = 0.7) -> Optional[str]:
        """
        Call Google Gemini model.
        
        Args:
            messages: List of message dictionaries
            system_prompt: System instruction
            temperature: Sampling temperature
            
        Returns:
            Model response or None if error
        """
        if not self.gemini_client:
            return None
            
        try:
            # Create model with system instruction
            model = google.generativeai.GenerativeModel(
                model_name='gemini-2.0-flash',
                system_instruction=system_prompt
            )
            
            # Format conversation for Gemini
            conversation_text = ""
            for msg in messages:
                if msg['role'] == 'user':
                    conversation_text += f"Human: {msg['content']}\n"
                elif msg['role'] == 'assistant':
                    conversation_text += f"Assistant: {msg['content']}\n"
            
            response = model.generate_content(conversation_text)
            return response.text
        except Exception as e:
            print(f"Error calling Gemini: {e}")
            return None
    
    def call_deepseek(self, messages: List[Dict], model: str = "deepseek-chat", 
                     temperature: float = 0.3) -> Optional[str]:
        """
        Call DeepSeek model.
        
        Args:
            messages: List of message dictionaries
            model: DeepSeek model to use
            temperature: Sampling temperature (lower for analysis)
            
        Returns:
            Model response or None if error
        """
        if not self.deepseek_client:
            return None
            
        try:
            response = self.deepseek_client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"Error calling DeepSeek: {e}")
            return None
    
    def conduct_debate(self, topic: str, rounds: int = 3, 
                      participants: Optional[List[str]] = None) -> Dict:
        """
        Conduct a multi-round debate on a given topic.
        
        Args:
            topic: The debate topic
            rounds: Number of debate rounds
            participants: List of models to include ['gpt', 'claude', 'gemini']
            
        Returns:
            Dictionary containing full debate transcript and summary
        """
        if participants is None:
            participants = ['gpt', 'claude', 'gemini']
        
        # Filter available participants
        available_participants = []
        if 'gpt' in participants and self.openai_client:
            available_participants.append('gpt')
        if 'claude' in participants and self.claude_client:
            available_participants.append('claude')
        if 'gemini' in participants and self.gemini_client:
            available_participants.append('gemini')
        
        if len(available_participants) < 2:
            raise ValueError("Need at least 2 available models to conduct debate")
        
        print(f"\n{'='*80}")
        print(f"STARTING DEBATE: {topic}")
        print(f"Participants: {', '.join(available_participants)}")
        print(f"Rounds: {rounds}")
        print(f"{'='*80}\n")
        
        # Define system prompts for each model
        system_prompts = {
            'gpt': f"""You are participating in a structured debate about: "{topic}". 
Present your arguments clearly and logically. Engage thoughtfully with other participants' 
points while maintaining your perspective. Be respectful but persuasive. Keep responses 
concise but substantive (2-3 paragraphs max).""",
            
            'claude': f"""You are participating in a thoughtful debate about: "{topic}". 
Present well-reasoned arguments and engage constructively with other viewpoints. 
Be analytical and consider multiple perspectives while advocating your position. 
Keep responses focused and substantive (2-3 paragraphs max).""",
            
            'gemini': f"""You are in a structured debate about: "{topic}". 
Present clear, evidence-based arguments. Engage with other participants' points 
thoughtfully and offer unique insights. Be persuasive but respectful. 
Keep responses concise and impactful (2-3 paragraphs max)."""
        }
        
        # Initialize debate history
        debate_history = []
        conversation_context = {}
        for participant in available_participants:
            conversation_context[participant] = []
        
        # Conduct debate rounds
        for round_num in range(1, rounds + 1):
            print(f"\n--- ROUND {round_num} ---\n")
            
            round_responses = {}
            
            for participant in available_participants:
                print(f"{participant.upper()} is thinking...")
                
                # Build context for this participant
                context_messages = [
                    {"role": "system", "content": system_prompts[participant]},
                    {"role": "user", "content": f"This is round {round_num} of {rounds}. "}
                ]
                
                # Add conversation history
                if round_num == 1:
                    context_messages[-1]["content"] += f"Present your opening position on: '{topic}'"
                else:
                    context_messages[-1]["content"] += f"Here's what others have said so far:\n\n"
                    for entry in debate_history:
                        if entry['participant'] != participant:
                            context_messages[-1]["content"] += f"{entry['participant'].upper()}: {entry['response']}\n\n"
                    context_messages[-1]["content"] += f"Now respond with your perspective for round {round_num}:"
                
                # Get response from the model
                response = None
                if participant == 'gpt':
                    response = self.call_gpt(context_messages)
                elif participant == 'claude':
                    response = self.call_claude(
                        [{"role": "user", "content": context_messages[-1]["content"]}],
                        system_prompts[participant]
                    )
                elif participant == 'gemini':
                    response = self.call_gemini(
                        [{"role": "user", "content": context_messages[-1]["content"]}],
                        system_prompts[participant]
                    )
                
                if response:
                    round_responses[participant] = response
                    debate_history.append({
                        'round': round_num,
                        'participant': participant,
                        'response': response
                    })
                    
                    print(f"\n{participant.upper()}:")
                    print("-" * 40)
                    print(response)
                    print()
                else:
                    print(f"Error: {participant} could not respond")
                
                # Small delay between calls
                time.sleep(0.5)
        
        # Generate summary with DeepSeek
        print(f"\n{'='*80}")
        print("GENERATING SUMMARY AND CONCLUSION...")
        print(f"{'='*80}\n")
        
        summary = self._generate_summary(topic, debate_history)
        
        return {
            'topic': topic,
            'participants': available_participants,
            'rounds': rounds,
            'debate_history': debate_history,
            'summary': summary
        }
    
    def _generate_summary(self, topic: str, debate_history: List[Dict]) -> str:
        """
        Use DeepSeek to generate a comprehensive summary and conclusion.
        
        Args:
            topic: The debate topic
            debate_history: List of all debate exchanges
            
        Returns:
            Summary and conclusion from DeepSeek
        """
        if not self.deepseek_client:
            return "Summary unavailable - DeepSeek not configured"
        
        # Format the debate for DeepSeek analysis
        debate_transcript = f"DEBATE TOPIC: {topic}\n\n"
        
        current_round = 1
        for entry in debate_history:
            if entry['round'] != current_round:
                current_round = entry['round']
                debate_transcript += f"\n--- ROUND {current_round} ---\n\n"
            
            debate_transcript += f"{entry['participant'].upper()}:\n"
            debate_transcript += f"{entry['response']}\n\n"
        
        summary_prompt = [
            {
                "role": "system", 
                "content": """You are an expert analyst and debate moderator. Your task is to provide 
a comprehensive, unbiased analysis of the debate that just occurred. Provide:

1. A brief summary of each participant's main arguments
2. Key points of agreement and disagreement
3. Strengths and weaknesses of different positions
4. Your objective conclusion about which arguments were most compelling and why
5. Areas where further discussion might be valuable

Be thorough but concise. Present your analysis in clear, structured markdown format."""
            },
            {
                "role": "user",
                "content": f"Please analyze this debate and provide your summary and conclusions:\n\n{debate_transcript}"
            }
        ]
        
        summary = self.call_deepseek(summary_prompt)
        
        if summary:
            print("DEEPSEEK ANALYSIS:")
            print("-" * 60)
            print(summary)
        else:
            summary = "Error generating summary"
        
        return summary


def llm_debate(topic: str, rounds: int = 3, participants: Optional[List[str]] = None) -> Dict:
    """
    Convenience function to conduct an LLM debate.
    
    Args:
        topic: The debate topic
        rounds: Number of rounds (default: 3)
        participants: List of models ['gpt', 'claude', 'gemini'] (default: all available)
        
    Returns:
        Dictionary with complete debate results
        
    Example:
        result = llm_debate("Is artificial intelligence more beneficial or harmful to society?")
        print(result['summary'])
    """
    debater = LLMDebater()
    return debater.conduct_debate(topic, rounds, participants)


# Example usage
if __name__ == "__main__":
    # Example debate topics
    topics = [
        "Is artificial intelligence more beneficial or harmful to society?",
        "Should companies prioritize profit or social responsibility?",
        "Is remote work better than in-office work for productivity?",
        "Should governments regulate social media platforms?",
        "Is renewable energy feasible as a complete replacement for fossil fuels?"
    ]
    
    # Run a sample debate
    try:
        topic = "Is artificial intelligence more beneficial or harmful to society?"
        
        print("Starting LLM Debate System...")
        print(f"Topic: {topic}")
        
        result = llm_debate(
            topic=topic,
            rounds=2,  # Shorter for testing
            participants=['gpt', 'claude', 'gemini']
        )
        
        print(f"\n{'='*80}")
        print("DEBATE COMPLETE!")
        print(f"Topic: {result['topic']}")
        print(f"Participants: {result['participants']}")
        print(f"Total exchanges: {len(result['debate_history'])}")
        print(f"{'='*80}")
        
    except Exception as e:
        print(f"Error running debate: {e}")
        import traceback
        traceback.print_exc()