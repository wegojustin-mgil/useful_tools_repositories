"""
Web page summarization module using OpenAI API.

This module provides functionality to fetch web pages and generate
markdown summaries using OpenAI's GPT models.
"""

import os
import requests
from bs4 import BeautifulSoup
from openai import OpenAI
from dotenv import load_dotenv


# Headers for web scraping to avoid blocking
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/117.0.0.0 Safari/537.36"
}


class Website:
    """
    A class to represent a webpage and extract its content.
    """
    
    def __init__(self, url):
        """
        Create this Website object from the given url using the BeautifulSoup library.
        
        Args:
            url (str): The URL of the website to fetch and parse.
        """
        self.url = url
        response = requests.get(url, headers=HEADERS)
        soup = BeautifulSoup(response.content, 'html.parser')
        self.title = soup.title.string if soup.title else "No title found"
        
        # Remove irrelevant elements from the body
        if soup.body:
            for irrelevant in soup.body(["script", "style", "img", "input"]):
                irrelevant.decompose()
            self.text = soup.body.get_text(separator="\n", strip=True)
        else:
            self.text = ""


def get_openai_client():
    """
    Initialize and return an OpenAI client.
    
    Returns:
        OpenAI: An initialized OpenAI client.
    """
    load_dotenv(override=True)
    api_key = os.getenv('OPENAI_API_KEY')
    
    if not api_key:
        raise ValueError("No OpenAI API key found. Please set OPENAI_API_KEY in your .env file.")
    
    return OpenAI(api_key=api_key)


def create_summary_prompt(website):
    """
    Create a user prompt for summarizing website content.
    
    Args:
        website (Website): A Website object containing the page content.
    
    Returns:
        str: A formatted prompt string for the AI model.
    """
    prompt = f"You are looking at a website titled {website.title}"
    prompt += "\nThe contents of this website is as follows; "
    prompt += "please provide a short summary of this website in markdown. "
    prompt += "If it includes news or announcements, then summarize these too.\n\n"
    prompt += website.text
    return prompt


def display_summary(url, model="gpt-4o-mini"):
    """
    Fetch a URL and return a markdown summary of its content.
    
    Args:
        url (str): The URL of the website to summarize.
        model (str, optional): The OpenAI model to use. Defaults to "gpt-4o-mini".
    
    Returns:
        str: A markdown-formatted summary of the website content.
    
    Raises:
        requests.RequestException: If there's an error fetching the URL.
        ValueError: If the OpenAI API key is not configured.
    """
    # System prompt for the AI assistant
    system_prompt = (
        "You are an assistant that analyzes the contents of a website "
        "and provides a short summary, ignoring text that might be navigation related. "
        "Respond in markdown."
    )
    
    # Fetch and parse the website
    website = Website(url)
    
    # Create messages for the API call
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": create_summary_prompt(website)}
    ]
    
    # Get OpenAI client and make the API call
    client = get_openai_client()
    response = client.chat.completions.create(
        model=model,
        messages=messages
    )
    
    return response.choices[0].message.content


# Example usage
if __name__ == "__main__":
    # Example: Summarize a website
    try:
        url = "https://milltech.com"
        summary = display_summary(url)
        print(f"Summary of {url}:\n")
        print(summary)
    except Exception as e:
        print(f"Error: {e}")