"""
Company brochure generation module using OpenAI API.

This module provides functionality to automatically create marketing brochures
for companies by analyzing their website content and generating comprehensive
markdown summaries.
"""

import os
import json
import requests
from typing import List, Dict
from bs4 import BeautifulSoup
from openai import OpenAI
from dotenv import load_dotenv


# Default model to use
DEFAULT_MODEL = 'gpt-4o-mini'

# Headers for web scraping to avoid blocking
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/117.0.0.0 Safari/537.36"
}


class Website:
    """
    A utility class to represent a Website that we have scraped, now with links.
    """

    def __init__(self, url: str):
        """
        Initialize a Website object by fetching and parsing the given URL.
        
        Args:
            url (str): The URL of the website to fetch and parse.
        """
        self.url = url
        response = requests.get(url, headers=HEADERS)
        self.body = response.content
        soup = BeautifulSoup(self.body, 'html.parser')
        self.title = soup.title.string if soup.title else "No title found"
        
        if soup.body:
            for irrelevant in soup.body(["script", "style", "img", "input"]):
                irrelevant.decompose()
            self.text = soup.body.get_text(separator="\n", strip=True)
        else:
            self.text = ""
            
        links = [link.get('href') for link in soup.find_all('a')]
        self.links = [link for link in links if link]

    def get_contents(self) -> str:
        """
        Get formatted webpage contents.
        
        Returns:
            str: Formatted string with webpage title and contents.
        """
        return f"Webpage Title:\n{self.title}\nWebpage Contents:\n{self.text}\n\n"


def get_openai_client() -> OpenAI:
    """
    Initialize and return an OpenAI client.
    
    Returns:
        OpenAI: An initialized OpenAI client.
        
    Raises:
        ValueError: If no OpenAI API key is found.
    """
    load_dotenv(override=True)
    api_key = os.getenv('OPENAI_API_KEY')
    
    if not api_key:
        raise ValueError("No OpenAI API key found. Please set OPENAI_API_KEY in your .env file.")
    
    return OpenAI(api_key=api_key)


def get_link_system_prompt() -> str:
    """
    Create the system prompt for link extraction.
    
    Returns:
        str: System prompt for the link extraction task.
    """
    prompt = "You are provided with a list of links found on a webpage. "
    prompt += "You are able to decide which of the links would be most relevant to include in a brochure about the company, "
    prompt += "such as links to an About page, or a Company page, or Careers/Jobs pages.\n"
    prompt += "You should respond in JSON as in this example:"
    prompt += """
{
    "links": [
        {"type": "about page", "url": "https://full.url/goes/here/about"},
        {"type": "careers page", "url": "https://another.full.url/careers"}
    ]
}
"""
    return prompt


def get_links_user_prompt(website: Website) -> str:
    """
    Create the user prompt for link extraction.
    
    Args:
        website (Website): Website object containing the links.
        
    Returns:
        str: User prompt for link extraction.
    """
    user_prompt = f"Here is the list of links on the website of {website.url} - "
    user_prompt += "please decide which of these are relevant web links for a brochure about the company, "
    user_prompt += "respond with the full https URL in JSON format. "
    user_prompt += "Do not include Terms of Service, Privacy, email links.\n"
    user_prompt += "Links (some might be relative links):\n"
    user_prompt += "\n".join(website.links)
    return user_prompt


def get_links(url: str, model: str = DEFAULT_MODEL) -> Dict:
    """
    Extract relevant links from a website for brochure creation.
    
    Args:
        url (str): The URL to analyze.
        model (str): The OpenAI model to use.
        
    Returns:
        dict: JSON object containing relevant links.
    """
    website = Website(url)
    client = get_openai_client()
    
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": get_link_system_prompt()},
            {"role": "user", "content": get_links_user_prompt(website)}
        ],
        response_format={"type": "json_object"}
    )
    
    result = response.choices[0].message.content
    return json.loads(result)


def get_all_details(url: str, model: str = DEFAULT_MODEL) -> str:
    """
    Gather all website details including landing page and relevant linked pages.
    
    Args:
        url (str): The main website URL.
        model (str): The OpenAI model to use.
        
    Returns:
        str: Combined content from all relevant pages.
    """
    result = "Landing page:\n"
    result += Website(url).get_contents()
    
    links = get_links(url, model)
    print("Found links:", links)
    
    for link in links["links"]:
        result += f"\n\n{link['type']}\n"
        result += Website(link["url"]).get_contents()
    
    return result


def get_brochure_system_prompt() -> str:
    """
    Create the system prompt for brochure generation.
    
    Returns:
        str: System prompt for brochure generation.
    """
    return ("You are an assistant that analyzes the contents of several relevant pages from a company website "
            "and creates a short brochure about the company for prospective customers, investors and recruits. "
            "Respond in markdown. "
            "Include details of company culture, customers and careers/jobs if you have the information.")


def get_brochure_user_prompt(company_name: str, url: str, model: str = DEFAULT_MODEL) -> str:
    """
    Create the user prompt for brochure generation.
    
    Args:
        company_name (str): The name of the company.
        url (str): The company's website URL.
        model (str): The OpenAI model to use.
        
    Returns:
        str: User prompt for brochure generation.
    """
    user_prompt = f"You are looking at a company called: {company_name}\n"
    user_prompt += "Here are the contents of its landing page and other relevant pages; "
    user_prompt += "use this information to build a short brochure of the company in markdown.\n"
    user_prompt += get_all_details(url, model)
    
    # Truncate if more than 5,000 characters
    user_prompt = user_prompt[:5_000]
    
    return user_prompt


def create_brochure(company_name: str, url: str, model: str = DEFAULT_MODEL) -> str:
    """
    Create a marketing brochure for a company based on their website content.
    
    Args:
        company_name (str): The name of the company.
        url (str): The company's website URL.
        model (str, optional): The OpenAI model to use. Defaults to 'gpt-4o-mini'.
        
    Returns:
        str: A markdown-formatted brochure about the company.
        
    Raises:
        requests.RequestException: If there's an error fetching the URL.
        ValueError: If the OpenAI API key is not configured.
    """
    client = get_openai_client()
    
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": get_brochure_system_prompt()},
            {"role": "user", "content": get_brochure_user_prompt(company_name, url, model)}
        ],
    )
    
    result = response.choices[0].message.content
    return result


# Example usage
if __name__ == "__main__":
    try:
        # Example: Create a brochure for MillTech
        company = "MillTech"
        website = "https://milltech.com"
        
        print(f"Creating brochure for {company}...\n")
        brochure = create_brochure(company, website)
        
        print("=" * 80)
        print(f"BROCHURE FOR {company.upper()}")
        print("=" * 80)
        print(brochure)
        
    except Exception as e:
        print(f"Error creating brochure: {e}")