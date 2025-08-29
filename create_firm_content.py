"""
Comprehensive firm content scraping module for Q&A chatbot knowledge base.

This module provides functionality to recursively scrape entire websites,
intelligently filter content using Llama 3.2 via Ollama, and structure
the data in markdown format suitable for training Q&A chatbots.
"""

import os
import json
import time
import hashlib
import requests
from urllib.parse import urljoin, urlparse, urlunparse
from typing import Set, Dict, List, Optional, Tuple
from collections import deque
from datetime import datetime
from bs4 import BeautifulSoup
import ollama
from dotenv import load_dotenv


# Configuration
MAX_DEPTH = 5  # Maximum recursion depth for scraping
MAX_PAGES = 500  # Maximum number of pages to scrape
REQUEST_DELAY = 0.5  # Delay between requests in seconds
TIMEOUT = 10  # Request timeout in seconds
DEFAULT_OLLAMA_MODEL = 'llama3.2'  # Ollama model to use

# Headers for web scraping
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/117.0.0.0 Safari/537.36"
}

# URL patterns to exclude (common non-content pages)
EXCLUDE_PATTERNS = [
    '/cdn-cgi/', '/wp-admin/', '/admin/', '/.well-known/',
    '.pdf', '.jpg', '.jpeg', '.png', '.gif', '.svg', '.ico',
    '.zip', '.tar', '.gz', '.exe', '.dmg', '.pkg',
    '.mp3', '.mp4', '.wav', '.avi', '.mov',
    'mailto:', 'javascript:', 'tel:', '#'
]


class WebPage:
    """
    Represents a single scraped webpage with its content and metadata.
    """
    
    def __init__(self, url: str, depth: int = 0):
        """
        Initialize a WebPage object.
        
        Args:
            url (str): The URL of the page
            depth (int): Depth level in the scraping tree
        """
        self.url = url
        self.depth = depth
        self.title = ""
        self.text = ""
        self.links = []
        self.scraped = False
        self.relevant = True
        self.error = None
        self.content_hash = None
        
    def scrape(self) -> bool:
        """
        Scrape the webpage content.
        
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            response = requests.get(self.url, headers=HEADERS, timeout=TIMEOUT)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Extract title
            self.title = soup.title.string if soup.title else "No title"
            
            # Remove non-content elements
            if soup.body:
                for element in soup.body(['script', 'style', 'nav', 'footer', 'header']):
                    element.decompose()
                    
                # Extract text
                self.text = soup.body.get_text(separator='\n', strip=True)
                
                # Generate content hash to detect duplicates
                self.content_hash = hashlib.md5(self.text.encode()).hexdigest()
                
                # Extract links
                for link in soup.find_all('a', href=True):
                    href = link.get('href')
                    if href:
                        # Convert relative URLs to absolute
                        absolute_url = urljoin(self.url, href)
                        self.links.append(absolute_url)
            
            self.scraped = True
            return True
            
        except requests.RequestException as e:
            self.error = str(e)
            self.scraped = False
            return False
        except Exception as e:
            self.error = f"Unexpected error: {str(e)}"
            self.scraped = False
            return False
    
    def to_markdown(self) -> str:
        """
        Convert page content to structured markdown.
        
        Returns:
            str: Markdown representation of the page
        """
        md = f"## {self.title}\n\n"
        md += f"**URL:** {self.url}\n"
        md += f"**Depth:** {self.depth}\n"
        md += f"**Scraped:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
        
        if self.text:
            md += "### Content\n\n"
            # Limit text length for readability
            text_preview = self.text[:5000] if len(self.text) > 5000 else self.text
            md += text_preview
            if len(self.text) > 5000:
                md += f"\n\n*[Content truncated - {len(self.text)} total characters]*"
        
        md += "\n\n---\n\n"
        return md


class OllamaFilter:
    """
    Uses Ollama with Llama 3.2 to intelligently filter URLs for relevance.
    """
    
    def __init__(self, model: str = DEFAULT_OLLAMA_MODEL):
        """
        Initialize the Ollama filter.
        
        Args:
            model (str): Ollama model to use
        """
        self.model = model
        self._check_ollama()
    
    def _check_ollama(self):
        """Check if Ollama is available and model is installed."""
        try:
            # Check if Ollama is running
            response = ollama.list()
            models = [m['name'] for m in response['models']]
            
            # Check if the required model is available
            if not any(self.model in m for m in models):
                print(f"Model {self.model} not found. Attempting to pull...")
                ollama.pull(self.model)
                
        except Exception as e:
            print(f"Warning: Ollama not available or model not found: {e}")
            print("Falling back to rule-based filtering")
            self.model = None
    
    def is_relevant_for_qa(self, url: str, title: str, text_preview: str) -> Tuple[bool, str]:
        """
        Determine if a page is relevant for Q&A chatbot knowledge base.
        
        Args:
            url (str): Page URL
            title (str): Page title
            text_preview (str): Preview of page content (first 500 chars)
            
        Returns:
            Tuple[bool, str]: (is_relevant, reasoning)
        """
        if not self.model:
            # Fallback to rule-based filtering
            return self._rule_based_filter(url, title, text_preview)
        
        try:
            prompt = f"""You are evaluating web pages for inclusion in a Q&A chatbot knowledge base about a firm/company.

Analyze this page and determine if it contains useful information for answering questions about the company.

URL: {url}
Title: {title}
Content Preview: {text_preview[:500]}

Consider:
1. Does it contain factual information about the company (products, services, team, history, values)?
2. Is it substantive content (not just navigation, legal disclaimers, or cookie notices)?
3. Would it help answer user questions about the company?

Respond with JSON:
{{"relevant": true/false, "reasoning": "brief explanation"}}"""

            response = ollama.chat(
                model=self.model,
                messages=[{'role': 'user', 'content': prompt}],
                format='json'
            )
            
            result = json.loads(response['message']['content'])
            return result.get('relevant', True), result.get('reasoning', 'No reasoning provided')
            
        except Exception as e:
            print(f"Ollama evaluation error: {e}")
            # Fall back to being inclusive
            return True, "Error in evaluation - including by default"
    
    def _rule_based_filter(self, url: str, title: str, text_preview: str) -> Tuple[bool, str]:
        """
        Simple rule-based filtering as fallback.
        
        Args:
            url (str): Page URL
            title (str): Page title  
            text_preview (str): Preview of page content
            
        Returns:
            Tuple[bool, str]: (is_relevant, reasoning)
        """
        url_lower = url.lower()
        
        # Skip common non-content pages
        skip_keywords = ['privacy', 'terms', 'cookie', 'legal', 'sitemap', 'search']
        if any(keyword in url_lower for keyword in skip_keywords):
            return False, "Common non-content page"
        
        # Include likely content pages
        include_keywords = ['about', 'product', 'service', 'team', 'blog', 'news', 
                          'career', 'contact', 'solution', 'pricing', 'faq']
        if any(keyword in url_lower for keyword in include_keywords):
            return True, "Likely content page"
        
        # Check if there's substantial text
        if len(text_preview) < 100:
            return False, "Too little content"
        
        return True, "Default inclusion"


class FirmContentScraper:
    """
    Main scraper class for comprehensive firm content extraction.
    """
    
    def __init__(self, firm_name: str, base_url: str, max_depth: int = MAX_DEPTH, 
                 max_pages: int = MAX_PAGES):
        """
        Initialize the scraper.
        
        Args:
            firm_name (str): Name of the firm
            base_url (str): Starting URL for scraping
            max_depth (int): Maximum recursion depth
            max_pages (int): Maximum number of pages to scrape
        """
        self.firm_name = firm_name
        self.base_url = base_url
        self.base_domain = urlparse(base_url).netloc
        self.max_depth = max_depth
        self.max_pages = max_pages
        
        self.visited_urls: Set[str] = set()
        self.content_hashes: Set[str] = set()
        self.pages: List[WebPage] = []
        self.filter = OllamaFilter()
        
    def _normalize_url(self, url: str) -> str:
        """
        Normalize URL for consistency.
        
        Args:
            url (str): URL to normalize
            
        Returns:
            str: Normalized URL
        """
        parsed = urlparse(url.lower())
        # Remove fragment and normalize path
        normalized = urlunparse((
            parsed.scheme,
            parsed.netloc,
            parsed.path.rstrip('/'),
            parsed.params,
            parsed.query,
            ''  # Remove fragment
        ))
        return normalized
    
    def _should_scrape_url(self, url: str, current_depth: int) -> bool:
        """
        Determine if a URL should be scraped.
        
        Args:
            url (str): URL to check
            current_depth (int): Current depth in recursion
            
        Returns:
            bool: True if URL should be scraped
        """
        # Check depth limit
        if current_depth >= self.max_depth:
            return False
        
        # Check page limit
        if len(self.visited_urls) >= self.max_pages:
            return False
        
        # Normalize and check if already visited
        normalized = self._normalize_url(url)
        if normalized in self.visited_urls:
            return False
        
        # Check if URL is from same domain
        parsed = urlparse(url)
        if parsed.netloc != self.base_domain:
            return False
        
        # Check exclusion patterns
        url_lower = url.lower()
        if any(pattern in url_lower for pattern in EXCLUDE_PATTERNS):
            return False
        
        return True
    
    def scrape_recursive(self, url: str = None, depth: int = 0) -> None:
        """
        Recursively scrape website starting from given URL.
        
        Args:
            url (str): Starting URL (uses base_url if None)
            depth (int): Current depth level
        """
        if url is None:
            url = self.base_url
        
        # BFS queue for URLs to process
        queue = deque([(url, depth)])
        
        while queue and len(self.visited_urls) < self.max_pages:
            current_url, current_depth = queue.popleft()
            
            # Check if we should scrape this URL
            if not self._should_scrape_url(current_url, current_depth):
                continue
            
            # Mark as visited
            normalized_url = self._normalize_url(current_url)
            self.visited_urls.add(normalized_url)
            
            # Create and scrape page
            page = WebPage(current_url, current_depth)
            print(f"Scraping [{current_depth}/{self.max_depth}]: {current_url}")
            
            if page.scrape():
                # Check for duplicate content
                if page.content_hash in self.content_hashes:
                    print(f"  → Duplicate content, skipping")
                    continue
                
                self.content_hashes.add(page.content_hash)
                
                # Check relevance using Ollama
                text_preview = page.text[:500] if page.text else ""
                is_relevant, reasoning = self.filter.is_relevant_for_qa(
                    page.url, page.title, text_preview
                )
                
                page.relevant = is_relevant
                print(f"  → {'✓ Relevant' if is_relevant else '✗ Not relevant'}: {reasoning}")
                
                if is_relevant:
                    self.pages.append(page)
                    
                    # Add child links to queue
                    if current_depth < self.max_depth - 1:
                        for link in page.links:
                            if self._should_scrape_url(link, current_depth + 1):
                                queue.append((link, current_depth + 1))
            else:
                print(f"  → Failed: {page.error}")
            
            # Rate limiting
            time.sleep(REQUEST_DELAY)
    
    def generate_structured_markdown(self) -> str:
        """
        Generate structured markdown document from scraped content.
        
        Returns:
            str: Complete markdown document
        """
        md = f"# {self.firm_name} - Complete Knowledge Base\n\n"
        md += f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        md += f"**Base URL:** {self.base_url}\n"
        md += f"**Pages Scraped:** {len(self.pages)}\n"
        md += f"**Total URLs Visited:** {len(self.visited_urls)}\n\n"
        
        md += "## Table of Contents\n\n"
        
        # Group pages by depth level
        pages_by_depth = {}
        for page in self.pages:
            if page.depth not in pages_by_depth:
                pages_by_depth[page.depth] = []
            pages_by_depth[page.depth].append(page)
        
        # Generate TOC
        for depth in sorted(pages_by_depth.keys()):
            md += f"### Level {depth}\n"
            for page in pages_by_depth[depth]:
                md += f"- [{page.title}]({page.url})\n"
            md += "\n"
        
        md += "---\n\n"
        md += "## Content\n\n"
        
        # Add all page content
        for depth in sorted(pages_by_depth.keys()):
            md += f"### Depth Level {depth}\n\n"
            for page in pages_by_depth[depth]:
                md += page.to_markdown()
        
        return md
    
    def save_to_file(self, filename: str = None) -> str:
        """
        Save scraped content to markdown file.
        
        Args:
            filename (str): Output filename (auto-generated if None)
            
        Returns:
            str: Path to saved file
        """
        if filename is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            safe_name = "".join(c for c in self.firm_name if c.isalnum() or c in (' ', '-', '_')).rstrip()
            filename = f"{safe_name}_knowledge_base_{timestamp}.md"
        
        content = self.generate_structured_markdown()
        
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(content)
        
        return filename


def create_firm_content(firm_name: str, url: str, max_depth: int = 3, 
                       max_pages: int = 100, output_file: str = None) -> str:
    """
    Create comprehensive firm content knowledge base from website.
    
    Args:
        firm_name (str): Name of the firm/company
        url (str): Starting URL for scraping
        max_depth (int): Maximum recursion depth (default: 3)
        max_pages (int): Maximum pages to scrape (default: 100)
        output_file (str): Output filename (auto-generated if None)
        
    Returns:
        str: Structured markdown content
        
    Example:
        content = create_firm_content("Acme Corp", "https://acmecorp.com")
    """
    print(f"\n{'='*60}")
    print(f"Starting comprehensive content extraction for {firm_name}")
    print(f"Base URL: {url}")
    print(f"Max Depth: {max_depth}, Max Pages: {max_pages}")
    print(f"{'='*60}\n")
    
    # Initialize and run scraper
    scraper = FirmContentScraper(firm_name, url, max_depth, max_pages)
    scraper.scrape_recursive()
    
    print(f"\n{'='*60}")
    print(f"Scraping Complete!")
    print(f"Pages scraped: {len(scraper.pages)}")
    print(f"Total URLs visited: {len(scraper.visited_urls)}")
    print(f"{'='*60}\n")
    
    # Generate markdown
    markdown_content = scraper.generate_structured_markdown()
    
    # Save to file if requested
    if output_file or output_file is None:
        saved_file = scraper.save_to_file(output_file)
        print(f"Content saved to: {saved_file}")
    
    return markdown_content


# Example usage
if __name__ == "__main__":
    # Load environment variables
    load_dotenv(override=True)
    
    # Example: Scrape a company website
    try:
        firm = "MillTech"
        website = "https://milltech.com"
        
        print(f"Creating knowledge base for {firm}...")
        content = create_firm_content(
            firm_name=firm,
            url=website,
            max_depth=50,  # Limit depth for testing
            max_pages=5000  # Limit pages for testing
        )
        
        print("\n" + "="*60)
        print("PREVIEW OF GENERATED CONTENT (first 1000 chars):")
        print("="*60)
        print(content[:1000])
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()