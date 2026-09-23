import requests
from bs4 import BeautifulSoup
from app.tools.registry import ToolRegistry
from app.utils.logger import get_logger

logger = get_logger("voltix.tools.web_inspector")

@ToolRegistry.register(
    name="fetch_local_web_page",
    description="Fetches and extracts clean text, headings, and links from a local or remote web page URL."
)
def fetch_local_web_page(url: str, max_chars: int = 4000) -> dict:
    """
    Fetches HTML content from a local or remote URL, strips scripts/styles, and extracts readable text.
    
    Args:
        url: The full HTTP/HTTPS URL (e.g. 'http://127.0.0.1:5000' or 'http://localhost:5000/api/models').
        max_chars: Maximum characters to return in the content body.
    """
    try:
        headers = {"User-Agent": "VoltixAgent/2.0 (EEE Assistant)"}
        response = requests.get(url, timeout=10, headers=headers)
        response.raise_for_status()

        soup = BeautifulSoup(response.text, "html.parser")

        # Strip scripts, styles, noscript, and SVGs
        for element in soup(["script", "style", "noscript", "svg", "iframe"]):
            element.decompose()

        title = soup.title.string.strip() if soup.title and soup.title.string else "No title"

        # Extract headings
        headings = [h.get_text(strip=True) for h in soup.find_all(["h1", "h2", "h3"]) if h.get_text(strip=True)]

        # Extract clean text
        text = soup.get_text(separator="\n", strip=True)
        truncated_text = text[:max_chars] + ("\n...[content truncated]" if len(text) > max_chars else "")

        # Extract links
        links = []
        for a in soup.find_all("a", href=True):
            link_text = a.get_text(strip=True)
            if link_text and not a["href"].startswith("#") and not a["href"].startswith("javascript:"):
                links.append({"text": link_text, "href": a["href"]})

        return {
            "status": "success",
            "url": url,
            "title": title,
            "headings": headings[:8],
            "links_sample": links[:8],
            "content": truncated_text,
            "char_count": len(text)
        }
    except Exception as e:
        logger.warning(f"Error fetching URL {url}: {e}")
        return {
            "status": "error",
            "url": url,
            "error": f"Failed to fetch web page at {url}: {str(e)}"
        }
