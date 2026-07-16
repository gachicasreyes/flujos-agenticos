import arxiv
import json
import os
from typing import List
from mcp.server.fastmcp import FastMCP
from typing import Literal 
from tavily import TavilyClient

try:
    from dotenv import load_dotenv
except ImportError:
    load_dotenv = None


PAPER_DIR = "papers"

if load_dotenv:
    load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), ".env"))

# Clase orquestadora de MCP
mcp = FastMCP("research")


#####
# Decoradores de herramientas MCP 

@mcp.tool()
def search_papers(topic: str, max_results: int = 5) -> List[str]:
    """
    Búsqueda de temas en arXiv a través de palabras clave (nota: no es búsqueda semántica!!).
    
    Args:
        topic: The topic to search for
        max_results: Maximum number of results to retrieve (default: 5)
        
    Returns:
        List of paper IDs found in the search
    """
    
    # Inicio de cliente de arXiv
    client = arxiv.Client()

    # Función de búsqueda de arXiv
    search = arxiv.Search(
        query = topic,
        max_results = max_results,
        sort_by = arxiv.SortCriterion.Relevance
    )

    papers = client.results(search)
    
    # Creación de directorio para tema consultado
    path = os.path.join(PAPER_DIR, topic.lower().replace(" ", "_"))
    os.makedirs(path, exist_ok=True)
    
    file_path = os.path.join(path, "papers_info.json")

    # Carga de información de papers
    try:
        with open(file_path, "r") as json_file:
            papers_info = json.load(json_file)
    except (FileNotFoundError, json.JSONDecodeError):
        papers_info = {}

    # Procesamiento de información de cada paper
    paper_ids = []
    for paper in papers:
        paper_ids.append(paper.get_short_id())
        paper_info = {
            'title': paper.title,
            'authors': [author.name for author in paper.authors],
            'summary': paper.summary,
            'pdf_url': paper.pdf_url,
            'published': str(paper.published.date())
        }
        papers_info[paper.get_short_id()] = paper_info
    
    # Almacenar información de papers como archivo JSON
    with open(file_path, "w") as json_file:
        json.dump(papers_info, json_file, indent=2)
    
    print(f"Results are saved in: {file_path}")
    
    return paper_ids

@mcp.tool()
def extract_info(paper_id: str) -> str:
    """
    Search for information about a specific paper across all topic directories.
    
    Args:
        paper_id: The ID of the paper to look for
        
    Returns:
        JSON string with paper information if found, error message if not found
    """
 
    for item in os.listdir(PAPER_DIR):
        item_path = os.path.join(PAPER_DIR, item)
        if os.path.isdir(item_path):
            file_path = os.path.join(item_path, "papers_info.json")
            if os.path.isfile(file_path):
                try:
                    with open(file_path, "r") as json_file:
                        papers_info = json.load(json_file)
                        if paper_id in papers_info:
                            return json.dumps(papers_info[paper_id], indent=2)
                except (FileNotFoundError, json.JSONDecodeError) as e:
                    print(f"Error reading {file_path}: {str(e)}")
                    continue
    
    return f"There's no saved information related to paper {paper_id}."


@mcp.tool()
def web_search(
    query: str,
    max_results: int = 5,
    search_depth: Literal["basic", "advanced"] = "basic",
    include_answer: bool = False,
    include_raw_content: bool = False,
) -> dict:
    """
    Search the web using Tavily and return structured results.
    Use this for current information, documentation, news, and general web research.
    """

    max_results = max(1, min(max_results, 10))
    api_key = os.getenv("TAVILY_API_KEY")
    if not api_key:
        return {
            "query": query,
            "answer": None,
            "results": [],
            "error": "TAVILY_API_KEY is not set. Add it to .env or the server environment.",
        }

    client = TavilyClient(api_key=api_key)

    response = client.search(
        query=query,
        max_results=max_results,
        search_depth=search_depth,
        include_answer=include_answer,
        include_raw_content=include_raw_content,
    )

    results = []
    for item in response.get("results", []):
        results.append(
            {
                "title": item.get("title"),
                "url": item.get("url"),
                "content": item.get("content"),
                "score": item.get("score"),
            }
        )

    return {
        "query": query,
        "answer": response.get("answer"),
        "results": results,
    }

if __name__ == "__main__":
    # Initialize and run the server
    mcp.run(transport='stdio')
