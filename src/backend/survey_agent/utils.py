"""Utilities and Tools for the survey agent system.

This module provides various utility functions and classes that support the
operation of the survey agent, including paper retrieval, decup overlap checking,
and content summarization tools.
"""



from datetime import datetime
from typing import List

import arxiv
import requests
from langchain.chat_models import init_chat_model
from langchain.tools import tool




# ========== load env variables
import os
from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv())  # read local .env file

from survey_agent.state_scope import BaseSearchRequest, Paper


### ========== summarization models
summarization_model = init_chat_model(
    model='Pro/deepseek-ai/DeepSeek-R1',
    model_provider='openai',
    base_url=os.getenv('LLM_BASE_URL'),
    api_key=os.getenv('LLM_API_KEY'),
    temperature=0,
    max_retries=3,
)


### ========== utilities

def get_today_str() -> str:
    """Get current date in a human-readable format."""
    # Avoid %-d for Windows compatibility.
    return datetime.now().strftime("%a %b %d, %Y").replace(" 0", " ")


### ========== tools

# arxiv retrieval tool
@tool("search_arxiv_papers", args_schema=BaseSearchRequest)
def search_arxiv_papers(
    query: str,
    max_results: int = 5,
    date_from: datetime | None = None,
    date_to: datetime | None = None,
) -> List[dict]:
    """
    Search for papers on arXiv.org using the arxiv SDK.
    """
    full_query = query
    if (date_from or date_to) and "submittedDate:" not in query:
        start = (date_from or datetime(1900, 1, 1)).strftime("%Y%m%d%H%M")
        end = (date_to or datetime.utcnow()).strftime("%Y%m%d%H%M")
        full_query = f"({query}) AND submittedDate:[{start} TO {end}]"

    client = arxiv.Client(
        page_size=max_results,
        delay_seconds=3.0,
        num_retries=3,
    )

    search = arxiv.Search(
        query=full_query,
        max_results=max_results,
        sort_by=arxiv.SortCriterion.Relevance,
        sort_order=arxiv.SortOrder.Descending,
    )

    results: List[dict] = []
    try:
        for result in client.results(search):
            authors = [author.name for author in result.authors]
            published_year = result.published.year if result.published else None
            venue = f"arXiv:{result.primary_category}" if result.primary_category else "arXiv"
            summary = result.summary.replace("\n", " ") if result.summary else None
            title = result.title.replace("\n", " ") if result.title else None
            source_id = result.get_short_id() if hasattr(result, "get_short_id") else None

            results.append(
                Paper(
                    source="arxiv",
                    source_id=source_id,
                    title=title,
                    authors=authors,
                    abstract=summary,
                    year=published_year,
                    venue=venue,
                    doi=result.doi,
                    url=result.entry_id,
                    pdf_url=result.pdf_url,
                ).model_dump()
            )
    except Exception:
        return []

    return results
# acm_dl retrieval tool
@tool("search_acm_papers", args_schema=BaseSearchRequest)
def search_acm_papers(
    query: str,
    max_results: int = 5,
    date_from: datetime | None = None,
    date_to: datetime | None = None,
) -> List[dict]:
    """
    Search ACM papers via Crossref and return normalized metadata.
    """
    base_url = "https://api.crossref.org/works"
    filters = ["member:320"]

    if date_from:
        filters.append(f"from-pub-date:{date_from.date().isoformat()}")
    if date_to:
        filters.append(f"until-pub-date:{date_to.date().isoformat()}")

    base_params = {
        "filter": ",".join(filters),
        "rows": max_results,
        "sort": "relevance",
        "mailto": os.getenv("CROSSREF_MAILTO", "research-agent@example.com"),
    }

    items = []
    for params in (
        {**base_params, "query.bibliographic": query},
        {**base_params, "query": query},
    ):
        try:
            response = requests.get(base_url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
        except requests.RequestException:
            continue

        items = data.get("message", {}).get("items", [])
        if items:
            break

    if not items:
        return []
    results: List[dict] = []

    for item in items:
        title_list = item.get("title", [])
        title = title_list[0] if title_list else None
        doi = item.get("DOI")

        authors = []
        for author in item.get("author", []):
            given = author.get("given", "")
            family = author.get("family", "")
            full_name = f"{given} {family}".strip()
            if full_name:
                authors.append(full_name)

        published = None
        date_parts = item.get("published-print", {}).get("date-parts") or item.get(
            "published-online", {}
        ).get("date-parts")
        if date_parts and date_parts[0]:
            published = date_parts[0][0]

        venue_list = item.get("container-title", [])
        venue = venue_list[0] if venue_list else None

        landing_page = f"https://dl.acm.org/doi/{doi}" if doi else None
        pdf_link = f"https://dl.acm.org/doi/pdf/{doi}" if doi else None

        results.append(
            Paper(
                source="acm",
                source_id=doi,
                title=title,
                authors=authors,
                abstract=item.get("abstract"),
                year=published,
                venue=venue,
                doi=doi,
                url=landing_page,
                pdf_url=pdf_link,
            ).model_dump()
        )

    return results
