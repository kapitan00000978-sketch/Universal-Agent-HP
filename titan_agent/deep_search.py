import asyncio
import logging
import re
import urllib.request
from typing import Any

try:
    from ddgs import DDGS
except ImportError:
    from duckduckgo_search import DDGS


class DeepSearchEngine:
    """
    Autonomous Deep Search Engine:
    Decomposes queries, gathers cross-source intelligence, deep scrapes pages,
    and synthesizes comprehensive research.
    """
    def __init__(self, max_subqueries: int = 3, max_pages_to_scrape: int = 3):
        self.max_subqueries = max_subqueries
        self.max_pages_to_scrape = max_pages_to_scrape

    async def _search_query(self, query: str, max_results: int = 4) -> list[dict[str, str]]:
        loop = asyncio.get_running_loop()
        def _exec():
            results = []
            try:
                with DDGS() as ddgs:
                    for r in ddgs.text(query, max_results=max_results):
                        results.append({
                            "title": r.get("title", ""),
                            "url": r.get("href", ""),
                            "snippet": r.get("body", "")
                        })
            except Exception as e:
                # Search backends raise provider-specific exceptions (not just
                # OSError/RuntimeError) for timeouts, rate limits and offline
                # networks. Treat an unavailable source as an empty result so
                # one failed backend does not abort the entire research task.
                log = logging.getLogger(__name__)
                log.debug("DDGS search failed for %s: %s", query, e)
            return results
        return await loop.run_in_executor(None, _exec)

    async def _scrape_url(self, url: str) -> str:
        loop = asyncio.get_running_loop()
        def _exec():
            try:
                req = urllib.request.Request(
                    url,
                    headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
                )
                with urllib.request.urlopen(req, timeout=10) as resp:
                    html = resp.read().decode('utf-8', errors='ignore')
                text = re.sub(r'<script.*?</script>', '', html, flags=re.DOTALL | re.IGNORECASE)
                text = re.sub(r'<style.*?</style>', '', text, flags=re.DOTALL | re.IGNORECASE)
                text = re.sub(r'<[^>]+>', ' ', text)
                clean_lines = [line.strip() for line in text.splitlines() if line.strip()]
                return "\n".join(clean_lines)[:4000]
            except (urllib.error.URLError, OSError, ValueError) as e:
                log = logging.getLogger(__name__)
                log.debug("Scrape failed for %s: %s", url, e)
                return ""
        return await loop.run_in_executor(None, _exec)

    async def run(self, topic: str) -> dict[str, Any]:
        """
        Executes a multi-angle deep search on a topic.
        """
        # Formulate search sub-queries
        sub_queries = [
            topic,
            f"{topic} overview technical details",
            f"{topic} latest updates 2025 2026"
        ]

        # Execute parallel searches
        search_tasks = [self._search_query(q) for q in sub_queries]
        search_results_lists = await asyncio.gather(*search_tasks)

        unique_sources = {}
        for r_list in search_results_lists:
            for item in r_list:
                url = item.get("url")
                if url and url not in unique_sources:
                    unique_sources[url] = item

        sources_list = list(unique_sources.values())

        # Scrape top pages for deep context
        scrape_tasks = [self._scrape_url(s["url"]) for s in sources_list[:self.max_pages_to_scrape]]
        page_contents = await asyncio.gather(*scrape_tasks)

        scraped_data = []
        for i, content in enumerate(page_contents):
            if content and i < len(sources_list):
                scraped_data.append({
                    "title": sources_list[i]["title"],
                    "url": sources_list[i]["url"],
                    "content": content
                })

        return {
            "topic": topic,
            "sub_queries": sub_queries,
            "total_sources_found": len(sources_list),
            "sources": sources_list[:8],
            "deep_pages": scraped_data
        }