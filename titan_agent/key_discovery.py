import aiohttp
import asyncio
import logging
import re
from typing import Tuple, List

log = logging.getLogger(__name__)

async def try_fetch_models(base_url: str, api_key: str, is_gemini: bool = False) -> List[str]:
    headers = {"Authorization": f"Bearer {api_key}"}
    if is_gemini:
        # Gemini uses a different endpoint if using Google's direct API, but we use openai compatibility
        # For Gemini OpenAI compat, standard /v1/models works if base_url ends in /openai
        pass
    
    url = f"{base_url.rstrip('/')}/models"
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(url, headers=headers, timeout=5.0) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    models = []
                    if "data" in data:
                        models = [m["id"] for m in data["data"]]
                    return models
    except Exception as e:
        log.debug(f"Failed to fetch models from {base_url}: {e}")
    return []

async def discover_api_key(key: str) -> Tuple[str, str, List[str]]:
    """
    Given an API key, guess the provider, test the /models endpoint, 
    and return (provider_name, base_url, list_of_models).
    """
    key = key.strip()
    
    # 1. Groq
    if key.startswith("gsk_"):
        models = await try_fetch_models("https://api.groq.com/openai/v1", key)
        if models:
            return "groq", "https://api.groq.com/openai/v1", models
            
    # 2. Gemini
    elif key.startswith("AIza"):
        models = await try_fetch_models("https://generativelanguage.googleapis.com/v1beta/openai", key)
        if models:
            return "gemini", "https://generativelanguage.googleapis.com/v1beta/openai", models
            
    # 3. DeepSeek or OpenAI or OpenRouter (all start with sk-)
    elif key.startswith("sk-"):
        # Try DeepSeek first
        models = await try_fetch_models("https://api.deepseek.com/v1", key)
        if models:
            return "deepseek", "https://api.deepseek.com/v1", models
            
        # Try OpenAI
        models = await try_fetch_models("https://api.openai.com/v1", key)
        if models:
            return "openai", "https://api.openai.com/v1", models
            
        # Try OpenRouter
        models = await try_fetch_models("https://openrouter.ai/api/v1", key)
        if models:
            return "openrouter", "https://openrouter.ai/api/v1", models
            
        # Try Mistral
        models = await try_fetch_models("https://api.mistral.ai/v1", key)
        if models:
            return "mistral", "https://api.mistral.ai/v1", models
            
    return "", "", []
