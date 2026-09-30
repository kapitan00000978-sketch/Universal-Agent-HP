import asyncio
import json
import logging
import re
from typing import Any

import aiohttp

from .config import (
    COMPLETIONS_API_KEY,
    COMPLETIONS_BASE_URL,
    DEEPSEEK_API_KEY,
    DEFAULT_MODEL,
    DEFAULT_PROVIDER,
    GLM_API_KEY,
    GLM_BASE_URL,
    GROQ_API_KEY,
    KIMI_API_KEY,
    KIMI_BASE_URL,
    OLLAMA_BASE_URL,
    OMNI_API_KEY,
    OMNI_BASE_URL,
    OPENAI_API_KEY,
    OPENAI_BASE_URL,
    OPENROUTER_API_KEY,
    GEMINI_API_KEY,
    GEMINI_BASE_URL,
    SAMBANOVA_API_KEY,
    SAMBANOVA_BASE_URL,
    GITHUB_TOKEN,
    GITHUB_MODELS_BASE_URL,
    MISTRAL_API_KEY,
    MISTRAL_BASE_URL,
    HUGGINGFACE_API_KEY,
    HUGGINGFACE_BASE_URL,
    EXTRA_LLM_X_API_KEY,
    EXTRA_LLM_X_BASE_URL,
    EXTRA_LLM_X_MODEL,
    TOGETHER_API_KEY,
    TOGETHER_BASE_URL,
    CEREBRAS_API_KEY,
    CEREBRAS_BASE_URL,
    COHERE_API_KEY,
    COHERE_BASE_URL,
    provider_default_model,
    provider_fallback_chain,
    provider_fallback_models,
)
from .token_limit import TokenRateLimiter, estimate_tokens

log = logging.getLogger(__name__)

# Providers that need an API key before they can serve a request, mapped to the
# module-level key variable they read (read via globals() so tests/runtime can
# swap keys live). Everything NOT listed here cannot take part in the fallback
# chain unless it already has its key baked in.
_PROVIDER_API_KEY_ATTR = {
    "openrouter": "OPENROUTER_API_KEY",
    "groq": "GROQ_API_KEY",
    "gemini": "GEMINI_API_KEY",
    "sambanova": "SAMBANOVA_API_KEY",
    "github": "GITHUB_TOKEN",
    "mistral": "MISTRAL_API_KEY",
    "huggingface": "HUGGINGFACE_API_KEY",
    "deepseek": "DEEPSEEK_API_KEY",
    "kimi": "KIMI_API_KEY",
    "glm": "GLM_API_KEY",
    "omni": "OMNI_API_KEY",
    "completions": "COMPLETIONS_API_KEY",
    "openai": "OPENAI_API_KEY",
    "extra-llm-x": "EXTRA_LLM_X_API_KEY",
    "extra_llm_x": "EXTRA_LLM_X_API_KEY",
    "elx": "EXTRA_LLM_X_API_KEY",
    "together": "TOGETHER_API_KEY",
    "cerebras": "CEREBRAS_API_KEY",
    "cohere": "COHERE_API_KEY",
}
# Local / browser / free providers never need a key.
_LOCAL_PROVIDERS = frozenset({"ollama", "lmstudio", "puter", "g4f", "tgpt", "laya", "laya-mlx", "extra-llm-x", "extra_llm_x", "elx"})


class LLMResponse:
    def __init__(self, content: str = "", tool_calls: list[dict[str, Any]] | None = None, thoughts: str = ""):
        self.content = content or ""
        self.tool_calls = tool_calls or []
        self.thoughts = thoughts or ""

    def to_dict(self):
        return {
            "content": self.content,
            "tool_calls": self.tool_calls,
            "thoughts": self.thoughts
        }

class LLMClient:
    def __init__(self, provider: str | None = None, model: str | None = None):
        self.provider = provider or DEFAULT_PROVIDER
        self.model = model or DEFAULT_MODEL
        self._setup_credentials()
        # Global token throughput guardrail (214,000 tokens/s cap).
        self.token_limiter = TokenRateLimiter()

    def _setup_credentials(self):
        if self.provider == "openrouter":
            self.base_url = "https://openrouter.ai/api/v1"
            self.api_key = OPENROUTER_API_KEY
        elif self.provider == "groq":
            self.base_url = "https://api.groq.com/openai/v1"
            self.api_key = GROQ_API_KEY
        elif self.provider == "deepseek":
            self.base_url = "https://api.deepseek.com/v1"
            self.api_key = DEEPSEEK_API_KEY
        elif self.provider == "kimi":
            # Kimi K3 (Moonshot) — frontier-class open-weight model, OpenAI-compatible.
            self.base_url = KIMI_BASE_URL
            self.api_key = KIMI_API_KEY
        elif self.provider == "glm":
            # GLM-5.3 Flash (Zhipu) — cheap strong brain, OpenAI-compatible.
            self.base_url = GLM_BASE_URL
            self.api_key = GLM_API_KEY
        elif self.provider == "omni":
            # OmniRoute — self-hosted AI gateway (localhost:20128) with smart
            # auto-routing models (auto, auto/coding, auto/fast, ...).
            self.base_url = OMNI_BASE_URL
            self.api_key = OMNI_API_KEY
        elif self.provider == "ollama":
            self.base_url = f"{OLLAMA_BASE_URL}/v1"
            self.api_key = "ollama"
        elif self.provider == "lmstudio":
            self.base_url = "http://localhost:1234/v1"
            self.api_key = "lm-studio"
        elif self.provider == "puter":
            # Puter.js runs fully client-side in the browser (free DeepSeek etc.).
            self.base_url = ""
            self.api_key = ""
        elif self.provider == "g4f":
            # GPT4Free library (no key required)
            self.base_url = ""
            self.api_key = ""
        elif self.provider == "tgpt":
            # Python-tGPT library (no key required)
            self.base_url = ""
            self.api_key = ""
        elif self.provider in {"laya", "laya-mlx"}:
            # Laya is a local typed-decision model, not a chat-completion model.
            self.base_url = ""
            self.api_key = ""
        elif self.provider == "completions":
            # Completions.me — free OpenAI-compatible gateway (Claude Opus/GPT-5/Gemini/Grok).
            self.base_url = COMPLETIONS_BASE_URL
            self.api_key = COMPLETIONS_API_KEY
        elif self.provider == "gemini":
            # Google AI Studio (Gemini 2.0 Flash / Pro)
            self.base_url = GEMINI_BASE_URL
            self.api_key = GEMINI_API_KEY
        elif self.provider == "sambanova":
            # SambaNova Cloud (LLaMA 3.3 70B, Qwen 2.5 Coder)
            self.base_url = SAMBANOVA_BASE_URL
            self.api_key = SAMBANOVA_API_KEY
        elif self.provider == "github":
            # GitHub Models (GPT-4o, Phi-4, LLaMA)
            self.base_url = GITHUB_MODELS_BASE_URL
            self.api_key = GITHUB_TOKEN
        elif self.provider == "mistral":
            # Mistral AI / Codestral
            self.base_url = MISTRAL_BASE_URL
            self.api_key = MISTRAL_API_KEY
        elif self.provider == "huggingface":
            # Hugging Face Serverless Inference
            self.base_url = HUGGINGFACE_BASE_URL
            self.api_key = HUGGINGFACE_API_KEY
        elif self.provider in ("extra-llm-x", "extra_llm_x", "elx"):
            # Extra LLM X — Unified Autonomous AI Gateway
            base_url = EXTRA_LLM_X_BASE_URL
            if "localhost:3000" in base_url or "127.0.0.1:3000" in base_url:
                try:
                    import socket
                    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                        s.settimeout(0.3)
                        if s.connect_ex(("127.0.0.1", 3000)) != 0:
                            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s2:
                                s2.settimeout(0.3)
                                if s2.connect_ex(("127.0.0.1", 3001)) == 0:
                                    base_url = base_url.replace(":3000", ":3001")
                except Exception:
                    pass
            self.base_url = base_url
            self.api_key = EXTRA_LLM_X_API_KEY or "elx-master-admin-key"
            if self.model == DEFAULT_MODEL:
                self.model = EXTRA_LLM_X_MODEL
        elif self.provider == "together":
            self.base_url = TOGETHER_BASE_URL
            self.api_key = TOGETHER_API_KEY
        elif self.provider == "cerebras":
            self.base_url = CEREBRAS_BASE_URL
            self.api_key = CEREBRAS_API_KEY
        elif self.provider == "cohere":
            self.base_url = COHERE_BASE_URL
            self.api_key = COHERE_API_KEY
        else:
            self.base_url = OPENAI_BASE_URL
            self.api_key = OPENAI_API_KEY

    def set_model(self, provider: str, model: str, api_key: str | None = None, base_url: str | None = None):
        self.provider = provider
        self.model = model
        if api_key:
            self.api_key = api_key
        if base_url:
            self.base_url = base_url
        else:
            self._setup_credentials()

    async def detect_local_models() -> dict[str, list[str]]:
        """Scans Ollama and LM Studio for local models."""
        log = logging.getLogger(__name__)
        results = {"ollama": [], "lmstudio": []}
        async with aiohttp.ClientSession() as session:
            # Check Ollama
            try:
                async with session.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=aiohttp.ClientTimeout(total=2)) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        results["ollama"] = [m["name"] for m in data.get("models", [])]
            except (aiohttp.ClientError, asyncio.TimeoutError) as e:
                log.debug("Ollama model detection failed: %s", e)
            # Check LM Studio
            try:
                async with session.get("http://localhost:1234/v1/models", timeout=aiohttp.ClientTimeout(total=2)) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        results["lmstudio"] = [m["id"] for m in data.get("data", [])]
            except (aiohttp.ClientError, asyncio.TimeoutError) as e:
                log.debug("LM Studio model detection failed: %s", e)
        return results

    def _extract_thoughts_and_tools(self, text: str) -> tuple[str, str, list[dict[str, Any]]]:
        thoughts = ""
        tool_calls = []

        # Extract thoughts <thought>...</thought> or <think>...</think> (DeepSeek R1 format)
        thought_match = re.search(r"<(?:thought|scratchpad|think)>(.*?)</(?:thought|scratchpad|think)>", text, re.DOTALL | re.IGNORECASE)
        if thought_match:
            thoughts = thought_match.group(1).strip()
            text = re.sub(r"<(?:thought|scratchpad|think)>.*?</(?:thought|scratchpad|think)>", "", text, flags=re.DOTALL | re.IGNORECASE).strip()

        # Extract tool calls in Hermes 3 format: <tool_call>{"name": "...", "arguments": {...}}</tool_call>
        tool_matches = re.finditer(r"<tool_call>\s*(.*?)\s*</tool_call>", text, re.DOTALL | re.IGNORECASE)
        for match in tool_matches:
            raw_call = match.group(1).strip()
            try:
                parsed = json.loads(raw_call)
                if "name" in parsed:
                    tool_calls.append({
                        "id": f"call_{len(tool_calls)+1}",
                        "type": "function",
                        "function": {
                            "name": parsed["name"],
                            "arguments": json.dumps(parsed.get("arguments", parsed.get("parameters", {})))
                        }
                    })
            except json.JSONDecodeError:
                pass

        text = re.sub(r"<tool_call>.*?</tool_call>", "", text, flags=re.DOTALL | re.IGNORECASE).strip()
        return text, thoughts, tool_calls

    def _build_fallback_chain(self) -> list[tuple[str, str]]:
        """(provider, model) pairs to try, primary first, deduplicated.

        The chain comes from ``TITAN_PROVIDER_FALLBACK_CHAIN`` (read at call
        time so tests and the runtime can change it on the fly). Providers that
        need an API key but have none configured are skipped entirely — wasting
        a doomed HTTP round-trip is pointless when the sibling may already work.
        """
        chain: list[tuple[str, str]] = [(self.provider, self.model)]
        seen = {self.provider}
        overrides = provider_fallback_models()
        for provider in provider_fallback_chain():
            if provider in seen:
                continue
            seen.add(provider)
            if not self._has_credentials(provider):
                continue
            model = overrides.get(provider) or provider_default_model(provider) or self.model
            chain.append((provider, model))
        return chain

    @classmethod
    def _has_credentials(cls, provider: str) -> bool:
        """False when the provider needs an API key but none is configured."""
        if provider in _LOCAL_PROVIDERS:
            return True
        attr = _PROVIDER_API_KEY_ATTR.get(provider)
        return bool(attr and globals().get(attr))

    @staticmethod
    def _is_transient_api_error(exc: RuntimeError) -> bool:
        """API-level failures that should roll over to the next provider.

        The project convention is that auth errors (401/403) and malformed
        requests are NOT transient — a bad key on one provider will not be fixed
        by another provider, so they fail fast. Rate limits, server hiccups and
        missing-key/provider-configuration errors are genuinely worth a sibling
        attempt.
        """
        msg = str(exc)
        if "is not configured" in msg or "missing API key" in msg:
            return True
        match = re.search(r"\[(\d{3})\]", msg)
        if not match:
            return False
        status = int(match.group(1))
        return status in (408, 429) or 500 <= status <= 599

    async def chat_completion(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        temperature: float = 0.6,
        max_tokens: int = 4096
    ) -> LLMResponse:
        """Run chat over the primary provider, falling back down the chain.

        Transient failures (network errors, timeouts, HTTP 408/429/5xx) and an
        unconfigured primary roll over to the next provider from
        ``TITAN_PROVIDER_FALLBACK_CHAIN``; auth (401/403) and malformed-request
        errors fail fast. The client's configured provider/model are restored
        afterwards so a fallback never silently reconfigures the session.
        """
        chain = self._build_fallback_chain()
        original = (self.provider, self.model, self.base_url, self.api_key)
        errors: list[str] = []
        try:
            for provider, model in chain:
                if provider != self.provider or model != self.model:
                    self.set_model(provider, model)
                try:
                    return await self._chat_completion_once(
                        messages, tools=tools, temperature=temperature, max_tokens=max_tokens
                    )
                except (aiohttp.ClientError, OSError, asyncio.TimeoutError, RuntimeError) as exc:
                    if isinstance(exc, RuntimeError) and not self._is_transient_api_error(exc):
                        raise  # auth / malformed request — not eligible for fallback
                    errors.append(f"{provider} ({model}): {exc}")
                    log.warning("LLM provider '%s' (%s) failed, trying next: %s", provider, model, exc)
                    continue
            detail = " | ".join(errors) if errors else "all providers exhausted"
            raise RuntimeError(f"All providers failed. Errors: {detail}")
        finally:
            # Restore the configured provider/model/credentials so a fallback
            # mid-flight does not permanently swap the client's configuration.
            self.provider, self.model = original[0], original[1]
            self.base_url, self.api_key = original[2], original[3]

    async def _chat_completion_once(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        temperature: float = 0.6,
        max_tokens: int = 4096
    ) -> LLMResponse:
        if self.provider == "puter":
            raise RuntimeError(
                "Puter.js runs only inside the browser (Web UI). "
                "For server-side usage, choose Completions (free), OpenRouter, DeepSeek, Groq, OpenAI or Ollama."
            )
            
        # --- GPT4Free Integration ---
        if self.provider == "g4f":
            import g4f.client
            try:
                g4f_client = g4f.client.Client()
                response = await asyncio.wait_for(
                    asyncio.to_thread(
                        g4f_client.chat.completions.create,
                        model=self.model or "gpt-4o",
                        messages=messages,
                    ),
                    timeout=25.0
                )
                choice = response.choices[0] if hasattr(response, "choices") and response.choices else None
                content = choice.message.content if choice and hasattr(choice, "message") and choice.message else ""
                content = content or ""
                cleaned_content, thoughts, parsed_tools = self._extract_thoughts_and_tools(str(content))
                return LLMResponse(content=cleaned_content, tool_calls=parsed_tools, thoughts=thoughts)
            except asyncio.TimeoutError:
                raise RuntimeError("G4F free endpoint timed out after 25s. Switch to browser Puter.js UI or provide OpenRouter/Ollama.")
            except Exception as e:
                raise RuntimeError(f"G4F API Error: {e}")

        # Laya models return typed choices/scores/probabilities; they cannot
        # implement this autoregressive chat-completion API or generate code.
        if self.provider in {"laya", "laya-mlx"}:
            raise RuntimeError(
                "Laya is a local typed-decision engine, not a chat/text-generation provider. "
                "Use the laya_decide tool for choice/score/yes-no inference, or configure a "
                "chat model (API provider, Ollama, or LM Studio) for conversation and coding."
            )

        # --- Python-tGPT Integration ---
        if self.provider == "tgpt":
            import pytgpt.auto
            try:
                # Initialize the AUTO provider (which picks the fastest/best free provider)
                tgpt_bot = pytgpt.auto.AsyncAUTO(is_conversation=False)
                
                # Format messages into a single prompt string
                prompt_text = ""
                for m in messages:
                    prompt_text += f"{m['role'].upper()}: {m['content']}\n\n"
                
                response_text = ""
                async_ask = await tgpt_bot.chat(prompt_text, stream=True)
                async for text_chunk in async_ask:
                    response_text += text_chunk
                
                cleaned_content, thoughts, parsed_tools = self._extract_thoughts_and_tools(response_text)
                return LLMResponse(content=cleaned_content, tool_calls=parsed_tools, thoughts=thoughts)
            except Exception as e:
                raise RuntimeError(f"TGPT API Error: {e}")

        if not self.base_url:
            raise RuntimeError(f"Provider '{self.provider}' is not configured (missing API key).")

        # Enforce the token rate cap (214k tokens/s): reserve input estimate +
        # max_output before the call; the bucket waits if the budget is exceeded.
        await self.token_limiter.acquire(estimate_tokens(messages, tools, max_tokens))

        url = f"{self.base_url}/chat/completions"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
        if self.provider == "openrouter":
            headers["HTTP-Referer"] = "https://github.com/titan-agent"
            headers["X-Title"] = "Titan Agent"
        if self.provider == "completions":
            # Completions.me sits behind Cloudflare (error 1010 blocks non-browser agents).
            headers["User-Agent"] = (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                "(KHTML, like Gecko) Chrome/126.0 Safari/537.36"
            )
            headers["Accept"] = "application/json, text/plain, */*"

        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens
        }
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = "auto"

        async with aiohttp.ClientSession() as session, session.post(url, headers=headers, json=payload, timeout=aiohttp.ClientTimeout(total=180)) as resp:
                if resp.status != 200:
                    err_body = await resp.text()
                    hint = ""
                    if resp.status in (401, 403):
                        hint = (
                            " (Hint: the API key is invalid or not set. "
                            "Choose Puter.js (no key), Ollama, or Completions (free) from the Web UI settings, "
                            "or add the correct API key to the .env file)"
                        )
                    raise RuntimeError(
                        f"LLM API Error [{resp.status}] from {self.provider} ({self.model}): {err_body}{hint}"
                    )

                data = await resp.json()
                choice = data["choices"][0]
                message = choice.get("message", {})
                content = message.get("content") or ""
                native_tool_calls = message.get("tool_calls") or []

                cleaned_content, thoughts, parsed_tools = self._extract_thoughts_and_tools(content)

                all_tool_calls = list(native_tool_calls)
                all_tool_calls.extend(parsed_tools)

                return LLMResponse(
                    content=cleaned_content,
                    tool_calls=all_tool_calls,
                    thoughts=thoughts
                )
