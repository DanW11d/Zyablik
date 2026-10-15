"""Клиент языковой модели для проекта Зяблик.

Работает с любым OpenAI-совместимым API: OpenRouter (облако) или Ollama (локально).
Смена провайдера и модели делается только через .env, код менять не нужно.
"""
import os
import time
from typing import Iterator

from dotenv import load_dotenv
from openai import APIConnectionError, APIStatusError, OpenAI, RateLimitError

load_dotenv()

PROVIDERS = {
    "openrouter": {"base_url": "https://openrouter.ai/api/v1", "key_env": "OPENROUTER_API_KEY"},
    "ollama": {"base_url": "http://localhost:11434/v1", "key_env": None},
}

# Укороченный системный промпт по приложению Ж ТЗ. Полный текст подставим позже.
SYSTEM_PROMPT_BE = (
    "Ты размаўляеш з карыстальнікам па-беларуску, нарматыўнай літаратурнай мовай. "
    "Не ўжывай калькі з рускай мовы. Калі карыстальнік піша па-руску, адказвай па-беларуску. "
    "Не выдумляй фактаў: калі не ведаеш адказу, так і скажы. "
    "Адказвай каротка, два-пяць сказаў, без markdown."
)


def _config():
    provider = os.getenv("LLM_PROVIDER", "openrouter").lower()
    if provider not in PROVIDERS:
        raise SystemExit(f"Неизвестный LLM_PROVIDER: {provider}. Допустимо: {', '.join(PROVIDERS)}")
    spec = PROVIDERS[provider]
    base_url = os.getenv("LLM_BASE_URL") or spec["base_url"]
    api_key = os.getenv(spec["key_env"], "") if spec["key_env"] else "ollama"
    if spec["key_env"] and not api_key:
        raise SystemExit(f"Не задан {spec['key_env']} в .env")
    model = os.getenv("LLM_MODEL", "")
    if not model:
        raise SystemExit("Не задан LLM_MODEL в .env (список бесплатных: python test_llm.py --list-free)")
    fallbacks = [m.strip() for m in os.getenv("LLM_FALLBACK_MODELS", "").split(",") if m.strip()]
    return provider, base_url, api_key, [model] + fallbacks


def get_client() -> tuple[OpenAI, list[str]]:
    _, base_url, api_key, models = _config()
    return OpenAI(base_url=base_url, api_key=api_key, timeout=60), models


def chat(messages: list[dict], system: str = SYSTEM_PROMPT_BE, max_tokens: int = 600,
         temperature: float = 0.4) -> str:
    """Один запрос без потока. При лимите (429) пробует запасные модели."""
    client, models = get_client()
    full = [{"role": "system", "content": system}] + messages
    last_err = None
    for model in models:
        for attempt in range(2):
            try:
                r = client.chat.completions.create(
                    model=model, messages=full, max_tokens=max_tokens, temperature=temperature)
                return r.choices[0].message.content or ""
            except RateLimitError as e:
                last_err = e
                time.sleep(1 + 2 * attempt)
            except (APIConnectionError, APIStatusError) as e:
                last_err = e
                break  # эта модель не отвечает, пробуем следующую
    raise RuntimeError(f"Все модели недоступны: {last_err}")


def chat_stream(messages: list[dict], system: str = SYSTEM_PROMPT_BE, max_tokens: int = 600,
                temperature: float = 0.4) -> Iterator[str]:
    """Потоковая выдача по кускам текста (для SSE на сервере)."""
    client, models = get_client()
    full = [{"role": "system", "content": system}] + messages
    stream = client.chat.completions.create(
        model=models[0], messages=full, max_tokens=max_tokens,
        temperature=temperature, stream=True)
    for chunk in stream:
        if chunk.choices and chunk.choices[0].delta.content:
            yield chunk.choices[0].delta.content


def list_free_models() -> list[str]:
    """Бесплатные модели OpenRouter (цена за вход и выход равна нулю)."""
    import json
    import urllib.request
    with urllib.request.urlopen("https://openrouter.ai/api/v1/models", timeout=30) as r:
        data = json.load(r)["data"]
    free = []
    for m in data:
        p = m.get("pricing", {})
        if str(p.get("prompt")) in ("0", "0.0") and str(p.get("completion")) in ("0", "0.0"):
            free.append(m["id"])
    return sorted(free)
