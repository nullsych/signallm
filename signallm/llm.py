"""Minimal client for OpenAI-compatible chat endpoints with tool calling.

Ollama (http://localhost:11434/v1) and llama.cpp's llama-server (http://localhost:8080/v1) both
speak this protocol, so one client covers both. Standard library only.
"""
import json
import urllib.error
import urllib.request

OLLAMA_URL = "http://localhost:11434"
LLAMA_SERVER_URL = "http://localhost:8080"
# tried in order when picking a default Ollama model; all handle tool calling reasonably
PREFERRED_MODELS = ("qwen3", "qwen2.5", "llama3.1", "llama3.2", "mistral-nemo", "mistral", "granite")


class LLMError(Exception):
    """Backend unreachable or returned something unusable; the message is meant for the user."""


def _http(method: str, url: str, body: dict | None = None, timeout: float = 300, api_key: str | None = None):
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    req = urllib.request.Request(url, method=method, headers=headers,
                                 data=json.dumps(body).encode() if body is not None else None)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read().decode() or "null")
    except urllib.error.HTTPError as e:
        detail = e.read().decode(errors="replace")[:300]
        e.close()
        raise LLMError(f"LLM backend returned HTTP {e.code}: {detail}") from None
    except (urllib.error.URLError, TimeoutError, ConnectionError) as e:
        raise LLMError(f"cannot reach LLM backend at {url}: {getattr(e, 'reason', e)}") from None


def detect_backend(timeout: float = 1.5) -> tuple[str, str | None] | None:
    """Returns (openai_base_url, default_model) for the first backend that answers, else None."""
    try:
        tags = _http("GET", OLLAMA_URL + "/api/tags", timeout=timeout)
        names = [m["name"] for m in tags.get("models", [])]
        pick = next((n for p in PREFERRED_MODELS for n in names if n.startswith(p)), names[0] if names else None)
        return OLLAMA_URL + "/v1", pick
    except LLMError:
        pass
    try:
        _http("GET", LLAMA_SERVER_URL + "/health", timeout=timeout)
        return LLAMA_SERVER_URL + "/v1", "default"  # llama-server serves one model, name is ignored
    except LLMError:
        return None


class LLMClient:
    def __init__(self, base_url: str, model: str, api_key: str | None = None,
                 temperature: float = 0.2, timeout: float = 300):
        self.base_url = base_url.rstrip("/")
        self.model, self.api_key = model, api_key
        self.temperature, self.timeout = temperature, timeout

    def chat(self, messages: list[dict], tools: list[dict]) -> dict:
        """One completion. Returns the assistant message dict (content and/or tool_calls)."""
        r = _http("POST", self.base_url + "/chat/completions", timeout=self.timeout, api_key=self.api_key,
                  body={"model": self.model, "messages": messages, "tools": tools,
                        "temperature": self.temperature, "stream": False})
        try:
            return r["choices"][0]["message"]
        except (KeyError, IndexError, TypeError):
            raise LLMError(f"unexpected response from LLM backend: {str(r)[:200]}") from None
