"""Chat clients with tool calling. Standard library only.

Two protocols, one interface (chat(messages, tools) -> assistant message):
- OllamaClient talks to Ollama's native /api/chat. Context size, thinking and keep-alive are
  passed per request, so the user never has to configure the Ollama server.
- LLMClient talks to any OpenAI-compatible endpoint (llama.cpp's llama-server, LM Studio, ...).
"""
import json
import urllib.error
import urllib.request

OLLAMA_URL = "http://localhost:11434"
LLAMA_SERVER_URL = "http://localhost:8080"
# tried in order when picking a default Ollama model; all handle tool calling reasonably
DEFAULT_MODEL = "qwen3:4b-instruct"  # ~2.5 GB, no reasoning; plain qwen3:4b is now a thinking-only model
NUM_CTX = 8192               # tool schemas + scan results do not fit in Ollama's 4096 default
KEEP_ALIVE = "30m"           # keep the model loaded between questions
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


def detect_backend(timeout: float = 1.5) -> tuple[str, str, str | None] | None:
    """Returns (kind, base_url, default_model) for the first backend that answers, else None.
    kind is "ollama" (native API, base_url without /v1) or "openai"."""
    try:
        names = ollama_models(OLLAMA_URL, timeout)
        pick = pick_model(names)
        return "ollama", OLLAMA_URL, pick
    except LLMError:
        pass
    try:
        _http("GET", LLAMA_SERVER_URL + "/health", timeout=timeout)
        return "openai", LLAMA_SERVER_URL + "/v1", "default"  # llama-server serves one model
    except LLMError:
        return None


def pick_model(names: list[str]) -> str | None:
    """Best installed model for tool calling on a CPU: prefer non-reasoning ('instruct') builds."""
    for prefix in PREFERRED_MODELS:
        matches = [n for n in names if n.startswith(prefix)]
        if matches:
            return next((n for n in matches if "instruct" in n), matches[0])
    return names[0] if names else None


def ollama_models(base_url: str, timeout: float = 5) -> list[str]:
    return [m["name"] for m in _http("GET", base_url.rstrip("/") + "/api/tags", timeout=timeout).get("models", [])]


def pull_model(base_url: str, name: str, on_progress=lambda status, done, total: None):
    """Download a model through Ollama, reporting progress. Raises LLMError on failure."""
    req = urllib.request.Request(base_url.rstrip("/") + "/api/pull", method="POST",
                                 headers={"Content-Type": "application/json"},
                                 data=json.dumps({"model": name, "stream": True}).encode())
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            for line in r:
                ev = json.loads(line)
                if "error" in ev:
                    raise LLMError(f"could not download {name}: {ev['error']}")
                on_progress(ev.get("status", ""), ev.get("completed", 0), ev.get("total", 0))
    except urllib.error.HTTPError as e:
        e.close()
        raise LLMError(f"could not download {name}: HTTP {e.code}") from None
    except (urllib.error.URLError, TimeoutError, ConnectionError) as e:
        raise LLMError(f"could not download {name}: {getattr(e, 'reason', e)}") from None


class OllamaClient:
    def __init__(self, base_url: str, model: str, temperature: float = 0.2, timeout: float = 600,
                 think: bool = False, num_ctx: int = NUM_CTX):
        self.base_url = base_url.rstrip("/")
        self.model, self.temperature, self.timeout = model, temperature, timeout
        self.think, self.num_ctx = think, num_ctx  # reasoning is minutes per answer on a CPU: off
        self.last_stats: dict = {}  # timings and token counts of the last call, for debugging

    def chat(self, messages: list[dict], tools: list[dict]) -> dict:
        r = _http("POST", self.base_url + "/api/chat", timeout=self.timeout,
                  body={"model": self.model, "messages": messages, "tools": tools, "stream": False,
                        "think": self.think, "keep_alive": KEEP_ALIVE,
                        "options": {"num_ctx": self.num_ctx, "temperature": self.temperature}})
        ns = 1e9
        self.last_stats = {"prompt_tokens": r.get("prompt_eval_count", 0),
                           "prompt_s": r.get("prompt_eval_duration", 0) / ns,
                           "gen_tokens": r.get("eval_count", 0), "gen_s": r.get("eval_duration", 0) / ns,
                           "load_s": r.get("load_duration", 0) / ns}
        try:
            return r["message"]
        except (KeyError, TypeError):
            raise LLMError(f"unexpected response from Ollama: {str(r)[:200]}") from None


class LLMClient:
    def __init__(self, base_url: str, model: str, api_key: str | None = None,
                 temperature: float = 0.2, timeout: float = 600, think: bool = False):
        self.base_url = base_url.rstrip("/")
        self.model, self.api_key = model, api_key
        self.temperature, self.timeout = temperature, timeout
        self.think = think
        self.last_stats: dict = {}

    def chat(self, messages: list[dict], tools: list[dict]) -> dict:
        """One completion. Returns the assistant message dict (content and/or tool_calls)."""
        body = {"model": self.model, "messages": messages, "tools": tools,
                "temperature": self.temperature, "stream": False}
        if not self.think:
            body["reasoning_effort"] = "none"
        r = _http("POST", self.base_url + "/chat/completions", timeout=self.timeout,
                  api_key=self.api_key, body=body)
        u = r.get("usage") or {}
        self.last_stats = {"prompt_tokens": u.get("prompt_tokens", 0), "gen_tokens": u.get("completion_tokens", 0)}
        try:
            return r["choices"][0]["message"]
        except (KeyError, IndexError, TypeError):
            raise LLMError(f"unexpected response from LLM backend: {str(r)[:200]}") from None


def make_client(kind: str, base_url: str, model: str, api_key: str | None = None, think: bool = False):
    if kind == "ollama":
        return OllamaClient(base_url, model, think=think)
    return LLMClient(base_url, model, api_key, think=think)
