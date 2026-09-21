"""LLM client against a local fake of the OpenAI-compatible / Ollama endpoints."""
import json
import threading
import unittest
from http.server import BaseHTTPRequestHandler, HTTPServer
from unittest import mock

from signallm import llm


class Handler(BaseHTTPRequestHandler):
    requests = []
    tags = {"models": [{"name": "llama3.2:1b"}, {"name": "qwen3:8b"}]}
    status = 200
    pull_events = []

    def log_message(self, *a):
        pass

    def _send(self, obj, status=200):
        body = json.dumps(obj).encode()
        self.send_response(status)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        self._send(self.tags if self.path == "/api/tags" else {"status": "ok"})

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        Handler.requests.append((self.path, dict(self.headers), body))
        if Handler.status != 200:
            return self._send({"error": {"message": "model not found"}}, Handler.status)
        if self.path == "/api/chat":
            return self._send({"message": {"role": "assistant", "content": "", "tool_calls": [
                {"function": {"name": "describe", "arguments": {}}}]}})
        if self.path == "/api/pull":
            self.send_response(200)
            self.end_headers()
            for ev in Handler.pull_events:
                self.wfile.write((json.dumps(ev) + "\n").encode())
            return
        self._send({"choices": [{"message": {"role": "assistant", "content": "hi", "tool_calls": [
            {"id": "x", "type": "function", "function": {"name": "describe", "arguments": "{}"}}]}}]})


class LLM(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.srv = HTTPServer(("127.0.0.1", 0), Handler)
        cls.url = f"http://127.0.0.1:{cls.srv.server_port}"
        threading.Thread(target=cls.srv.serve_forever, daemon=True).start()

    @classmethod
    def tearDownClass(cls):
        cls.srv.shutdown()
        cls.srv.server_close()

    def setUp(self):
        Handler.requests.clear()
        Handler.status = 200

    def test_chat_sends_tools_and_returns_message(self):
        c = llm.LLMClient(self.url + "/v1/", "m", api_key="k", temperature=0.1)
        msg = c.chat([{"role": "user", "content": "q"}], [{"type": "function"}])
        self.assertEqual(msg["tool_calls"][0]["function"]["name"], "describe")
        path, headers, body = Handler.requests[0]
        self.assertEqual(path, "/v1/chat/completions")
        self.assertEqual(headers["Authorization"], "Bearer k")
        self.assertEqual((body["model"], body["stream"], body["temperature"]), ("m", False, 0.1))
        self.assertEqual(body["tools"], [{"type": "function"}])

    def test_http_error_becomes_llm_error_with_detail(self):
        Handler.status = 404
        with self.assertRaisesRegex(llm.LLMError, "404.*model not found"):
            llm.LLMClient(self.url + "/v1", "nope").chat([], [])

    def test_unreachable_backend(self):
        with self.assertRaisesRegex(llm.LLMError, "cannot reach"):
            llm.LLMClient("http://127.0.0.1:9/v1", "m", timeout=2).chat([], [])

    def test_detect_prefers_tool_capable_model(self):
        with mock.patch.object(llm, "OLLAMA_URL", self.url):
            self.assertEqual(llm.detect_backend(), ("ollama", self.url, "qwen3:8b"))

    def test_detect_with_no_models_pulled(self):
        Handler.tags = {"models": []}
        try:
            with mock.patch.object(llm, "OLLAMA_URL", self.url):
                self.assertEqual(llm.detect_backend(), ("ollama", self.url, None))
        finally:
            Handler.tags = {"models": [{"name": "llama3.2:1b"}, {"name": "qwen3:8b"}]}

    def test_detect_nothing_running(self):
        with mock.patch.object(llm, "OLLAMA_URL", "http://127.0.0.1:9"), \
                mock.patch.object(llm, "LLAMA_SERVER_URL", "http://127.0.0.1:9"):
            self.assertIsNone(llm.detect_backend(timeout=1))

    def test_detect_llama_server_fallback(self):
        with mock.patch.object(llm, "OLLAMA_URL", "http://127.0.0.1:9"), \
                mock.patch.object(llm, "LLAMA_SERVER_URL", self.url):
            self.assertEqual(llm.detect_backend(), ("openai", self.url + "/v1", "default"))

    def test_ollama_client_sets_context_think_and_keepalive_per_request(self):
        msg = llm.OllamaClient(self.url, "m").chat([{"role": "user", "content": "q"}], [{"type": "function"}])
        self.assertEqual(msg["tool_calls"][0]["function"]["arguments"], {})
        path, _, body = Handler.requests[0]
        self.assertEqual(path, "/api/chat")
        self.assertIs(body["think"], False)
        self.assertEqual(body["options"]["num_ctx"], llm.NUM_CTX)
        self.assertGreaterEqual(llm.NUM_CTX, 8192)
        self.assertEqual(body["keep_alive"], llm.KEEP_ALIVE)
        self.assertIs(body["stream"], False)

    def test_ollama_client_think_can_be_enabled(self):
        llm.OllamaClient(self.url, "m", think=True).chat([], [])
        self.assertIs(Handler.requests[0][2]["think"], True)

    def test_openai_client_disables_reasoning_by_default(self):
        llm.LLMClient(self.url + "/v1", "m").chat([], [])
        self.assertEqual(Handler.requests[0][2]["reasoning_effort"], "none")
        Handler.requests.clear()
        llm.LLMClient(self.url + "/v1", "m", think=True).chat([], [])
        self.assertNotIn("reasoning_effort", Handler.requests[0][2])

    def test_ollama_models_and_pull_progress(self):
        self.assertEqual(llm.ollama_models(self.url), ["llama3.2:1b", "qwen3:8b"])
        Handler.pull_events = [{"status": "pulling manifest"},
                               {"status": "pulling abc", "total": 100, "completed": 50},
                               {"status": "success"}]
        seen = []
        llm.pull_model(self.url, "qwen3:4b", lambda *a: seen.append(a))
        self.assertEqual(seen[1], ("pulling abc", 50, 100))
        self.assertEqual(seen[-1][0], "success")
        self.assertEqual(Handler.requests[0][2]["model"], "qwen3:4b")

    def test_pull_error_event_is_reported(self):
        Handler.pull_events = [{"error": "pull model manifest: file does not exist"}]
        with self.assertRaisesRegex(llm.LLMError, "could not download nope.*does not exist"):
            llm.pull_model(self.url, "nope")

    def test_pick_model_prefers_non_reasoning_build(self):
        # plain qwen3:4b is a thinking-only model now; the instruct build answers in seconds
        self.assertEqual(llm.pick_model(["qwen3:4b", "qwen3:4b-instruct", "llama3.2:1b"]), "qwen3:4b-instruct")
        self.assertEqual(llm.pick_model(["llama3.2:1b", "mistral:7b"]), "llama3.2:1b")
        self.assertEqual(llm.pick_model(["gemma3:4b"]), "gemma3:4b")
        self.assertIsNone(llm.pick_model([]))
        self.assertIn("instruct", llm.DEFAULT_MODEL)

    def test_make_client_picks_protocol(self):
        self.assertIsInstance(llm.make_client("ollama", self.url, "m"), llm.OllamaClient)
        self.assertIsInstance(llm.make_client("openai", self.url, "m"), llm.LLMClient)


if __name__ == "__main__":
    unittest.main()
