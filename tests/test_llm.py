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
            self.assertEqual(llm.detect_backend(), (self.url + "/v1", "qwen3:8b"))

    def test_detect_with_no_models_pulled(self):
        Handler.tags = {"models": []}
        try:
            with mock.patch.object(llm, "OLLAMA_URL", self.url):
                self.assertEqual(llm.detect_backend(), (self.url + "/v1", None))
        finally:
            Handler.tags = {"models": [{"name": "llama3.2:1b"}, {"name": "qwen3:8b"}]}

    def test_detect_nothing_running(self):
        with mock.patch.object(llm, "OLLAMA_URL", "http://127.0.0.1:9"), \
                mock.patch.object(llm, "LLAMA_SERVER_URL", "http://127.0.0.1:9"):
            self.assertIsNone(llm.detect_backend(timeout=1))

    def test_detect_llama_server_fallback(self):
        with mock.patch.object(llm, "OLLAMA_URL", "http://127.0.0.1:9"), \
                mock.patch.object(llm, "LLAMA_SERVER_URL", self.url):
            self.assertEqual(llm.detect_backend(), (self.url + "/v1", "default"))


if __name__ == "__main__":
    unittest.main()
