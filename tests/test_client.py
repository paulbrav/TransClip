import json
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from granite_speach.client import InferenceClient
from granite_speach.settings import Settings


class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        self._json({"status": "ready"})

    def do_POST(self):
        length = int(self.headers["content-length"])
        payload = json.loads(self.rfile.read(length).decode("utf-8"))
        self._json({"text": "ok", "payload": payload, "path": self.path})

    def log_message(self, format, *args):
        return

    def _json(self, payload):
        data = json.dumps(payload).encode("utf-8")
        self.send_response(200)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


class ClientTests(unittest.TestCase):
    def test_client_calls_service(self):
        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            host, port = server.server_address
            client = InferenceClient(Settings(host=host, port=port))
            self.assertEqual(client.health()["status"], "ready")
            response = client.transcribe(__file__, cleanup=False)
            self.assertEqual(response["text"], "ok")
            self.assertFalse(response["payload"]["cleanup"])
            cleaned = client.cleanup_transcribe(__file__)
            self.assertEqual(cleaned["path"], "/cleanup/transcribe")
            toggled = client.record_toggle(cleanup=True)
            self.assertTrue(toggled["payload"]["cleanup"])
            stopped = client.record_stop(discard=True)
            self.assertTrue(stopped["payload"]["discard"])
            started = client.record_start()
            self.assertEqual(started["payload"], {})
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=2)


if __name__ == "__main__":
    unittest.main()
