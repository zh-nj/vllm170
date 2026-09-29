# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU-only OpenAI-shaped fixture; never used to measure model performance."""

import argparse
import json
import os
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument(
        "--mode", choices=["good", "fail", "bad", "crash"], default="good"
    )
    args = parser.parse_args()
    if args.mode == "fail":
        raise SystemExit(17)

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            self.reply({"data": [{"id": "cpu-fixture"}]})

        def do_POST(self):
            self.rfile.read(int(self.headers.get("Content-Length", 0)))
            self.reply(
                {
                    "choices": [{"text": "ok"}],
                    "usage": {
                        "completion_tokens": 0 if args.mode == "bad" else 1,
                    },
                }
            )

        def reply(self, data):
            body = json.dumps(data).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args):
            pass  # Health polling is deliberately quiet in the CPU fixture.

    if args.mode == "crash":
        threading.Timer(3, lambda: os._exit(19)).start()
    ThreadingHTTPServer(("127.0.0.1", args.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
