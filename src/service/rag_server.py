"""
rag_server.py
------------------------------------------------------------
Wraps the Phase 1-3 RAG pipeline (chunking, retrieval, generation)
and Phase 2's evaluation checks behind an HTTP API, so the Express
app (app/) can call it like any backend service instead of needing
to reimplement RAG logic in JavaScript.

BUILT ON PYTHON'S STDLIB `http.server`, NOT FastAPI/Flask — this
sandbox has no internet access to install either, and stdlib-only
means this file actually runs, right now, with zero setup, instead
of being "correct but untested" like Phase 3's GPU scripts. For a
real deployment you'd likely swap this for FastAPI (better request
validation, async support, auto-generated docs) — that's a
mechanical swap of the routing layer, not a rewrite of the RAG
logic underneath, which is exactly why the logic lives in Phase
1-3's modules and this file is a thin HTTP wrapper around them.

Endpoints:
  GET  /health
  POST /ask   { "question": "..." }
       -> { "answer", "sources", "latencyMs", "numericAccuracy",
            "ungroundedNumbers", "isRefusal", "sourceOverlap" }
------------------------------------------------------------
"""

import json
import os
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from ..cli import load_sample_sections, build_llm
from ..pipeline.chunker import chunk_sections
from ..rag.retriever import Retriever
from ..rag.qa_chain import RAGChain
from ..eval.numeric_accuracy import check_numeric_accuracy
from ..eval.hallucination_checker import check_hallucination_signals

# Render/Fly/Heroku-style hosts inject PORT at runtime — reading it
# here (falling back to 8010 for local dev) is what makes this
# deployable as-is instead of needing host-specific edits.
PORT = int(os.environ.get("PORT", 8010))

# Built once at server startup, reused across requests — rebuilding
# the retriever per-request would re-fit the TF-IDF vectorizer every
# time, which is wasted work for a corpus that isn't changing.
_sections = load_sample_sections()
_chunks = chunk_sections(
    _sections, company="AAPL", filing_type="10-K", source_id="sample-fy2025",
    chunk_size_words=120, overlap_words=30,
)
_retriever = Retriever.build(_chunks)


def _build_chain(llm_backend: str) -> RAGChain:
    return RAGChain(_retriever, build_llm(llm_backend), k=3)


class RAGRequestHandler(BaseHTTPRequestHandler):
    def _send_json(self, status: int, payload: dict):
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == "/health":
            self._send_json(200, {"status": "ok", "chunksIndexed": len(_chunks)})
        else:
            self._send_json(404, {"error": "not found"})

    def do_POST(self):
        if self.path != "/ask":
            self._send_json(404, {"error": "not found"})
            return

        content_length = int(self.headers.get("Content-Length", 0))
        raw_body = self.rfile.read(content_length)

        try:
            body = json.loads(raw_body)
            question = body["question"]
            llm_backend = body.get("llmBackend", "mock")
        except (json.JSONDecodeError, KeyError):
            self._send_json(400, {"error": "expected JSON body: {\"question\": \"...\"}"})
            return

        start = time.perf_counter()
        chain = _build_chain(llm_backend)
        result = chain.ask(question)
        latency_ms = (time.perf_counter() - start) * 1000

        source_context = "\n".join(s.chunk.text for s in result.sources)
        numeric = check_numeric_accuracy(result.answer, source_context)
        hallucination = check_hallucination_signals(result.answer, source_context)

        self._send_json(200, {
            "question": question,
            "answer": result.answer,
            "sources": [
                {"citation": s.chunk.citation, "score": s.score, "text": s.chunk.text[:200]}
                for s in result.sources
            ],
            "latencyMs": round(latency_ms, 1),
            "numericAccuracy": numeric.accuracy,
            "ungroundedNumbers": numeric.ungrounded_numbers,
            "isRefusal": hallucination.is_refusal,
            "sourceOverlap": round(hallucination.source_overlap, 3),
            "flaggedLowOverlap": hallucination.flagged_low_overlap,
        })

    def log_message(self, format, *args):
        print(f"[rag_server] {self.address_string()} - {format % args}")


def main():
    server = ThreadingHTTPServer(("0.0.0.0", PORT), RAGRequestHandler)
    print(f"RAG service listening on http://localhost:{PORT}")
    print(f"  GET  /health")
    print(f"  POST /ask  {{\"question\": \"...\", \"llmBackend\": \"mock|anthropic|openai\"}}")
    print(f"Indexed {len(_chunks)} chunks from {len(_sections)} sections")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down.")
        server.shutdown()


if __name__ == "__main__":
    main()
