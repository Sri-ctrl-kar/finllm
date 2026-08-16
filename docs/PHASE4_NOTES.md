# Phase 4 Notes — Full-Stack App

## Two services, one stack

```
Browser (dashboard.html)
      │
      ▼
Express app (app/, port 3002)  ──▶  metricsStore.js (data/metrics.jsonl)
      │
      ▼  HTTP
Python RAG service (src/service/rag_server.py, port 8010)
      │
      ▼
Phase 1-3's chunker/retriever/qa_chain/eval modules
```

Two languages, on purpose: the RAG core (chunking, retrieval, eval
checks) is Python because that's the natural language for the ML work
in Phases 1-3. The app layer is Express/Node — reusing the skill set
from RideSync/TradeSync — because a monitoring dashboard and API layer
is a web-backend problem, not an ML problem. This is also a realistic
shape for a real system: an ML service behind an internal API, fronted
by an application layer that handles logging, cost tracking, and the
user-facing surface.

## What's proven working vs. what needs `npm install`

| Piece | Status |
|---|---|
| `src/service/rag_server.py` | **Real, tested live** — started it, hit `/health` and `/ask` with real HTTP requests, got correct answers back |
| `app/src/metricsStore.js` | **Real, tested directly** — logged synthetic requests, verified flag-rate/accuracy-averaging/cost-estimate math with real assertions |
| The `ask.js` → RAG service → `metricsStore` integration | **Proven correct** — exercised the exact same fetch+log logic that `ask.js` uses, directly, against the live Python service, end-to-end |
| Express itself (`app/src/server.js`, routing) | Needs `npm install` — no internet in this sandbox, same limitation as RideSync/TradeSync. Code is complete and syntax-checked, not live-tested through Express's own router |

This is a meaningfully stronger position than "trust me, the code is
right" — every piece of actual logic (the Python service, the metrics
math, the integration flow) was run for real. The only untested part is
Express's routing itself, which is a thin, well-worn layer doing
`req.body` → call a function → `res.json()` — the least likely place
for a bug to hide, and exactly the same situation as the last two
projects.

## Running it for real

```bash
# Terminal 1 — from finllm-rag/ (the Python root, not app/)
python -m src.service.rag_server

# Terminal 2 — from finllm-rag/app/
npm install
npm start

# Open http://localhost:3002/dashboard.html
```

Ask a few questions through the dashboard, then watch the stats update —
hallucination flag rate, p95 latency, avg numeric accuracy, running cost
estimate. Switch `llmBackend` to `anthropic` in a request body (or edit
`ask.js`'s default) once you have `ANTHROPIC_API_KEY` set, to see real
generated answers instead of MockClient's placeholder text — the mock
answers will always show 0% numeric accuracy and get flagged, which is
correct behavior (they're not real answers), not a bug.

## What Phase 5 adds on top of this

- Deploy both services somewhere real (Fly.io / Render / a VPS) so
  there's a public link, not just localhost
- Swap the JSONL metrics file for something queryable at scale (SQLite
  is the natural next step — same idea as noted for RideSync's
  in-memory state)
- Write the technical report summarizing all four phases with real
  numbers from each
