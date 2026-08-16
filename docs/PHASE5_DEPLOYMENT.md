# Phase 5 — Deployment Guide (Render, free tier)

## Before you start: push this to GitHub

Render deploys from a GitHub repo, not a zip upload. If you haven't already:

```bash
cd finllm-rag
git init
git add .
git commit -m "FinLLM-RAG: Phases 1-4 complete"
```

Create a new repo on GitHub, then:
```bash
git remote add origin https://github.com/<your-username>/finllm-rag.git
git push -u origin main
```

## Deploy both services

1. Go to [render.com](https://render.com), sign up/log in (free — no card needed for the free tier).
2. Click **New +** → **Blueprint**.
3. Connect your GitHub account and select the `finllm-rag` repo.
4. Render reads `render.yaml` from the repo root automatically and shows both services
   (`finllm-rag-service` and `finllm-rag-console`) ready to deploy. Click **Apply**.
5. Wait for both builds to finish (a few minutes — the Python service installs numpy/scikit-learn,
   the Node service runs `npm install`).

## Wire the two services together (one manual step)

Render assigns each service's public URL only after its first deploy, so this can't be
pre-filled automatically:

1. Once `finllm-rag-service` finishes deploying, copy its URL from the Render dashboard
   (looks like `https://finllm-rag-service-xxxx.onrender.com`).
2. Go to `finllm-rag-console`'s **Environment** tab, set `RAG_SERVICE_URL` to that URL
   (no trailing slash).
3. Click **Manual Deploy** → **Deploy latest commit** on `finllm-rag-console` to pick up the change.

## Verify it

Open `finllm-rag-console`'s URL + `/dashboard.html`. Ask a question. If you see a real answer
(or MockClient's placeholder text — that's expected without an API key set) with sources and
stats updating, both services are correctly talking to each other in production.

## Honest free-tier caveats

- **Free services spin down after 15 minutes of no traffic**, and take ~30-50 seconds to wake
  back up on the next request. If you're demoing this live in an interview, send it a request
  a minute or two before you need it, or mention this trade-off out loud — it's a completely
  normal thing to explain, and shows you understand production infra trade-offs, not a flaw
  to hide.
- **The demo runs with MockClient by default** (see `requirements-deploy.txt`) — it proves the
  retrieval + eval pipeline works, but doesn't generate real natural-language answers. To get
  real answers: uncomment `anthropic` in `requirements-deploy.txt`, set `ANTHROPIC_API_KEY` in
  `finllm-rag-service`'s environment variables on Render, and change the dashboard's request
  (or `ask.js`'s default) to send `"llmBackend": "anthropic"`. This costs a small amount per
  request — Render's free tier doesn't cover LLM API usage, that's billed separately by Anthropic.

## Alternative: Fly.io

Render was chosen because its `render.yaml` blueprint deploys both services from one repo with
no CLI setup — good for a first deployment. Fly.io is a reasonable alternative if you want more
control (persistent volumes, no cold-start spin-down on some plans), but needs the `flyctl` CLI
and a `fly.toml` per service instead of one shared blueprint — a reasonable next step once you're
comfortable with Render's version and want to compare.
