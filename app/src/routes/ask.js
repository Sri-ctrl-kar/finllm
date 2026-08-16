const express = require('express');
const { logRequest, estimateCostUsd } = require('../metricsStore');

const RAG_SERVICE_URL = process.env.RAG_SERVICE_URL || 'http://localhost:8010';

function createAskRoute() {
  const router = express.Router();

  router.post('/', async (req, res) => {
    try {
      const { question, llmBackend } = req.body || {};
      if (!question) {
        return res.status(400).json({ error: 'question is required' });
      }

      const ragResponse = await fetch(`${RAG_SERVICE_URL}/ask`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ question, llmBackend: llmBackend || 'mock' }),
      });

      if (!ragResponse.ok) {
        const errBody = await ragResponse.json().catch(() => ({}));
        return res.status(502).json({ error: 'RAG service error', detail: errBody });
      }

      const result = await ragResponse.json();

      const promptChars = result.sources.reduce((sum, s) => sum + s.text.length, 0) + question.length;
      const estimatedCostUsd = estimateCostUsd(promptChars, result.answer.length);

      const logged = logRequest({
        question,
        answer: result.answer,
        latencyMs: result.latencyMs,
        numericAccuracy: result.numericAccuracy,
        ungroundedNumbers: result.ungroundedNumbers,
        isRefusal: result.isRefusal,
        sourceOverlap: result.sourceOverlap,
        flaggedLowOverlap: result.flaggedLowOverlap,
        estimatedCostUsd,
      });

      res.json({ ...result, estimatedCostUsd, loggedAt: logged.timestamp });
    } catch (err) {
      console.error(err);
      res.status(502).json({
        error: 'Could not reach the RAG service. Is it running?',
        hint: `Start it with: python -m src.service.rag_server  (from the finllm-rag/ root, not app/)`,
      });
    }
  });

  return router;
}

module.exports = { createAskRoute };
