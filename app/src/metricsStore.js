/**
 * metricsStore.js
 * ------------------------------------------------------------
 * Every call to the RAG service gets logged here: latency,
 * grounding/hallucination signals, and an estimated cost. This is
 * what the dashboard reads from — the actual point of Phase 4 is
 * that hallucination rate and latency aren't one-time benchmark
 * numbers (Phase 2/3), they're something you watch continuously
 * across real usage, the way a production system actually would.
 *
 * Storage: a flat JSON file, append-on-write. That's the right
 * amount of infrastructure for a portfolio project's traffic
 * volume — a real production system would use a time-series DB
 * (see docs/PHASE4_NOTES.md for the upgrade path), but building
 * one here would be complexity that doesn't teach anything extra.
 * ------------------------------------------------------------
 */

const fs = require('fs');
const path = require('path');

const METRICS_FILE = path.join(__dirname, '..', 'data', 'metrics.jsonl');

// Rough cost model: most hosted LLM APIs price per 1K tokens, and
// token count is roughly word_count / 0.75 (English text averages
// ~0.75 words per token). This is a deliberately simple estimate,
// not a real billing reconciliation — good enough to show a COST
// TREND on the dashboard, not to generate an invoice.
const ESTIMATED_COST_PER_1K_TOKENS_USD = 0.003; // roughly mid-range for a small hosted model

function estimateCostUsd(promptChars, answerChars) {
  const totalWords = (promptChars + answerChars) / 5; // ~5 chars/word average
  const estimatedTokens = totalWords / 0.75;
  return (estimatedTokens / 1000) * ESTIMATED_COST_PER_1K_TOKENS_USD;
}

function ensureDataDir() {
  const dir = path.dirname(METRICS_FILE);
  if (!fs.existsSync(dir)) fs.mkdirSync(dir, { recursive: true });
}

function logRequest(entry) {
  ensureDataDir();
  const record = { ...entry, timestamp: new Date().toISOString() };
  fs.appendFileSync(METRICS_FILE, JSON.stringify(record) + '\n');
  return record;
}

function readAll() {
  ensureDataDir();
  if (!fs.existsSync(METRICS_FILE)) return [];
  return fs
    .readFileSync(METRICS_FILE, 'utf-8')
    .split('\n')
    .filter(Boolean)
    .map((line) => JSON.parse(line));
}

function recent(limit = 20) {
  const all = readAll();
  return all.slice(-limit).reverse();
}

function summary() {
  const all = readAll();
  if (all.length === 0) {
    return {
      totalRequests: 0,
      avgLatencyMs: 0,
      p95LatencyMs: 0,
      hallucinationFlagRate: 0,
      refusalRate: 0,
      avgNumericAccuracy: 0,
      totalEstimatedCostUsd: 0,
    };
  }

  const latencies = all.map((r) => r.latencyMs).sort((a, b) => a - b);
  const p95Index = Math.min(latencies.length - 1, Math.floor(0.95 * latencies.length));

  return {
    totalRequests: all.length,
    avgLatencyMs: round(latencies.reduce((a, b) => a + b, 0) / latencies.length),
    p95LatencyMs: round(latencies[p95Index]),
    hallucinationFlagRate: round(
      all.filter((r) => r.flaggedLowOverlap || (r.ungroundedNumbers && r.ungroundedNumbers.length > 0)).length /
        all.length,
      3
    ),
    refusalRate: round(all.filter((r) => r.isRefusal).length / all.length, 3),
    avgNumericAccuracy: round(
      all.reduce((sum, r) => sum + (r.numericAccuracy ?? 0), 0) / all.length,
      3
    ),
    totalEstimatedCostUsd: round(
      all.reduce((sum, r) => sum + (r.estimatedCostUsd ?? 0), 0),
      4
    ),
  };
}

function round(n, decimals = 1) {
  const factor = 10 ** decimals;
  return Math.round(n * factor) / factor;
}

module.exports = { logRequest, readAll, recent, summary, estimateCostUsd };
