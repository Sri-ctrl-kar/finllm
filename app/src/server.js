const express = require('express');
const path = require('path');

const { createAskRoute } = require('./routes/ask');
const { createMetricsRoutes } = require('./routes/metrics');

const PORT = process.env.PORT || 3002;

const app = express();
app.use(express.json());
app.use(express.static(path.join(__dirname, '..', 'public')));

app.get('/health', (req, res) => res.json({ status: 'ok' }));
app.use('/api/ask', createAskRoute());
app.use('/api/metrics', createMetricsRoutes());

app.listen(PORT, () => {
  console.log(`FinLLM-RAG Console listening on http://localhost:${PORT}`);
  console.log(`Dashboard: http://localhost:${PORT}/dashboard.html`);
  console.log(`Make sure the RAG service is also running:`);
  console.log(`  (from finllm-rag/ root) python -m src.service.rag_server`);
});
