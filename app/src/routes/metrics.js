const express = require('express');
const { summary, recent } = require('../metricsStore');

function createMetricsRoutes() {
  const router = express.Router();

  router.get('/summary', (req, res) => {
    res.json(summary());
  });

  router.get('/recent', (req, res) => {
    const limit = Number(req.query.limit) || 20;
    res.json(recent(limit));
  });

  return router;
}

module.exports = { createMetricsRoutes };
