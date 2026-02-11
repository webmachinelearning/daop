// src/qos/estimate-qos-interp.js

import { timeModelDatabase } from "./interpolation/time-model.js";

/**
 * Estimate QoS using direct interpolation from measured benchmark data.
 *
 * This estimator directly predicts per-operator execution time from
 * measured (inputSize -> time) data points using polynomial regression.
 *
 * @param {import("../daop-graph.js").DAOPGraph} daopGraph
 * @param {Object} [options={}]
 * @returns {Object} QoS report
 */
export function estimateQoSInterp(daopGraph, options = {}) {
  const ir = daopGraph.ir;
  let totalTimeMs = 0;
  const breakdown = [];

  for (const node of ir.nodes) {
    const inputDescs = node.inputs.map(id => ir.getOperand(id));
    const primaryElements = (inputDescs[0] && inputDescs[0].desc)
      ? inputDescs[0].desc.elements : 0;

    const predictedMs = timeModelDatabase.predict(node.opType, primaryElements);
    totalTimeMs += predictedMs;

    breakdown.push({
      opType: node.opType,
      timeMs: predictedMs,
      inputElements: primaryElements,
    });
  }

  // Graph-level dispatch overhead (single dispatch for compiled graph)
  const graphDispatchOverheadMs = 0.5 + ir.nodes.length * 0.005;
  totalTimeMs += graphDispatchOverheadMs;

  const performanceTier = totalTimeMs < 16 ? "excellent"
    : totalTimeMs < 100 ? "good"
    : totalTimeMs < 1000 ? "fair"
    : totalTimeMs < 10000 ? "moderate"
    : totalTimeMs < 30000 ? "slow"
    : totalTimeMs < 60000 ? "very-slow"
    : "poor";

  return {
    performanceTier,
    internal: {
      totalTimeMs,
      graphDispatchOverheadMs,
      breakdown,
      method: "interpolation",
    },
  };
}
