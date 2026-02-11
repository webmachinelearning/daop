// src/index.js

/**
 * DAOP Library — Public API
 *
 * Usage:
 *   import { initDAOP, detectWebNNSupport } from "./src/index.js";
 *   const result = initDAOP();
 *   if (!result.ok) { showError(result.error); return; }
 *   // Now use standard WebNN API — DAOP layer is active
 */

export { initDAOP, detectWebNNSupport, getNativeML } from "./polyfill.js";
export { DAOPContext } from "./daop-context.js";
export { DAOPGraphBuilder } from "./daop-graph-builder.js";
export { DAOPGraph } from "./daop-graph.js";
export { timeModelDatabase } from "./qos/interpolation/time-model.js";
export { estimateQoSInterp } from "./qos/estimate-qos-interp.js";
