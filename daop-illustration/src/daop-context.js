// src/daop-context.js

import { estimateQoSInterp } from "./qos/estimate-qos-interp.js";
import { DAOPGraphBuilder } from "./daop-graph-builder.js";

/**
 * DAOPContext — wraps native MLContext with DAOP extensions.
 *
 * Provides the standard MLContext interface plus:
 * - estimateQoS(graph, options) — QoS estimation on weightless graph
 * - bindConstants(graph, constants) — late-bind weight buffers
 * - compute(graph, inputs) — replay IR → native WebNN → execute
 */
export class DAOPContext {
  /**
   * @param {MLContext} nativeContext - The real browser-provided MLContext
   * @param {Object} options - Creation options (deviceType, etc.)
   */
  constructor(nativeContext, options = {}) {
    this._native = nativeContext;
    this.deviceType = options.deviceType || "gpu";
  }

  /** Access the underlying native MLContext */
  get nativeContext() {
    return this._native;
  }

  /**
   * DAOP Extension: Estimate QoS for a weightless graph.
   *
   * @param {import("./daop-graph.js").DAOPGraph} graph
   * @param {Object} [options={}]
   * @returns {Object} QoS report with performanceTier
   */
  async estimateQoS(graph, options = {}) {
    return estimateQoSInterp(graph, options);
  }

  /**
   * DAOP Extension: Bind constants to a weightless graph.
   *
   * @param {import("./daop-graph.js").DAOPGraph} graph
   * @param {Object<string, ArrayBufferView>} constants - label → typed array
   */
  async bindConstants(graph, constants) {
    graph.bindConstants(constants);
    return { status: "success" };
  }

  /**
   * DAOP Extension: Pre-compile a fully-bound graph for execution.
   *
   * Replays the IR into native WebNN and compiles it. Call this before
   * compute() to separate compilation latency from inference latency.
   *
   * @param {import("./daop-graph.js").DAOPGraph} graph
   * @returns {Promise<void>}
   */
  async compileGraph(graph) {
    if (!graph.isFullyBound()) {
      throw new Error("[DAOP] Cannot compile: not all constants are bound. Call bindConstants() first.");
    }
    await graph.compile(this._native);
  }

  /**
   * Compile and execute a fully-bound graph.
   *
   * Replays IR into native WebNN, compiles, dispatches, and returns results.
   *
   * @param {import("./daop-graph.js").DAOPGraph} graph
   * @param {Object<string, MLTensor|ArrayBufferView>} inputs
   * @returns {Promise<Object<string, ArrayBuffer>>}
   */
  async compute(graph, inputs) {
    if (!graph.isFullyBound()) {
      throw new Error("[DAOP] Cannot compute: not all constants are bound. Call bindConstants() first.");
    }

    // Compile (replay IR → native WebNN graph)
    const { graph: nativeGraph, nativeContext } = await graph.compile(this._native);

    // Create input/output tensors
    const inputTensors = {};
    const outputTensors = {};

    // Create input MLTensors
    for (const irInput of graph.ir.getInputs()) {
      const inputData = inputs[irInput.name];
      if (!inputData) {
        throw new Error(`[DAOP] Missing input "${irInput.name}"`);
      }
      const tensor = await nativeContext.createTensor({
        dataType: irInput.desc.dataType,
        shape: irInput.desc.shape,
        writable: true,
        readable: false,
      });
      nativeContext.writeTensor(tensor, inputData);
      inputTensors[irInput.name] = tensor;
    }

    // Create output MLTensors
    for (const [outName, outOpId] of graph.ir.outputs) {
      const outOp = graph.ir.getOperand(outOpId);
      const tensor = await nativeContext.createTensor({
        dataType: outOp.desc.dataType,
        shape: outOp.desc.shape,
        writable: false,
        readable: true,
      });
      outputTensors[outName] = tensor;
    }

    // Dispatch
    nativeContext.dispatch(nativeGraph, inputTensors, outputTensors);

    // Read back results
    const results = {};
    for (const [name, tensor] of Object.entries(outputTensors)) {
      results[name] = await nativeContext.readTensor(tensor);
    }

    // Cleanup tensors
    for (const t of Object.values(inputTensors)) t.destroy();
    for (const t of Object.values(outputTensors)) t.destroy();

    return results;
  }
}
