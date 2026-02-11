// src/qos/microbench/bench-runner.js

import { getNativeML, getNativeMLGraphBuilder } from "../../polyfill.js";
import { OP_BENCH_CONFIGS } from "./op-benchmarks.js";
import { timeModelDatabase } from "../interpolation/time-model.js";

function fillRandomBuffer(uint8) {
  const CHUNK = 65536;
  for (let offset = 0; offset < uint8.length; offset += CHUNK) {
    const end = Math.min(offset + CHUNK, uint8.length);
    crypto.getRandomValues(uint8.subarray(offset, end));
  }
}

export class BenchRunner {
  constructor(options = {}) {
    this.warmupIterations = options.warmupIterations || 5;
    this.measureIterations = options.measureIterations || 30;
    this.deviceType = options.deviceType || "gpu";
    /** @type {number} Measured dispatch+readTensor overhead in ms (set by _measureBaselineOverhead) */
    this._baselineOverheadMs = 0;
  }

  /**
   * Measure the fixed dispatch + readTensor overhead using a trivial graph.
   * Builds a tiny reshape [1,1,1,1] → [1,1,1,1], runs the same warmup +
   * batched-measurement loop used for real operators, and returns the median
   * per-dispatch time.  This captures GPU command-submission and sync cost
   * with negligible compute, so we can subtract it from real measurements.
   *
   * Runs multiple independent rounds and returns the median-of-medians for
   * higher accuracy, since this value is subtracted from every operator
   * measurement.
   */
  async _measureBaselineOverhead(onProgress) {
    if (onProgress) onProgress({ phase: "baseline", label: "measuring dispatch overhead" });

    const nativeML = getNativeML();
    const NativeBuilder = getNativeMLGraphBuilder();

    const rounds = 3;
    const roundMedians = [];

    for (let r = 0; r < rounds; r++) {
      const context = await nativeML.createContext({ deviceType: this.deviceType });
      const builder = new NativeBuilder(context);

      const shape = [1, 1, 1, 1];
      const input = builder.input("input", { dataType: "float32", shape });
      const output = builder.reshape(input, shape);
      const graph = await builder.build({ output });

      const inputTensor = await context.createTensor({
        dataType: "float32",
        shape,
        writable: true,
        readable: false,
      });
      const outputTensor = await context.createTensor({
        dataType: "float32",
        shape,
        writable: false,
        readable: true,
      });

      const inputData = new Float32Array(1);
      inputData[0] = 1.0;
      context.writeTensor(inputTensor, inputData);

      // Warmup
      for (let i = 0; i < this.warmupIterations; i++) {
        context.dispatch(graph, { input: inputTensor }, { output: outputTensor });
        await context.readTensor(outputTensor);
      }

      // Measurement — same batched loop as _benchmarkSizeVariant
      const batchSize = 10;
      const batchTimes = [];
      for (let i = 0; i < 50; i++) {
        const batchStart = performance.now();
        for (let j = 0; j < batchSize; j++) {
          context.dispatch(graph, { input: inputTensor }, { output: outputTensor });
        }
        await context.readTensor(outputTensor);
        const batchEnd = performance.now();
        batchTimes.push((batchEnd - batchStart) / batchSize);
      }

      inputTensor.destroy();
      outputTensor.destroy();

      batchTimes.sort((a, b) => a - b);
      roundMedians.push(batchTimes[Math.floor(batchTimes.length / 2)]);
    }

    // Median-of-medians across rounds
    roundMedians.sort((a, b) => a - b);
    return roundMedians[Math.floor(roundMedians.length / 2)];
  }

  /**
   * Benchmark a single size variant of an operator.
   * Builds the graph, warms up, measures, and returns timing statistics.
   */
  async _benchmarkSizeVariant(opType, sizeConfig, onProgress) {
    const nativeML = getNativeML();
    const context = await nativeML.createContext({ deviceType: this.deviceType });
    const NativeBuilder = getNativeMLGraphBuilder();
    const builder = new NativeBuilder(context);

    if (onProgress) onProgress({ phase: "building", opType, label: sizeConfig.label });

    const { outputs, inputName, inputShape, flops, bytes, outputShape, totalElements } = sizeConfig.buildGraph(builder);
    const graph = await builder.build(outputs);

    const inputTensor = await context.createTensor({
      dataType: "float32",
      shape: inputShape,
      writable: true,
      readable: false,
    });
    const outputTensor = await context.createTensor({
      dataType: "float32",
      shape: outputShape,
      writable: false,
      readable: true,
    });

    const inputData = new Float32Array(inputShape.reduce((a, b) => a * b, 1));
    fillRandomBuffer(new Uint8Array(inputData.buffer));
    context.writeTensor(inputTensor, inputData);

    if (onProgress) onProgress({ phase: "warmup", opType, label: sizeConfig.label });
    for (let i = 0; i < this.warmupIterations; i++) {
      context.dispatch(graph, { [inputName]: inputTensor }, { output: outputTensor });
      await context.readTensor(outputTensor);
    }

    if (onProgress) onProgress({ phase: "measuring", opType, label: sizeConfig.label });
    const batchSize = 10;
    const batchTimes = [];
    for (let i = 0; i < this.measureIterations; i++) {
      const batchStart = performance.now();
      for (let j = 0; j < batchSize; j++) {
        context.dispatch(graph, { [inputName]: inputTensor }, { output: outputTensor });
      }
      await context.readTensor(outputTensor);
      const batchEnd = performance.now();
      batchTimes.push((batchEnd - batchStart) / batchSize);
    }

    inputTensor.destroy();
    outputTensor.destroy();

    batchTimes.sort((a, b) => a - b);
    const medianMs = batchTimes[Math.floor(batchTimes.length / 2)];
    const p90Ms = batchTimes[Math.floor(batchTimes.length * 0.9)];
    const meanMs = batchTimes.reduce((a, b) => a + b, 0) / batchTimes.length;
    const minMs = batchTimes[0];

    // Compute throughput parameters from the overhead-corrected median time.
    const correctedMedianMs = Math.max(0.001, medianMs - this._baselineOverheadMs);
    const correctedP90Ms = Math.max(0.001, p90Ms - this._baselineOverheadMs);
    const correctedMeanMs = Math.max(0.001, meanMs - this._baselineOverheadMs);
    const correctedMinMs = Math.max(0.001, minMs - this._baselineOverheadMs);

    const gflops = (flops / 1e9) / (correctedMedianMs / 1000);
    const bandwidthGBs = (bytes / 1e9) / (correctedMedianMs / 1000);

    return {
      opType,
      label: sizeConfig.label,
      totalElements,
      medianMs: correctedMedianMs,
      p90Ms: correctedP90Ms,
      meanMs: correctedMeanMs,
      minMs: correctedMinMs,
      overheadMs: this._baselineOverheadMs,
      gflops,
      bandwidthGBs,
      arithmeticIntensity: flops / bytes,
      flops,
      bytes,
    };
  }

  /**
   * Benchmark a single operator across all its size variants.
   */
  async benchmarkOp(opType, onProgress) {
    const config = OP_BENCH_CONFIGS[opType];
    if (!config) {
      throw new Error(`[DAOP Bench] No benchmark config for op: ${opType}`);
    }

    const sizeResults = [];
    for (const sizeConfig of config.sizes) {
      const result = await this._benchmarkSizeVariant(opType, sizeConfig, onProgress);
      sizeResults.push(result);

      await new Promise(r => setTimeout(r, 30));
    }

    // Store in TimeModelDatabase for interpolation estimator
    for (const result of sizeResults) {
      timeModelDatabase.addDataPoint(opType, {
        totalElements: result.totalElements,
        medianMs: result.medianMs,
        minMs: result.minMs,
        p90Ms: result.p90Ms,
        label: result.label,
      });
    }

    if (onProgress) onProgress({ phase: "done", opType, result: sizeResults });

    return sizeResults;
  }

  async benchmarkAll(onProgress) {
    const results = [];
    const ops = Object.keys(OP_BENCH_CONFIGS);

    // Measure baseline dispatch+readTensor overhead before benchmarking operators
    this._baselineOverheadMs = await this._measureBaselineOverhead(onProgress);
    if (onProgress) {
      onProgress({
        phase: "baseline-done",
        baselineMs: this._baselineOverheadMs,
      });
    }

    for (let i = 0; i < ops.length; i++) {
      const opType = ops[i];
      if (onProgress) {
        onProgress({
          phase: "start",
          opType,
          index: i,
          total: ops.length,
        });
      }

      try {
        const sizeResults = await this.benchmarkOp(opType, onProgress);
        results.push({ opType, sizes: sizeResults });
      } catch (err) {
        console.error(`[DAOP Bench] Failed to benchmark ${opType}:`, err);
        results.push({ opType, error: err.message });
      }

      await new Promise(r => setTimeout(r, 50));
    }

    // Fit polynomial curves for the interpolation estimator
    timeModelDatabase.fitAll();

    return results;
  }

  getAvailableOps() {
    return Object.keys(OP_BENCH_CONFIGS);
  }
}
