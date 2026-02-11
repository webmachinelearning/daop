# DAOP Illustration: Reference Implementation

> **Note on Illustration Purposes**: This implementation is provided for **illustration purposes** to demonstrate the feasibility of the `estimateQoS()` API. It uses a simplified log-log polynomial interpolation approach. A production implementation could employ other strategies — such as Roofline models, learned cost models, hardware-specific operator libraries, or ML-based performance predictors — depending on the target platform and accuracy requirements.

## 1. Overview
This document describes the implementation strategy for the `estimateQoS()` API in the DAOP (Dynamic AI Offloading Protocol) illustration. The estimation strategy uses **log-log polynomial interpolation** based on operator-level micro-benchmarks.

The internals of these estimations are entirely opaque to the application. The application receives only a high-level performance tier (e.g., "excellent" or "fair"), allowing for hardware-agnostic offloading decisions without exposing raw timing data or device-specific characteristics.

## 2. Performance Tiers
The implementation categorizes the estimated graph latency into one of seven performance tiers. These tiers correspond to typical user experience expectations:

| Tier | Latency Threshold | Description |
|------|-------------------|-------------|
| excellent | < 16ms | Real-time (60fps) performance |
| good | < 100ms | Interactive / seamless UI |
| fair | < 1s | Responsive but noticeable |
| moderate | < 10s | Tolerable for background tasks |
| slow | < 30s | Significant wait time |
| very-slow | < 60s | Approaching timeout limits |
| poor | ≥ 60s | Unacceptable performance |

## 3. Estimation Strategy: Log-Log Polynomial Interpolation
Empirical observations show that operator execution time on modern hardware often follows a power-law relationship with the total number of processed elements:
`time ≈ a · (totalElements)^b`

By taking the logarithm of both sides, this relationship becomes linear in log-log space:
`log(time) = log(a) + b · log(totalElements)`

This implementation fits a **degree-1 polynomial** (linear) in log-log space:
`log(time) = c0 + c1 · log(n)`

The coefficients (`c0`, `c1`) are found using least-squares normal equations, solved via **Gaussian elimination with partial pivoting**.

#### Small-Size Noise Handling (Clamp)

At small input sizes, GPU dispatch overhead can dominate actual computation time, producing a U-shaped curve in log-log space — small inputs appear slower than medium ones. Left unchecked, a polynomial fit on this data extrapolates catastrophically for very small inputs.

To address this, the fitter applies a **left-side clamp**:
1. Find the measured point with the **minimum medianMs** (the "valley" of the U).
2. **Clamp** all points to the left of it to that minimum value.
3. **Fit** the degree-1 polynomial using only points from the minimum onward — clamped points are excluded from the fit.
4. At **prediction time**, any input size at or below the minimum-point's size returns the flat clamp value instead of polynomial extrapolation.

This ensures monotonic (non-decreasing) predictions: small inputs never produce absurdly high time estimates.

**Prediction Process:**
1. Calculate the natural log of the input element count: `ln_n = log(totalElements)`.
2. If `ln_n` is at or below the clamp boundary, return the clamped floor value directly.
3. Otherwise, evaluate the fitted polynomial: `ln_time = polyEval(coeffs, ln_n)`.
4. Revert to time domain: `estimatedTime = exp(ln_time)`.

If a polynomial has not yet been fitted for an operator (e.g., during the first calibration run), the system falls back to piecewise linear interpolation on the raw benchmarked data points.

## 4. Shape-Aware Micro-Benchmarks
To populate the estimation models, the implementation runs a suite of micro-benchmarks for supported operators across six size variants.

### Size Variants
Benchmarking across multiple sizes captures the "utilization curve" where small tensors may not fully saturate compute units.

| Variant | Representative Shape | Total Elements |
|---------|----------------------|----------------|
| xs | [1, 16, 16, 8] | 2,048 |
| small | [1, 32, 32, 16] | 16,384 |
| medium | [1, 64, 64, 24] | 98,304 |
| large | [1, 128, 128, 32] | 524,288 |
| xl | [1, 256, 256, 32] | 2,097,152 |
| xxl | [1, 512, 512, 32] | 8,388,608 |

### Benchmark Methodology
The system benchmarks 7 operator types: `conv2d`, `add`, `mul`, `relu`, `sigmoid`, `clamp`, and `averagePool2d`.

1. **Baseline Overhead Subtraction**: Before benchmarking real operators, the runner measures the dispatch + readTensor overhead using a trivial (reshape) graph at a small fixed size. To improve accuracy, this measurement is repeated across 3 independent rounds (each with 50 batched iterations), and the median-of-medians is used. This baseline is subtracted from each operator's measured time to isolate pure compute cost.
2. **Amortized Readback**: The runner dispatches 10 operations (batchSize=10) before a single `readTensor` call, further reducing per-dispatch synchronization overhead.
3. **Iterations**: Each benchmark performs 5 warmup runs followed by 30 timed iterations to find the median latency.
4. **Storage**: Raw data points `{ totalElements, medianMs }` are stored in the `TimeModelDatabase`.

## 5. End-to-End Estimation Flow

### Benchmark Phase (Offline/Calibration)
1. Measure baseline dispatch + readTensor overhead using a trivial graph.
2. Iterate through supported operator types and size variants.
3. Execute benchmarks, subtract baseline overhead, and record median latencies.
4. Store results in `TimeModelDatabase` and fit log-log polynomials.
5. Persist models to `localStorage` under the key `"daop_time_models"`.

### Estimation Phase (Online)
1. **Traverse Graph**: Walk the IR of the weightless graph.
2. **Sum Node Latencies**: For each node, look up the operator in `TimeModelDatabase`.
   - Call `predict(opType, inputElements)` to get the estimated time.
3. **Add Overhead**: Add a graph dispatch overhead: `0.5 + numNodes * 0.005 ms`.
4. **Assign Tier**: Map the total estimated latency to a performance tier string.

## 6. Project Structure

```
daop-illustration/
  src/
    index.js                     # Public API entry point
    polyfill.js                  # WebNN feature detection and DAOP initialization
    daop-context.js              # Wraps native MLContext; delegates to interpolation estimator
    daop-graph-builder.js        # IR graph builder supporting weightless constants
    daop-graph.js                # IR graph representation and Mermaid export
    ir/
      graph-ir.js                # Core IR definitions (TensorDesc, IRNode)
      shape-inference.js         # Shape inference logic for operators
    qos/
      estimate-qos-interp.js     # Interpolation-based QoS estimator
      interpolation/
        poly-fit.js              # Polynomial fitting (Normal Equations, Gaussian)
        time-model.js            # TimeModelDatabase (stores points, fits, predicts)
      microbench/
        bench-runner.js          # Hardware-specific benchmark execution engine
        op-benchmarks.js         # Operator configurations (xs, small, medium, large, xl, xxl)
  examples/
    background-blur/
      background-blur-demo.html  # Interactive two-column demo page
      selfie-model.js            # Model graph definition + weight loader
      blur-renderer.js           # Image processing + blur compositing
      meeting.jpg                # Sample input image
```

