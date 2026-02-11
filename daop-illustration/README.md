# DAOP Illustration

A working JavaScript illustration of the
[Dynamic AI Offloading Protocol (DAOP)](../explainer.md) using the WebNN API.

This project demonstrates how the proposed `estimateQoS()` API can work in practice:
an application builds a **weightless** computation graph, estimates performance via
operator micro-benchmarks and polynomial interpolation, decides whether to run locally
or offload to the cloud, and — if local — downloads weights and executes inference.

## Prerequisites

- A WebNN-capable browser (e.g., Chrome Canary with WebNN flags enabled).
  See [installation guide](https://webnn.io/en/learn/get-started/installation).
- Node.js (for the static file server).

## Running the Demo

```bash
npm install
npm start          # starts http://localhost:8080
```

Open `http://localhost:8080/examples/background-blur/background-blur-demo.html` in the WebNN-capable
browser.

### Background Blur Demo

The demo applies AI-powered background blur to a meeting photo using the MediaPipe Selfie
Segmentation model. The workflow follows the DAOP protocol:

1. **Build weightless graph** — the model topology is recorded without downloading weights.
2. **Estimate QoS** — per-operator micro-benchmarks and interpolation produce a performance tier.
3. **Offloading decision** — if the tier is acceptable (< 10 s), run locally; otherwise
   offload to cloud.
4. **Execute** — download weights, compile the native WebNN graph, run inference, apply blur.

The right column of the demo exposes DAOP internals: operator benchmarks, estimation
curves, computation graph visualization, and a timing comparison between estimated and
actual latency.

## Implementation Details

See [IMPLEMENTATION.md](./IMPLEMENTATION.md) for a detailed description of the estimation
strategy, including shape-aware micro-benchmarks, polynomial interpolation, and the 7-tier
performance classification.

## Project Structure

```
src/                             # DAOP library (reusable)
  index.js                       # Public API
  polyfill.js                    # WebNN detection + DAOP initialization
  daop-context.js                # Wraps native MLContext with estimateQoS / bindConstants / compute
  daop-graph-builder.js          # IR graph builder (weightless constants)
  daop-graph.js                  # IR graph + Mermaid visualization
  ir/
    graph-ir.js                  # TensorDesc, IROperand, IRNode, IRGraph
    shape-inference.js           # Shape inference for conv2d, pool, resample, etc.
  qos/
    estimate-qos-interp.js       # Interpolation-based QoS estimation + tier assignment
    interpolation/
      poly-fit.js                # Polynomial fitting (Normal Equations, Gaussian)
      time-model.js              # TimeModelDatabase (stores points, fits, predicts)
    microbench/
      bench-runner.js            # Multi-size benchmark runner
      op-benchmarks.js           # Per-op benchmark configurations (xs–xxl)
examples/
  background-blur/               # Background blur demo (self-contained)
    background-blur-demo.html    # Interactive two-column demo page
    selfie-model.js              # Model graph definition + weight loader
    blur-renderer.js             # Image processing + blur compositing
    meeting.jpg                  # Sample input image
```

## License

Apache 2.0
