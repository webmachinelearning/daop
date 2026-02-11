// src/daop-graph.js

import { getNativeMLGraphBuilder } from "./polyfill.js";

/**
 * DAOPGraph — IR graph container with replay capability.
 *
 * Holds the recorded IR from DAOPGraphBuilder. Provides:
 * - Topology inspection for QoS estimation and visualization
 * - Weight binding for labeled constants
 * - Replay into native WebNN for real execution
 */
export class DAOPGraph {
  /**
   * @param {import("./ir/graph-ir.js").IRGraph} ir
   * @param {import("./daop-context.js").DAOPContext} context
   */
  constructor(ir, context) {
    this._ir = ir;
    this._context = context;
    this._boundWeights = new Map();
    this._compiledGraph = null; // Cached native MLGraph
    this._compiledTensors = null; // Cached native MLTensors for constants
  }

  /** Access the IR for estimation and visualization */
  get ir() {
    return this._ir;
  }

  /** Get all nodes (operators) for iteration */
  get nodes() {
    return this._ir.nodes;
  }

  /** Get all operands */
  get operands() {
    return this._ir.operands;
  }

  /**
   * Bind weight buffers to labeled constants.
   *
   * @param {Object<string, ArrayBufferView>} constants - label → buffer map
   */
  bindConstants(constants) {
    for (const [label, buffer] of Object.entries(constants)) {
      const operand = this._ir.getOperand(label);
      if (!operand) {
        console.warn(`[DAOP] bindConstants: no operand with label "${label}"`);
        continue;
      }
      if (operand.kind !== "constant") {
        console.warn(`[DAOP] bindConstants: operand "${label}" is not a constant`);
        continue;
      }
      operand.buffer = buffer;
      this._boundWeights.set(label, buffer);
    }
    // Invalidate cached compilation
    this._compiledGraph = null;
    this._compiledTensors = null;
  }

  /**
   * Check if all constants have been bound.
   */
  isFullyBound() {
    return this._ir.isFullyBound();
  }

  /**
   * Replay the IR into a native WebNN graph and compile it.
   *
   * @param {MLContext} nativeContext - The real native WebNN context
   * @returns {Promise<{graph: MLGraph, nativeContext: MLContext}>}
   */
  async compile(nativeContext) {
    if (this._compiledGraph) {
      return this._compiledGraph;
    }

    // Verify all constants are bound
    const unbound = this._ir.getWeightlessConstants();
    if (unbound.length > 0) {
      const labels = unbound.map(op => op.label || op.id).join(", ");
      throw new Error(`[DAOP] Cannot compile: unbound constants: ${labels}`);
    }

    const NativeBuilder = getNativeMLGraphBuilder();
    const nativeBuilder = new NativeBuilder(nativeContext);
    const operandMap = new Map(); // IR operand ID → native MLOperand

    // 1. Create input operands
    for (const irOp of this._ir.getInputs()) {
      const nativeOperand = nativeBuilder.input(irOp.name, {
        dataType: irOp.desc.dataType,
        shape: irOp.desc.shape,
      });
      operandMap.set(irOp.id, nativeOperand);
    }

    // 2. Create constant operands (with bound buffers)
    for (const irOp of this._ir.getConstants()) {
      let nativeOperand;
      if (irOp.buffer !== null && typeof irOp.buffer === "number") {
        // Scalar constant
        nativeOperand = nativeBuilder.constant(irOp.desc.dataType, irOp.buffer);
      } else if (irOp.buffer !== null) {
        // Constant with buffer data
        nativeOperand = nativeBuilder.constant(
          { dataType: irOp.desc.dataType, shape: irOp.desc.shape },
          irOp.buffer
        );
      } else {
        throw new Error(`[DAOP] Constant "${irOp.label || irOp.id}" has no buffer`);
      }
      operandMap.set(irOp.id, nativeOperand);
    }

    // 3. Replay operator nodes in topological order (they're already recorded in order)
    for (const node of this._ir.nodes) {
      const nativeInputs = node.inputs.map(id => {
        const op = operandMap.get(id);
        if (!op) throw new Error(`[DAOP] Replay: missing operand ${id} for ${node.opType}`);
        return op;
      });

      let nativeOutput;
      const resolvedAttrs = { ...node.attrs };
      if (resolvedAttrs.bias && typeof resolvedAttrs.bias === "string") {
        resolvedAttrs.bias = operandMap.get(resolvedAttrs.bias);
        if (!resolvedAttrs.bias) {
          throw new Error(`[DAOP] Replay: missing bias operand for ${node.opType}`);
        }
      }

      switch (node.opType) {
        case "conv2d":
          nativeOutput = nativeBuilder.conv2d(nativeInputs[0], nativeInputs[1], resolvedAttrs);
          break;
        case "convTranspose2d":
          nativeOutput = nativeBuilder.convTranspose2d(nativeInputs[0], nativeInputs[1], resolvedAttrs);
          break;
        case "add":
          nativeOutput = nativeBuilder.add(nativeInputs[0], nativeInputs[1]);
          break;
        case "sub":
          nativeOutput = nativeBuilder.sub(nativeInputs[0], nativeInputs[1]);
          break;
        case "mul":
          nativeOutput = nativeBuilder.mul(nativeInputs[0], nativeInputs[1]);
          break;
        case "div":
          nativeOutput = nativeBuilder.div(nativeInputs[0], nativeInputs[1]);
          break;
        case "relu":
          nativeOutput = nativeBuilder.relu(nativeInputs[0]);
          break;
        case "sigmoid":
          nativeOutput = nativeBuilder.sigmoid(nativeInputs[0]);
          break;
        case "tanh":
          nativeOutput = nativeBuilder.tanh(nativeInputs[0]);
          break;
        case "clamp":
          nativeOutput = nativeBuilder.clamp(nativeInputs[0], resolvedAttrs);
          break;
        case "averagePool2d":
          nativeOutput = nativeBuilder.averagePool2d(nativeInputs[0], resolvedAttrs);
          break;
        case "maxPool2d":
          nativeOutput = nativeBuilder.maxPool2d(nativeInputs[0], resolvedAttrs);
          break;
        case "matmul":
          nativeOutput = nativeBuilder.matmul(nativeInputs[0], nativeInputs[1]);
          break;
        case "softmax":
          nativeOutput = nativeBuilder.softmax(nativeInputs[0], resolvedAttrs.axis);
          break;
        case "reshape":
          nativeOutput = nativeBuilder.reshape(nativeInputs[0], resolvedAttrs.newShape);
          break;
        case "transpose":
          nativeOutput = nativeBuilder.transpose(nativeInputs[0], resolvedAttrs);
          break;
        case "concat":
          nativeOutput = nativeBuilder.concat(nativeInputs, resolvedAttrs.axis);
          break;
        case "resample2d":
          nativeOutput = nativeBuilder.resample2d(nativeInputs[0], resolvedAttrs);
          break;
        default:
          throw new Error(`[DAOP] Replay: unsupported op "${node.opType}"`);
      }

      // Map IR output IDs to native operands
      if (Array.isArray(nativeOutput)) {
        node.outputs.forEach((id, i) => operandMap.set(id, nativeOutput[i]));
      } else {
        node.outputs.forEach(id => operandMap.set(id, nativeOutput));
      }
    }

    // 4. Build the native graph
    const nativeOutputs = {};
    for (const [name, operandId] of this._ir.outputs) {
      nativeOutputs[name] = operandMap.get(operandId);
    }

    const nativeGraph = await nativeBuilder.build(nativeOutputs);

    this._compiledGraph = { graph: nativeGraph, nativeContext };
    return this._compiledGraph;
  }

  /**
   * Generate Mermaid diagram code for this graph.
   * Used by the demo's right panel for visualization.
   *
   * @param {Object[]} [qosBreakdown] - Optional per-node QoS data for annotations
   * @returns {string} Mermaid flowchart code
   */
  toMermaid(qosBreakdown = null) {
    let code = "graph TD\n";
    code += "  classDef computeBound fill:#ffcdd2,stroke:#e53935,stroke-width:2px;\n";
    code += "  classDef memoryBound fill:#bbdefb,stroke:#1e88e5,stroke-width:2px;\n";
    code += "  classDef inputNode fill:#e8f5e9,stroke:#2e7d32,stroke-width:2px;\n";
    code += "  classDef constantNode fill:#fff3e0,stroke:#ef6c00,stroke-width:1px;\n";

    // Track which operand IDs are produced by which node
    const producedBy = new Map();

    this._ir.nodes.forEach((node, i) => {
      const nodeId = `op${i}`;
      let label;
      if (qosBreakdown && qosBreakdown[i]) {
        const est = qosBreakdown[i];
        const boundIcon = est.bottleneck === "compute" ? "COMPUTE" : "MEMORY";
        label = `"${boundIcon}<br/><b>${node.opType.toUpperCase()}</b><br/>${est.timeMs.toFixed(2)}ms"`;
        const cls = est.bottleneck === "compute" ? "computeBound" : "memoryBound";
        code += `  ${nodeId}[${label}]:::${cls}\n`;
      } else {
        // Show shapes in label
        const outOp = this._ir.getOperand(node.outputs[0]);
        const shapeStr = outOp ? outOp.desc.shape.join("x") : "?";
        label = `"<b>${node.opType.toUpperCase()}</b><br/>[${shapeStr}]"`;
        code += `  ${nodeId}[${label}]\n`;
      }

      node.outputs.forEach(outId => producedBy.set(outId, nodeId));
    });

    // Add edges
    this._ir.nodes.forEach((node, i) => {
      const nodeId = `op${i}`;
      node.inputs.forEach(inputId => {
        const sourceNode = producedBy.get(inputId);
        if (sourceNode) {
          code += `  ${sourceNode} --> ${nodeId}\n`;
        } else {
          // It's a graph input or constant
          const operand = this._ir.getOperand(inputId);
          if (operand) {
            const displayName = operand.name || operand.label || inputId;
            if (operand.kind === "input") {
              code += `  ${inputId}([${displayName}]):::inputNode --> ${nodeId}\n`;
            } else if (operand.kind === "constant") {
              // Don't clutter graph with every constant; only show labeled ones
              if (operand.label) {
                code += `  ${inputId}[/${operand.label}/]:::constantNode --> ${nodeId}\n`;
              }
            }
          }
        }
      });
    });

    return code;
  }
}
