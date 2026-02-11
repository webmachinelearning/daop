// src/ir/graph-ir.js

/**
 * DAOP Intermediate Representation for WebNN Graphs
 *
 * Records graph topology (operators, operands, connections) without
 * requiring actual weight buffers — enabling "weightless" QoS estimation.
 */

let _nextId = 0;

/** Generate a unique operand ID */
export function generateId(prefix = "op") {
  return `${prefix}_${_nextId++}`;
}

/** Reset ID counter (useful for testing) */
export function resetIdCounter() {
  _nextId = 0;
}

/**
 * Describes a tensor's metadata (no buffer data).
 */
export class TensorDesc {
  /**
   * @param {Object} opts
   * @param {number[]} opts.shape
   * @param {string} [opts.dataType="float32"]
   */
  constructor({ shape, dataType = "float32" }) {
    this.shape = [...shape];
    this.dataType = dataType;
  }

  /** Total number of elements */
  get elements() {
    return this.shape.reduce((a, b) => a * b, 1);
  }

  /** Bytes per element for this dataType */
  get bytesPerElement() {
    switch (this.dataType) {
      case "float32": return 4;
      case "float16": return 2;
      case "int32": return 4;
      case "int8": return 1;
      case "uint8": return 1;
      default: return 4;
    }
  }

  /** Total byte size */
  get byteSize() {
    return this.elements * this.bytesPerElement;
  }
}

/**
 * An operand in the IR graph.
 */
export class IROperand {
  /**
   * @param {Object} opts
   * @param {string} opts.id - Unique identifier
   * @param {string} opts.kind - "input" | "constant" | "intermediate"
   * @param {TensorDesc} opts.desc - Tensor descriptor
   * @param {string} [opts.name] - User-facing name (for inputs)
   * @param {string} [opts.label] - Label for late-binding (for constants)
   * @param {ArrayBufferView|null} [opts.buffer] - Actual data (null for weightless)
   */
  constructor({ id, kind, desc, name = null, label = null, buffer = null }) {
    this.id = id;
    this.kind = kind;
    this.desc = desc;
    this.name = name;
    this.label = label;
    this.buffer = buffer;
  }

  get isWeightless() {
    return this.kind === "constant" && this.buffer === null;
  }
}

/**
 * An operator node in the IR graph.
 */
export class IRNode {
  /**
   * @param {Object} opts
   * @param {string} opts.opType - WebNN op name ("conv2d", "add", etc.)
   * @param {string[]} opts.inputs - Input operand IDs
   * @param {string[]} opts.outputs - Output operand IDs
   * @param {Object} [opts.attrs={}] - Op-specific attributes (strides, pads, etc.)
   */
  constructor({ opType, inputs, outputs, attrs = {} }) {
    this.opType = opType;
    this.inputs = [...inputs];
    this.outputs = [...outputs];
    this.attrs = { ...attrs };
  }
}

/**
 * Complete IR graph: operands + operators in topological order.
 */
export class IRGraph {
  constructor() {
    /** @type {Map<string, IROperand>} */
    this.operands = new Map();
    /** @type {IRNode[]} */
    this.nodes = [];
    /** @type {Map<string, string>} output name → operand ID */
    this.outputs = new Map();
  }

  addOperand(operand) {
    this.operands.set(operand.id, operand);
  }

  addNode(node) {
    this.nodes.push(node);
  }

  getOperand(id) {
    return this.operands.get(id);
  }

  /** All input operands */
  getInputs() {
    return [...this.operands.values()].filter(op => op.kind === "input");
  }

  /** All constant operands */
  getConstants() {
    return [...this.operands.values()].filter(op => op.kind === "constant");
  }

  /** All weightless constants (need binding before compute) */
  getWeightlessConstants() {
    return this.getConstants().filter(op => op.isWeightless);
  }

  /** Check if all constants have buffers bound */
  isFullyBound() {
    return this.getConstants().every(op => !op.isWeightless);
  }
}
