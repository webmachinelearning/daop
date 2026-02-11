// src/daop-graph-builder.js

import { TensorDesc, IROperand, IRNode, IRGraph, generateId } from "./ir/graph-ir.js";
import { inferShape } from "./ir/shape-inference.js";
import { DAOPGraph } from "./daop-graph.js";

/**
 * DAOPGraphBuilder — mirrors WebNN MLGraphBuilder API.
 *
 * Builds an IR graph for weightless QoS estimation. When the application
 * later binds constants and calls compute(), the IR is replayed into a
 * real native MLGraphBuilder.
 */
export class DAOPGraphBuilder {
  /**
   * @param {DAOPContext} context
   */
  constructor(context) {
    this.context = context;
    this._ir = new IRGraph();
  }

  /**
   * Declare a named input operand.
   * Mirrors: MLGraphBuilder.input(name, descriptor)
   */
  input(name, descriptor) {
    const desc = new TensorDesc(descriptor);
    const operand = new IROperand({
      id: name,
      kind: "input",
      desc,
      name,
    });
    this._ir.addOperand(operand);
    return name;
  }

  /**
   * Declare a constant operand (optionally weightless).
   * Mirrors: MLGraphBuilder.constant(descriptor, bufferView?)
   *
   * DAOP Extension: When called with only a descriptor (no buffer),
   * creates a "weightless" constant that can be bound later via
   * context.bindConstants(). The descriptor MUST include a `label`
   * for late-binding identification.
   */
  constant(descriptorOrType, bufferOrValue = null) {
    // Support native 2-arg scalar form: constant(dataType, value)
    if (typeof descriptorOrType === "string") {
      return this.constantScalar(descriptorOrType, bufferOrValue);
    }
    const desc = new TensorDesc(descriptorOrType);
    const label = descriptorOrType.label || null;
    const id = label || generateId("const");
    const operand = new IROperand({
      id,
      kind: "constant",
      desc,
      label,
      buffer: bufferOrValue || null,
    });
    this._ir.addOperand(operand);
    return id;
  }

  /**
   * Scalar constant helper.
   * Mirrors: MLGraphBuilder.constant(dataType, value)
   */
  constantScalar(dataType, value) {
    const id = generateId("scalar");
    const desc = new TensorDesc({ shape: [], dataType });
    const operand = new IROperand({
      id,
      kind: "constant",
      desc,
      buffer: value, // Store scalar value directly
    });
    this._ir.addOperand(operand);
    return id;
  }

  // ─── Operators ───────────────────────────────────────────

  conv2d(input, filter, options = {}) {
    return this._addOp("conv2d", [input, filter], options);
  }

  convTranspose2d(input, filter, options = {}) {
    return this._addOp("convTranspose2d", [input, filter], options);
  }

  add(a, b) {
    return this._addOp("add", [a, b]);
  }

  sub(a, b) {
    return this._addOp("sub", [a, b]);
  }

  mul(a, b) {
    return this._addOp("mul", [a, b]);
  }

  div(a, b) {
    return this._addOp("div", [a, b]);
  }

  relu(input) {
    return this._addOp("relu", [input]);
  }

  sigmoid(input) {
    return this._addOp("sigmoid", [input]);
  }

  tanh(input) {
    return this._addOp("tanh", [input]);
  }

  clamp(input, options = {}) {
    return this._addOp("clamp", [input], options);
  }

  averagePool2d(input, options = {}) {
    return this._addOp("averagePool2d", [input], options);
  }

  maxPool2d(input, options = {}) {
    return this._addOp("maxPool2d", [input], options);
  }

  matmul(a, b) {
    return this._addOp("matmul", [a, b]);
  }

  softmax(input, axis) {
    return this._addOp("softmax", [input], { axis });
  }

  reshape(input, newShape) {
    return this._addOp("reshape", [input], { newShape });
  }

  transpose(input, options = {}) {
    return this._addOp("transpose", [input], options);
  }

  concat(inputs, axis) {
    return this._addOp("concat", inputs, { axis });
  }

  resample2d(input, options = {}) {
    return this._addOp("resample2d", [input], options);
  }

  // ─── Build ───────────────────────────────────────────────

  /**
   * Build the graph.
   * Mirrors: MLGraphBuilder.build(outputs)
   *
   * @param {Object<string, string>} outputs - Map of output name → operand ID
   * @returns {DAOPGraph}
   */
  build(outputs) {
    for (const [name, operandId] of Object.entries(outputs)) {
      this._ir.outputs.set(name, operandId);
    }
    return new DAOPGraph(this._ir, this.context);
  }

  // ─── Internal ────────────────────────────────────────────

  /**
   * Add an operator node to the IR, infer output shape, return output operand ID.
   */
  _addOp(opType, inputIds, attrs = {}) {
    const inputDescs = inputIds.map(id => {
      const operand = this._ir.getOperand(id);
      if (!operand) {
        throw new Error(`[DAOP] Unknown operand ID: ${id}`);
      }
      return { shape: operand.desc.shape, dataType: operand.desc.dataType };
    });

    const outputDescs = inferShape(opType, inputDescs, attrs);
    const outputIds = outputDescs.map((desc, i) => {
      const id = generateId(opType);
      const operand = new IROperand({
        id,
        kind: "intermediate",
        desc: new TensorDesc(desc),
      });
      this._ir.addOperand(operand);
      return id;
    });

    this._ir.addNode(new IRNode({
      opType,
      inputs: inputIds,
      outputs: outputIds,
      attrs,
    }));

    return outputIds.length === 1 ? outputIds[0] : outputIds;
  }
}
