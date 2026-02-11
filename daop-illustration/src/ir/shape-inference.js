// src/ir/shape-inference.js

/**
 * Shape inference for WebNN operators.
 *
 * Given input shapes and op attributes, computes the output shape(s).
 * Supports the operator subset used by Selfie Segmentation + common ops.
 */

/**
 * Infer output shape for a WebNN operator.
 *
 * @param {string} opType - Operator name
 * @param {Array<{shape: number[], dataType: string}>} inputs - Input descriptors
 * @param {Object} attrs - Operator attributes
 * @returns {{shape: number[], dataType: string}[]} Output descriptor(s)
 */
export function inferShape(opType, inputs, attrs = {}) {
  const fn = SHAPE_FNS[opType];
  if (!fn) {
    throw new Error(`[DAOP] Shape inference not implemented for op: ${opType}`);
  }
  return fn(inputs, attrs);
}

/**
 * Check if shape inference is available for an op type.
 */
export function hasShapeInference(opType) {
  return opType in SHAPE_FNS;
}

const SHAPE_FNS = {
  conv2d(inputs, attrs) {
    // inputs[0] = input tensor, inputs[1] = filter tensor
    // WebNN conv2d supports both NCHW and NHWC layouts
    const layout = attrs.inputLayout || "nchw";
    const input = inputs[0].shape;
    const filter = inputs[1].shape;

    let batch, inH, inW, outChannels, filterH, filterW;
    if (layout === "nchw") {
      [batch, , inH, inW] = input;
      // filter layout for nchw: [outChannels, inChannels/groups, filterH, filterW]
      const filterLayout = attrs.filterLayout || "oihw";
      if (filterLayout === "oihw") {
        [outChannels, , filterH, filterW] = filter;
      } else if (filterLayout === "hwio") {
        [filterH, filterW, , outChannels] = filter;
      } else {
        [outChannels, , filterH, filterW] = filter;
      }
    } else {
      // nhwc
      [batch, inH, inW, ] = input;
      const filterLayout = attrs.filterLayout || "ohwi";
      if (filterLayout === "ohwi") {
        [outChannels, filterH, filterW, ] = filter;
      } else if (filterLayout === "hwio") {
        [filterH, filterW, , outChannels] = filter;
      } else if (filterLayout === "ihwo") {
        // [inputChannels/groups, filterH, filterW, outputChannels]
        outChannels = filter[3];
        filterH = filter[1];
        filterW = filter[2];
      } else {
        [outChannels, filterH, filterW, ] = filter;
      }
    }

    const padding = attrs.padding || [0, 0, 0, 0]; // [top, bottom, left, right]
    const strides = attrs.strides || [1, 1];
    const dilations = attrs.dilations || [1, 1];

    const effectiveFilterH = (filterH - 1) * dilations[0] + 1;
    const effectiveFilterW = (filterW - 1) * dilations[1] + 1;
    const outH = Math.floor((inH + padding[0] + padding[1] - effectiveFilterH) / strides[0]) + 1;
    const outW = Math.floor((inW + padding[2] + padding[3] - effectiveFilterW) / strides[1]) + 1;

    const outShape = layout === "nchw"
      ? [batch, outChannels, outH, outW]
      : [batch, outH, outW, outChannels];

    return [{ shape: outShape, dataType: inputs[0].dataType }];
  },

  // Element-wise binary ops: output shape = broadcast(input shapes)
  add: (inputs) => [{ shape: broadcastShape(inputs[0].shape, inputs[1].shape), dataType: inputs[0].dataType }],
  sub: (inputs) => [{ shape: broadcastShape(inputs[0].shape, inputs[1].shape), dataType: inputs[0].dataType }],
  mul: (inputs) => [{ shape: broadcastShape(inputs[0].shape, inputs[1].shape), dataType: inputs[0].dataType }],
  div: (inputs) => [{ shape: broadcastShape(inputs[0].shape, inputs[1].shape), dataType: inputs[0].dataType }],

  // Element-wise unary ops: output shape = input shape
  relu: (inputs) => [{ shape: [...inputs[0].shape], dataType: inputs[0].dataType }],
  sigmoid: (inputs) => [{ shape: [...inputs[0].shape], dataType: inputs[0].dataType }],
  tanh: (inputs) => [{ shape: [...inputs[0].shape], dataType: inputs[0].dataType }],
  clamp: (inputs) => [{ shape: [...inputs[0].shape], dataType: inputs[0].dataType }],

  averagePool2d(inputs, attrs) {
    return [poolShape(inputs[0], attrs)];
  },

  maxPool2d(inputs, attrs) {
    return [poolShape(inputs[0], attrs)];
  },

  matmul(inputs) {
    const a = inputs[0].shape;
    const b = inputs[1].shape;
    // Support batched matmul: [...batch, M, K] x [...batch, K, N] -> [...batch, M, N]
    const m = a[a.length - 2];
    const n = b[b.length - 1];
    const batchDims = a.length > 2 ? a.slice(0, -2) : [];
    return [{ shape: [...batchDims, m, n], dataType: inputs[0].dataType }];
  },

  softmax(inputs) {
    return [{ shape: [...inputs[0].shape], dataType: inputs[0].dataType }];
  },

  reshape(inputs, attrs) {
    const newShape = attrs.newShape || attrs.shape;
    if (!newShape) {
      throw new Error("[DAOP] reshape requires newShape attribute");
    }
    return [{ shape: [...newShape], dataType: inputs[0].dataType }];
  },

  transpose(inputs, attrs) {
    const perm = attrs.permutation;
    if (!perm) {
      // Default: reverse dimensions
      const shape = [...inputs[0].shape].reverse();
      return [{ shape, dataType: inputs[0].dataType }];
    }
    const shape = perm.map(i => inputs[0].shape[i]);
    return [{ shape, dataType: inputs[0].dataType }];
  },

  concat(inputs, attrs) {
    const axis = attrs.axis || 0;
    const shape = [...inputs[0].shape];
    shape[axis] = inputs.reduce((sum, inp) => sum + inp.shape[axis], 0);
    return [{ shape, dataType: inputs[0].dataType }];
  },

  resample2d(inputs, attrs) {
    const layout = attrs.layout || "nchw";
    const shape = [...inputs[0].shape];

    if (attrs.axes && attrs.sizes) {
      const out = [...inputs[0].shape];
      attrs.axes.forEach((axis, i) => {
        out[axis] = attrs.sizes[i];
      });
      return [{ shape: out, dataType: inputs[0].dataType }];
    }
    if (attrs.axes && attrs.scales) {
      const out = [...inputs[0].shape];
      attrs.axes.forEach((axis, i) => {
        out[axis] = Math.floor(out[axis] * attrs.scales[i]);
      });
      return [{ shape: out, dataType: inputs[0].dataType }];
    }

    if (attrs.sizes) {
      // Explicit output sizes [outH, outW]
      if (layout === "nchw") {
        shape[2] = attrs.sizes[0];
        shape[3] = attrs.sizes[1];
      } else {
        shape[1] = attrs.sizes[0];
        shape[2] = attrs.sizes[1];
      }
    } else if (attrs.scales) {
      // Scale factors [scaleH, scaleW]
      if (layout === "nchw") {
        shape[2] = Math.floor(shape[2] * attrs.scales[0]);
        shape[3] = Math.floor(shape[3] * attrs.scales[1]);
      } else {
        shape[1] = Math.floor(shape[1] * attrs.scales[0]);
        shape[2] = Math.floor(shape[2] * attrs.scales[1]);
      }
    }
    return [{ shape, dataType: inputs[0].dataType }];
  },
  convTranspose2d(inputs, attrs) {
    const layout = attrs.inputLayout || "nchw";
    const input = inputs[0].shape;
    const filter = inputs[1].shape;

    let batch, inH, inW, outChannels, filterH, filterW;
    if (layout === "nchw") {
      [batch, , inH, inW] = input;
      [, outChannels, filterH, filterW] = filter;
    } else {
      // nhwc
      [batch, inH, inW, ] = input;
      const filterLayout = attrs.filterLayout || "ihwo";
      if (filterLayout === "ihwo") {
        outChannels = filter[3];
        filterH = filter[1];
        filterW = filter[2];
      } else if (filterLayout === "ohwi") {
        [outChannels, filterH, filterW, ] = filter;
      } else {
        outChannels = filter[3];
        filterH = filter[1];
        filterW = filter[2];
      }
    }

    const strides = attrs.strides || [1, 1];
    const padding = attrs.padding || [0, 0, 0, 0];
    const outputPadding = attrs.outputPadding || [0, 0];
    const dilations = attrs.dilations || [1, 1];

    if (attrs.outputSizes) {
      const [outH, outW] = attrs.outputSizes;
      const outShape = layout === "nchw"
        ? [batch, outChannels, outH, outW]
        : [batch, outH, outW, outChannels];
      return [{ shape: outShape, dataType: inputs[0].dataType }];
    }

    const outH = (inH - 1) * strides[0] - padding[0] - padding[1]
      + (filterH - 1) * dilations[0] + outputPadding[0] + 1;
    const outW = (inW - 1) * strides[1] - padding[2] - padding[3]
      + (filterW - 1) * dilations[1] + outputPadding[1] + 1;

    const outShape = layout === "nchw"
      ? [batch, outChannels, outH, outW]
      : [batch, outH, outW, outChannels];
    return [{ shape: outShape, dataType: inputs[0].dataType }];
  },
};

/**
 * Compute broadcast-compatible output shape.
 */
function broadcastShape(a, b) {
  const rank = Math.max(a.length, b.length);
  const result = new Array(rank);
  for (let i = 0; i < rank; i++) {
    const dimA = i < a.length ? a[a.length - 1 - i] : 1;
    const dimB = i < b.length ? b[b.length - 1 - i] : 1;
    if (dimA !== dimB && dimA !== 1 && dimB !== 1) {
      throw new Error(`[DAOP] Shape broadcast failed: ${a} vs ${b}`);
    }
    result[rank - 1 - i] = Math.max(dimA, dimB);
  }
  return result;
}

/**
 * Pool output shape helper.
 */
function poolShape(input, attrs) {
  const layout = attrs.layout || "nchw";
  const shape = input.shape;
  const windowDimensions = attrs.windowDimensions || [2, 2];
  const strides = attrs.strides || windowDimensions;
  const padding = attrs.padding || [0, 0, 0, 0];

  let batch, channels, inH, inW;
  if (layout === "nchw") {
    [batch, channels, inH, inW] = shape;
  } else {
    [batch, inH, inW, channels] = shape;
  }

  const outH = Math.floor((inH + padding[0] + padding[1] - windowDimensions[0]) / strides[0]) + 1;
  const outW = Math.floor((inW + padding[2] + padding[3] - windowDimensions[1]) / strides[1]) + 1;

  const outShape = layout === "nchw"
    ? [batch, channels, outH, outW]
    : [batch, outH, outW, channels];

  return { shape: outShape, dataType: input.dataType };
}
