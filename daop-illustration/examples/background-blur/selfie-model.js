// src/demo/background-blur/selfie-model.js
//
// MediaPipe Selfie Segmentation (General) — Model Graph Builder
// Input: [1, 256, 256, 3] NHWC float32 → Output: [1, 256, 256, 1] segmentation mask
//
// Architecture faithfully mirrors webmachinelearning/webnn-samples reference:
//   github.com/webmachinelearning/webnn-samples/blob/master/
//     selfie_segmentation/selfie_segmentation_general.js

const WEIGHTS_BASE_URL =
  "https://webmachinelearning.github.io/test-data/models/selfie_segmentation/general";

/**
 * Weight and bias tensor shapes from the official model metadata.
 *
 * Standard convs use filterLayout "ohwi": [outCh, H, W, inCh]
 * Depthwise convs use filterLayout "ihwo": [inCh/groups, H, W, outCh]
 * Bias is always 1D: [outChannels]
 */
const WEIGHT_SHAPES = {
  conv0:  { weight: [16, 3, 3, 3],    bias: [16]  },
  conv1:  { weight: [16, 1, 1, 16],   bias: [16]  },
  conv2:  { weight: [1, 3, 3, 16],    bias: [16]  },  // depthwise
  conv3:  { weight: [8, 1, 1, 16],    bias: [8]   },
  conv4:  { weight: [16, 1, 1, 8],    bias: [16]  },
  conv5:  { weight: [16, 1, 1, 16],   bias: [16]  },
  conv6:  { weight: [72, 1, 1, 16],   bias: [72]  },
  conv7:  { weight: [1, 3, 3, 72],    bias: [72]  },  // depthwise
  conv8:  { weight: [24, 1, 1, 72],   bias: [24]  },
  conv9:  { weight: [88, 1, 1, 24],   bias: [88]  },
  conv10: { weight: [1, 3, 3, 88],    bias: [88]  },  // depthwise
  conv11: { weight: [24, 1, 1, 88],   bias: [24]  },
  conv12: { weight: [96, 1, 1, 24],   bias: [96]  },
  conv13: { weight: [1, 5, 5, 96],    bias: [96]  },  // depthwise
  conv14: { weight: [24, 1, 1, 96],   bias: [24]  },
  conv15: { weight: [96, 1, 1, 24],   bias: [96]  },
  conv16: { weight: [32, 1, 1, 96],   bias: [32]  },
  conv17: { weight: [128, 1, 1, 32],  bias: [128] },
  conv18: { weight: [1, 5, 5, 128],   bias: [128] },  // depthwise
  conv19: { weight: [32, 1, 1, 128],  bias: [32]  },
  conv20: { weight: [128, 1, 1, 32],  bias: [128] },
  conv21: { weight: [32, 1, 1, 128],  bias: [32]  },
  conv22: { weight: [128, 1, 1, 32],  bias: [128] },
  conv23: { weight: [1, 5, 5, 128],   bias: [128] },  // depthwise
  conv24: { weight: [32, 1, 1, 128],  bias: [32]  },
  conv25: { weight: [128, 1, 1, 32],  bias: [128] },
  conv26: { weight: [32, 1, 1, 128],  bias: [32]  },
  conv27: { weight: [96, 1, 1, 32],   bias: [96]  },
  conv28: { weight: [1, 5, 5, 96],    bias: [96]  },  // depthwise
  conv29: { weight: [24, 1, 1, 96],   bias: [24]  },
  conv30: { weight: [96, 1, 1, 24],   bias: [96]  },
  conv31: { weight: [32, 1, 1, 96],   bias: [32]  },
  conv32: { weight: [96, 1, 1, 32],   bias: [96]  },
  conv33: { weight: [1, 5, 5, 96],    bias: [96]  },  // depthwise
  conv34: { weight: [24, 1, 1, 96],   bias: [24]  },
  conv35: { weight: [96, 1, 1, 24],   bias: [96]  },
  conv36: { weight: [32, 1, 1, 96],   bias: [32]  },
  conv37: { weight: [128, 1, 1, 32],  bias: [128] },
  conv38: { weight: [128, 1, 1, 32],  bias: [128] },
  conv39: { weight: [24, 1, 1, 128],  bias: [24]  },
  conv40: { weight: [24, 1, 1, 24],   bias: [24]  },
  conv41: { weight: [24, 1, 1, 24],   bias: [24]  },
  conv42: { weight: [24, 1, 1, 24],   bias: [24]  },
  conv43: { weight: [1, 3, 3, 24],    bias: [24]  },  // depthwise
  conv44: { weight: [16, 1, 1, 24],   bias: [16]  },
  conv45: { weight: [16, 1, 1, 16],   bias: [16]  },
  conv46: { weight: [16, 1, 1, 16],   bias: [16]  },
  conv47: { weight: [16, 1, 1, 16],   bias: [16]  },
  conv48: { weight: [1, 3, 3, 16],    bias: [16]  },  // depthwise
  conv49: { weight: [16, 1, 1, 16],   bias: [16]  },
  conv50: { weight: [16, 1, 1, 16],   bias: [16]  },
  conv51: { weight: [16, 1, 1, 16],   bias: [16]  },
  conv52: { weight: [16, 1, 1, 16],   bias: [16]  },
  conv53: { weight: [1, 3, 3, 16],    bias: [16]  },  // depthwise
  convTranspose0: { weight: [1, 2, 2, 16], bias: null },
};

/**
 * Depthwise convolutions (groups == inputChannels) use filterLayout "ihwo";
 * standard convolutions use "ohwi".
 */
function conv(builder, input, index, activation, options = {}) {
  const shapes = WEIGHT_SHAPES[`conv${index}`];
  const weight = builder.constant({
    shape: shapes.weight, dataType: "float32", label: `conv${index}_weight`,
  });
  const bias = builder.constant({
    shape: shapes.bias, dataType: "float32", label: `conv${index}_bias`,
  });

  const isDepthwise = options.groups > 1;
  const convOut = builder.conv2d(input, weight, {
    ...options,
    bias,
    inputLayout: "nhwc",
    filterLayout: isDepthwise ? "ihwo" : "ohwi",
  });

  if (activation === "relu") return builder.relu(convOut);
  if (activation === "sigmoid") return builder.sigmoid(convOut);
  return convOut;
}

/**
 * SubGraphA — hardswish-like activation fused with convolution:
 *   out = conv(input) * clamp(conv(input) + 3, 0, 6) * (1/6)
 */
function subGraphA(builder, input, convIndex, addB, mulA, convOptions = {}) {
  const c = conv(builder, input, convIndex, "", convOptions);
  const added = builder.add(c, addB);
  const clamped = builder.clamp(added, { minValue: 0, maxValue: 6 });
  const scaled = builder.mul(mulA, clamped);
  return builder.mul(c, scaled);
}

/**
 * SubGraphB — SE-like attention block:
 *   avgPool → conv(relu) → conv(sigmoid) → mul(mulTarget || input)
 */
function subGraphB(builder, input, convIndex, poolStride, mulTarget) {
  const strides = [poolStride, poolStride];
  const pooled = builder.averagePool2d(input, {
    windowDimensions: strides,
    strides,
    layout: "nhwc",
  });
  const reduced = conv(builder, pooled, convIndex, "relu");
  const gate = conv(builder, reduced, convIndex + 1, "sigmoid");
  return builder.mul(mulTarget || input, gate);
}

/**
 * Build the full Selfie Segmentation General graph.
 *
 * Weight constants include correct tensor shapes (from the official model
 * metadata) but no buffer data — they are "weightless". This enables shape
 * inference and QoS estimation before any weight download. Actual weight
 * buffers are attached later via graph.bindConstants().
 *
 * @param {MLGraphBuilder} builder - DAOP or native graph builder
 * @returns {{ graph: Object, weightMeta: null }}
 */
export function buildSelfieSegmentationGraph(builder) {
  const input = builder.input("input", { dataType: "float32", shape: [1, 256, 256, 3] });

  const addB = builder.constant(
    { shape: [1, 1, 1, 1], dataType: "float32", label: null },
    new Float32Array([3]),
  );
  const mulA = builder.constantScalar("float32", 0.1666666716337204);

  // ── Encoder ──────────────────────────────────────────────

  const sgA0 = subGraphA(builder, input, 0, addB, mulA, {
    strides: [2, 2],
    padding: [0, 1, 0, 1],
  });

  const c1 = conv(builder, sgA0, 1, "relu");
  const c2 = conv(builder, c1, 2, "relu", {
    strides: [2, 2],
    padding: [0, 1, 0, 1],
    groups: 16,
  });

  const sgB0 = subGraphB(builder, c2, 3, 64);

  const c5 = conv(builder, sgB0, 5, "");
  const c6 = conv(builder, c5, 6, "relu");
  const c7 = conv(builder, c6, 7, "relu", {
    strides: [2, 2],
    padding: [0, 1, 0, 1],
    groups: 72,
  });
  const c8 = conv(builder, c7, 8, "");

  const c9 = conv(builder, c8, 9, "relu");
  const c10 = conv(builder, c9, 10, "relu", {
    padding: [1, 1, 1, 1],
    groups: 88,
  });
  const c11 = conv(builder, c10, 11, "");
  const add0 = builder.add(c11, c8);

  const sgA1 = subGraphA(builder, add0, 12, addB, mulA);
  const sgA2 = subGraphA(builder, sgA1, 13, addB, mulA, {
    strides: [2, 2],
    padding: [1, 2, 1, 2],
    groups: 96,
  });
  const sgB1 = subGraphB(builder, sgA2, 14, 16);
  const c16 = conv(builder, sgB1, 16, "");

  const sgA3 = subGraphA(builder, c16, 17, addB, mulA);
  const sgA4 = subGraphA(builder, sgA3, 18, addB, mulA, {
    padding: [2, 2, 2, 2],
    groups: 128,
  });
  const sgB2 = subGraphB(builder, sgA4, 19, 16);
  const c21 = conv(builder, sgB2, 21, "");
  const add1 = builder.add(c21, c16);

  const sgA5 = subGraphA(builder, add1, 22, addB, mulA);
  const sgA6 = subGraphA(builder, sgA5, 23, addB, mulA, {
    padding: [2, 2, 2, 2],
    groups: 128,
  });
  const sgB3 = subGraphB(builder, sgA6, 24, 16);
  const c26 = conv(builder, sgB3, 26, "");
  const add2 = builder.add(c26, add1);

  const sgA7 = subGraphA(builder, add2, 27, addB, mulA);
  const sgA8 = subGraphA(builder, sgA7, 28, addB, mulA, {
    padding: [2, 2, 2, 2],
    groups: 96,
  });
  const sgB4 = subGraphB(builder, sgA8, 29, 16);
  const c31 = conv(builder, sgB4, 31, "");
  const add3 = builder.add(c31, add2);

  const sgA9 = subGraphA(builder, add3, 32, addB, mulA);
  const sgA10 = subGraphA(builder, sgA9, 33, addB, mulA, {
    padding: [2, 2, 2, 2],
    groups: 96,
  });
  const sgB5 = subGraphB(builder, sgA10, 34, 16);
  const c36 = conv(builder, sgB5, 36, "");
  const add4 = builder.add(c36, add3);

  // ── Decoder ──────────────────────────────────────────────

  const c37 = conv(builder, add4, 37, "relu");
  const avgPool0 = builder.averagePool2d(add4, {
    windowDimensions: [16, 16],
    strides: [16, 16],
    layout: "nhwc",
  });
  const c38 = conv(builder, avgPool0, 38, "sigmoid");
  const mul0 = builder.mul(c37, c38);

  const resample0 = builder.resample2d(mul0, {
    sizes: [32, 32],
    mode: "linear",
    axes: [1, 2],
  });
  const c39 = conv(builder, resample0, 39, "");
  const add5 = builder.add(c39, add0);

  const sgB6 = subGraphB(builder, add5, 40, 32, add0);
  const add6 = builder.add(sgB6, c39);

  const c42 = conv(builder, add6, 42, "relu");
  const c43 = conv(builder, c42, 43, "relu", {
    padding: [1, 1, 1, 1],
    groups: 24,
  });
  const add7 = builder.add(c42, c43);

  const resample1 = builder.resample2d(add7, {
    sizes: [64, 64],
    mode: "linear",
    axes: [1, 2],
  });
  const c44 = conv(builder, resample1, 44, "");
  const add8 = builder.add(c5, c44);

  const sgB7 = subGraphB(builder, add8, 45, 64, c5);
  const add9 = builder.add(sgB7, c44);

  const c47 = conv(builder, add9, 47, "relu");
  const c48 = conv(builder, c47, 48, "relu", {
    padding: [1, 1, 1, 1],
    groups: 16,
  });
  const add10 = builder.add(c47, c48);

  const resample2 = builder.resample2d(add10, {
    sizes: [128, 128],
    mode: "linear",
    axes: [1, 2],
  });
  const c49 = conv(builder, resample2, 49, "");
  const add11 = builder.add(sgA0, c49);

  const sgB8 = subGraphB(builder, add11, 50, 128, sgA0);
  const add12 = builder.add(sgB8, c49);

  const c52 = conv(builder, add12, 52, "relu");
  const c53 = conv(builder, c52, 53, "relu", {
    padding: [1, 1, 1, 1],
    groups: 16,
  });
  const add13 = builder.add(c52, c53);

  // ── Final: convTranspose2d 128→256 + sigmoid ───────────

  const convTransposeWeight = builder.constant({
    shape: [1, 2, 2, 16], dataType: "float32", label: "convTranspose0_weight",
  });
  const convTransposeBias = builder.constant(
    { shape: [1], dataType: "float32", label: null },
    new Float32Array([0.53271484375]),
  );
  const convTranspose = builder.convTranspose2d(add13, convTransposeWeight, {
    bias: convTransposeBias,
    padding: [0, 0, 0, 0],
    strides: [2, 2],
    outputSizes: [256, 256],
    filterLayout: "ohwi",
    inputLayout: "nhwc",
  });

  const output = builder.sigmoid(convTranspose);

  const graph = builder.build({ output });
  return { graph, weightMeta: null };
}

/**
 * @param {Function} [onProgress] - ({ label, phase }) callback
 * @returns {Promise<Object<string, Float32Array>>}
 */
export async function loadWeights(onProgress) {
  const [weightsInfo, biasesInfo] = await Promise.all([
    fetch(`${WEIGHTS_BASE_URL}/weights_nhwc.json`).then(r => r.json()),
    fetch(`${WEIGHTS_BASE_URL}/biases.json`).then(r => r.json()),
  ]);

  const [weightsBin, biasesBin] = await Promise.all([
    fetch(`${WEIGHTS_BASE_URL}/weights_nhwc.bin`).then(r => r.arrayBuffer()),
    fetch(`${WEIGHTS_BASE_URL}/biases.bin`).then(r => r.arrayBuffer()),
  ]);

  const weights = {};

  for (const [name, meta] of Object.entries(weightsInfo)) {
    const label = `${name}_weight`;
    weights[label] = new Float32Array(weightsBin, meta.dataOffset, meta.byteLength / 4);
    if (onProgress) onProgress({ label, phase: "weight" });
  }

  for (const [name, meta] of Object.entries(biasesInfo)) {
    const label = `${name}_bias`;
    weights[label] = new Float32Array(biasesBin, meta.dataOffset, meta.byteLength / 4);
    if (onProgress) onProgress({ label, phase: "bias" });
  }

  return weights;
}
