// src/daop/qos/microbench/op-benchmarks.js

function fillRandom(uint8) {
  const CHUNK = 65536;
  for (let offset = 0; offset < uint8.length; offset += CHUNK) {
    const end = Math.min(offset + CHUNK, uint8.length);
    crypto.getRandomValues(uint8.subarray(offset, end));
  }
}

/**
 * Each op benchmark defines multiple size variants (small/medium/large).
 * The bench runner tests all variants and stores per-size timing data,
 * enabling interpolation for accurate estimation on arbitrary shapes.
 *
 * Every buildGraph() must return { outputs, inputName, inputShape, flops,
 * bytes, outputShape, totalElements }.
 */
export const OP_BENCH_CONFIGS = {
  conv2d: {
    opType: "conv2d",
    sizes: [
      {
        label: "xs",
        buildGraph(builder) {
          const inputShape = [1, 16, 16, 8];
          const filterShape = [16, 3, 3, 8];
          const input = builder.input("input", { dataType: "float32", shape: inputShape });
          const filterData = new Float32Array(16 * 3 * 3 * 8);
          fillRandom(new Uint8Array(filterData.buffer));
          const filter = builder.constant({ dataType: "float32", shape: filterShape }, filterData);
          const output = builder.conv2d(input, filter, { inputLayout: "nhwc", filterLayout: "ohwi" });
          const outH = 14, outW = 14;
          const flops = 2 * outH * outW * 16 * 8 * 3 * 3;
          const totalElements = 1 * 16 * 16 * 8;
          const bytes = (1*16*16*8 + 16*3*3*8 + 1*outH*outW*16) * 4;
          return { outputs: { output }, inputName: "input", inputShape, flops, bytes, outputShape: [1, outH, outW, 16], totalElements };
        },
      },
      {
        label: "small",
        buildGraph(builder) {
          const inputShape = [1, 32, 32, 16];
          const filterShape = [32, 3, 3, 16];
          const input = builder.input("input", { dataType: "float32", shape: inputShape });
          const filterData = new Float32Array(32 * 3 * 3 * 16);
          fillRandom(new Uint8Array(filterData.buffer));
          const filter = builder.constant({ dataType: "float32", shape: filterShape }, filterData);
          const output = builder.conv2d(input, filter, { inputLayout: "nhwc", filterLayout: "ohwi" });
          const outH = 30, outW = 30;
          const flops = 2 * outH * outW * 32 * 16 * 3 * 3;
          const totalElements = 1 * 32 * 32 * 16;
          const bytes = (1*32*32*16 + 32*3*3*16 + 1*outH*outW*32) * 4;
          return { outputs: { output }, inputName: "input", inputShape, flops, bytes, outputShape: [1, outH, outW, 32], totalElements };
        },
      },
      {
        label: "medium",
        buildGraph(builder) {
          const inputShape = [1, 64, 64, 24];
          const filterShape = [48, 3, 3, 24];
          const input = builder.input("input", { dataType: "float32", shape: inputShape });
          const filterData = new Float32Array(48 * 3 * 3 * 24);
          fillRandom(new Uint8Array(filterData.buffer));
          const filter = builder.constant({ dataType: "float32", shape: filterShape }, filterData);
          const output = builder.conv2d(input, filter, { inputLayout: "nhwc", filterLayout: "ohwi" });
          const outH = 62, outW = 62;
          const flops = 2 * outH * outW * 48 * 24 * 3 * 3;
          const totalElements = 1 * 64 * 64 * 24;
          const bytes = (1*64*64*24 + 48*3*3*24 + 1*outH*outW*48) * 4;
          return { outputs: { output }, inputName: "input", inputShape, flops, bytes, outputShape: [1, outH, outW, 48], totalElements };
        },
      },
      {
        label: "large",
        buildGraph(builder) {
          const inputShape = [1, 128, 128, 32];
          const filterShape = [64, 3, 3, 32];
          const input = builder.input("input", { dataType: "float32", shape: inputShape });
          const filterData = new Float32Array(64 * 3 * 3 * 32);
          fillRandom(new Uint8Array(filterData.buffer));
          const filter = builder.constant({ dataType: "float32", shape: filterShape }, filterData);
          const output = builder.conv2d(input, filter, { inputLayout: "nhwc", filterLayout: "ohwi" });
          const outH = 126, outW = 126;
          const flops = 2 * outH * outW * 64 * 32 * 3 * 3;
          const totalElements = 1 * 128 * 128 * 32;
          const bytes = (1*128*128*32 + 64*3*3*32 + 1*outH*outW*64) * 4;
          return { outputs: { output }, inputName: "input", inputShape, flops, bytes, outputShape: [1, outH, outW, 64], totalElements };
        },
      },
      {
        label: "xl",
        buildGraph(builder) {
          const inputShape = [1, 256, 256, 32];
          const filterShape = [64, 3, 3, 32];
          const input = builder.input("input", { dataType: "float32", shape: inputShape });
          const filterData = new Float32Array(64 * 3 * 3 * 32);
          fillRandom(new Uint8Array(filterData.buffer));
          const filter = builder.constant({ dataType: "float32", shape: filterShape }, filterData);
          const output = builder.conv2d(input, filter, { inputLayout: "nhwc", filterLayout: "ohwi" });
          const outH = 254, outW = 254;
          const flops = 2 * outH * outW * 64 * 32 * 3 * 3;
          const totalElements = 1 * 256 * 256 * 32;
          const bytes = (1*256*256*32 + 64*3*3*32 + 1*outH*outW*64) * 4;
          return { outputs: { output }, inputName: "input", inputShape, flops, bytes, outputShape: [1, outH, outW, 64], totalElements };
        },
      },
      {
        label: "xxl",
        buildGraph(builder) {
          const inputShape = [1, 512, 512, 32];
          const filterShape = [64, 3, 3, 32];
          const input = builder.input("input", { dataType: "float32", shape: inputShape });
          const filterData = new Float32Array(64 * 3 * 3 * 32);
          fillRandom(new Uint8Array(filterData.buffer));
          const filter = builder.constant({ dataType: "float32", shape: filterShape }, filterData);
          const output = builder.conv2d(input, filter, { inputLayout: "nhwc", filterLayout: "ohwi" });
          const outH = 510, outW = 510;
          const flops = 2 * outH * outW * 64 * 32 * 3 * 3;
          const totalElements = 1 * 512 * 512 * 32;
          const bytes = (1*512*512*32 + 64*3*3*32 + 1*outH*outW*64) * 4;
          return { outputs: { output }, inputName: "input", inputShape, flops, bytes, outputShape: [1, outH, outW, 64], totalElements };
        },
      },
    ],
  },

  add: {
    opType: "add",
    sizes: [
      {
        label: "xs",
        buildGraph(builder) {
          const shape = [1, 16, 16, 8];
          const a = builder.input("input", { dataType: "float32", shape });
          const bData = new Float32Array(1 * 16 * 16 * 8);
          fillRandom(new Uint8Array(bData.buffer));
          const b = builder.constant({ dataType: "float32", shape }, bData);
          const output = builder.add(a, b);
          const elements = 1 * 16 * 16 * 8;
          return { outputs: { output }, inputName: "input", inputShape: shape, flops: elements, bytes: elements * 3 * 4, outputShape: shape, totalElements: elements };
        },
      },
      {
        label: "small",
        buildGraph(builder) {
          const shape = [1, 32, 32, 16];
          const a = builder.input("input", { dataType: "float32", shape });
          const bData = new Float32Array(1 * 32 * 32 * 16);
          fillRandom(new Uint8Array(bData.buffer));
          const b = builder.constant({ dataType: "float32", shape }, bData);
          const output = builder.add(a, b);
          const elements = 1 * 32 * 32 * 16;
          return { outputs: { output }, inputName: "input", inputShape: shape, flops: elements, bytes: elements * 3 * 4, outputShape: shape, totalElements: elements };
        },
      },
      {
        label: "medium",
        buildGraph(builder) {
          const shape = [1, 64, 64, 32];
          const a = builder.input("input", { dataType: "float32", shape });
          const bData = new Float32Array(1 * 64 * 64 * 32);
          fillRandom(new Uint8Array(bData.buffer));
          const b = builder.constant({ dataType: "float32", shape }, bData);
          const output = builder.add(a, b);
          const elements = 1 * 64 * 64 * 32;
          return { outputs: { output }, inputName: "input", inputShape: shape, flops: elements, bytes: elements * 3 * 4, outputShape: shape, totalElements: elements };
        },
      },
      {
        label: "large",
        buildGraph(builder) {
          const shape = [1, 128, 128, 64];
          const a = builder.input("input", { dataType: "float32", shape });
          const bData = new Float32Array(1 * 128 * 128 * 64);
          fillRandom(new Uint8Array(bData.buffer));
          const b = builder.constant({ dataType: "float32", shape }, bData);
          const output = builder.add(a, b);
          const elements = 1 * 128 * 128 * 64;
          return { outputs: { output }, inputName: "input", inputShape: shape, flops: elements, bytes: elements * 3 * 4, outputShape: shape, totalElements: elements };
        },
      },
      {
        label: "xl",
        buildGraph(builder) {
          const shape = [1, 256, 256, 32];
          const a = builder.input("input", { dataType: "float32", shape });
          const bData = new Float32Array(1 * 256 * 256 * 32);
          fillRandom(new Uint8Array(bData.buffer));
          const b = builder.constant({ dataType: "float32", shape }, bData);
          const output = builder.add(a, b);
          const elements = 1 * 256 * 256 * 32;
          return { outputs: { output }, inputName: "input", inputShape: shape, flops: elements, bytes: elements * 3 * 4, outputShape: shape, totalElements: elements };
        },
      },
      {
        label: "xxl",
        buildGraph(builder) {
          const shape = [1, 512, 512, 32];
          const a = builder.input("input", { dataType: "float32", shape });
          const bData = new Float32Array(1 * 512 * 512 * 32);
          fillRandom(new Uint8Array(bData.buffer));
          const b = builder.constant({ dataType: "float32", shape }, bData);
          const output = builder.add(a, b);
          const elements = 1 * 512 * 512 * 32;
          return { outputs: { output }, inputName: "input", inputShape: shape, flops: elements, bytes: elements * 3 * 4, outputShape: shape, totalElements: elements };
        },
      },
    ],
  },

  mul: {
    opType: "mul",
    sizes: [
      {
        label: "xs",
        buildGraph(builder) {
          const shape = [1, 16, 16, 8];
          const a = builder.input("input", { dataType: "float32", shape });
          const bData = new Float32Array(1 * 16 * 16 * 8);
          fillRandom(new Uint8Array(bData.buffer));
          const b = builder.constant({ dataType: "float32", shape }, bData);
          const output = builder.mul(a, b);
          const elements = 1 * 16 * 16 * 8;
          return { outputs: { output }, inputName: "input", inputShape: shape, flops: elements, bytes: elements * 3 * 4, outputShape: shape, totalElements: elements };
        },
      },
      {
        label: "small",
        buildGraph(builder) {
          const shape = [1, 32, 32, 16];
          const a = builder.input("input", { dataType: "float32", shape });
          const bData = new Float32Array(1 * 32 * 32 * 16);
          fillRandom(new Uint8Array(bData.buffer));
          const b = builder.constant({ dataType: "float32", shape }, bData);
          const output = builder.mul(a, b);
          const elements = 1 * 32 * 32 * 16;
          return { outputs: { output }, inputName: "input", inputShape: shape, flops: elements, bytes: elements * 3 * 4, outputShape: shape, totalElements: elements };
        },
      },
      {
        label: "medium",
        buildGraph(builder) {
          const shape = [1, 64, 64, 32];
          const a = builder.input("input", { dataType: "float32", shape });
          const bData = new Float32Array(1 * 64 * 64 * 32);
          fillRandom(new Uint8Array(bData.buffer));
          const b = builder.constant({ dataType: "float32", shape }, bData);
          const output = builder.mul(a, b);
          const elements = 1 * 64 * 64 * 32;
          return { outputs: { output }, inputName: "input", inputShape: shape, flops: elements, bytes: elements * 3 * 4, outputShape: shape, totalElements: elements };
        },
      },
      {
        label: "large",
        buildGraph(builder) {
          const shape = [1, 128, 128, 64];
          const a = builder.input("input", { dataType: "float32", shape });
          const bData = new Float32Array(1 * 128 * 128 * 64);
          fillRandom(new Uint8Array(bData.buffer));
          const b = builder.constant({ dataType: "float32", shape }, bData);
          const output = builder.mul(a, b);
          const elements = 1 * 128 * 128 * 64;
          return { outputs: { output }, inputName: "input", inputShape: shape, flops: elements, bytes: elements * 3 * 4, outputShape: shape, totalElements: elements };
        },
      },
      {
        label: "xl",
        buildGraph(builder) {
          const shape = [1, 256, 256, 32];
          const a = builder.input("input", { dataType: "float32", shape });
          const bData = new Float32Array(1 * 256 * 256 * 32);
          fillRandom(new Uint8Array(bData.buffer));
          const b = builder.constant({ dataType: "float32", shape }, bData);
          const output = builder.mul(a, b);
          const elements = 1 * 256 * 256 * 32;
          return { outputs: { output }, inputName: "input", inputShape: shape, flops: elements, bytes: elements * 3 * 4, outputShape: shape, totalElements: elements };
        },
      },
      {
        label: "xxl",
        buildGraph(builder) {
          const shape = [1, 512, 512, 32];
          const a = builder.input("input", { dataType: "float32", shape });
          const bData = new Float32Array(1 * 512 * 512 * 32);
          fillRandom(new Uint8Array(bData.buffer));
          const b = builder.constant({ dataType: "float32", shape }, bData);
          const output = builder.mul(a, b);
          const elements = 1 * 512 * 512 * 32;
          return { outputs: { output }, inputName: "input", inputShape: shape, flops: elements, bytes: elements * 3 * 4, outputShape: shape, totalElements: elements };
        },
      },
    ],
  },

  relu: {
    opType: "relu",
    sizes: [
      {
        label: "xs",
        buildGraph(builder) {
          const shape = [1, 16, 16, 8];
          const input = builder.input("input", { dataType: "float32", shape });
          const output = builder.relu(input);
          const elements = 1 * 16 * 16 * 8;
          return { outputs: { output }, inputName: "input", inputShape: shape, flops: elements, bytes: elements * 2 * 4, outputShape: shape, totalElements: elements };
        },
      },
      {
        label: "small",
        buildGraph(builder) {
          const shape = [1, 32, 32, 16];
          const input = builder.input("input", { dataType: "float32", shape });
          const output = builder.relu(input);
          const elements = 1 * 32 * 32 * 16;
          return { outputs: { output }, inputName: "input", inputShape: shape, flops: elements, bytes: elements * 2 * 4, outputShape: shape, totalElements: elements };
        },
      },
      {
        label: "medium",
        buildGraph(builder) {
          const shape = [1, 64, 64, 32];
          const input = builder.input("input", { dataType: "float32", shape });
          const output = builder.relu(input);
          const elements = 1 * 64 * 64 * 32;
          return { outputs: { output }, inputName: "input", inputShape: shape, flops: elements, bytes: elements * 2 * 4, outputShape: shape, totalElements: elements };
        },
      },
      {
        label: "large",
        buildGraph(builder) {
          const shape = [1, 128, 128, 64];
          const input = builder.input("input", { dataType: "float32", shape });
          const output = builder.relu(input);
          const elements = 1 * 128 * 128 * 64;
          return { outputs: { output }, inputName: "input", inputShape: shape, flops: elements, bytes: elements * 2 * 4, outputShape: shape, totalElements: elements };
        },
      },
      {
        label: "xl",
        buildGraph(builder) {
          const shape = [1, 256, 256, 32];
          const input = builder.input("input", { dataType: "float32", shape });
          const output = builder.relu(input);
          const elements = 1 * 256 * 256 * 32;
          return { outputs: { output }, inputName: "input", inputShape: shape, flops: elements, bytes: elements * 2 * 4, outputShape: shape, totalElements: elements };
        },
      },
      {
        label: "xxl",
        buildGraph(builder) {
          const shape = [1, 512, 512, 32];
          const input = builder.input("input", { dataType: "float32", shape });
          const output = builder.relu(input);
          const elements = 1 * 512 * 512 * 32;
          return { outputs: { output }, inputName: "input", inputShape: shape, flops: elements, bytes: elements * 2 * 4, outputShape: shape, totalElements: elements };
        },
      },
    ],
  },

  sigmoid: {
    opType: "sigmoid",
    sizes: [
      {
        label: "xs",
        buildGraph(builder) {
          const shape = [1, 16, 16, 8];
          const input = builder.input("input", { dataType: "float32", shape });
          const output = builder.sigmoid(input);
          const elements = 1 * 16 * 16 * 8;
          return { outputs: { output }, inputName: "input", inputShape: shape, flops: elements * 4, bytes: elements * 2 * 4, outputShape: shape, totalElements: elements };
        },
      },
      {
        label: "small",
        buildGraph(builder) {
          const shape = [1, 32, 32, 16];
          const input = builder.input("input", { dataType: "float32", shape });
          const output = builder.sigmoid(input);
          const elements = 1 * 32 * 32 * 16;
          return { outputs: { output }, inputName: "input", inputShape: shape, flops: elements * 4, bytes: elements * 2 * 4, outputShape: shape, totalElements: elements };
        },
      },
      {
        label: "medium",
        buildGraph(builder) {
          const shape = [1, 64, 64, 32];
          const input = builder.input("input", { dataType: "float32", shape });
          const output = builder.sigmoid(input);
          const elements = 1 * 64 * 64 * 32;
          return { outputs: { output }, inputName: "input", inputShape: shape, flops: elements * 4, bytes: elements * 2 * 4, outputShape: shape, totalElements: elements };
        },
      },
      {
        label: "large",
        buildGraph(builder) {
          const shape = [1, 128, 128, 64];
          const input = builder.input("input", { dataType: "float32", shape });
          const output = builder.sigmoid(input);
          const elements = 1 * 128 * 128 * 64;
          return { outputs: { output }, inputName: "input", inputShape: shape, flops: elements * 4, bytes: elements * 2 * 4, outputShape: shape, totalElements: elements };
        },
      },
      {
        label: "xl",
        buildGraph(builder) {
          const shape = [1, 256, 256, 32];
          const input = builder.input("input", { dataType: "float32", shape });
          const output = builder.sigmoid(input);
          const elements = 1 * 256 * 256 * 32;
          return { outputs: { output }, inputName: "input", inputShape: shape, flops: elements * 4, bytes: elements * 2 * 4, outputShape: shape, totalElements: elements };
        },
      },
      {
        label: "xxl",
        buildGraph(builder) {
          const shape = [1, 512, 512, 32];
          const input = builder.input("input", { dataType: "float32", shape });
          const output = builder.sigmoid(input);
          const elements = 1 * 512 * 512 * 32;
          return { outputs: { output }, inputName: "input", inputShape: shape, flops: elements * 4, bytes: elements * 2 * 4, outputShape: shape, totalElements: elements };
        },
      },
    ],
  },

  clamp: {
    opType: "clamp",
    sizes: [
      {
        label: "xs",
        buildGraph(builder) {
          const shape = [1, 16, 16, 8];
          const input = builder.input("input", { dataType: "float32", shape });
          const output = builder.clamp(input, { minValue: 0, maxValue: 6 });
          const elements = 1 * 16 * 16 * 8;
          return { outputs: { output }, inputName: "input", inputShape: shape, flops: elements * 2, bytes: elements * 2 * 4, outputShape: shape, totalElements: elements };
        },
      },
      {
        label: "small",
        buildGraph(builder) {
          const shape = [1, 32, 32, 16];
          const input = builder.input("input", { dataType: "float32", shape });
          const output = builder.clamp(input, { minValue: 0, maxValue: 6 });
          const elements = 1 * 32 * 32 * 16;
          return { outputs: { output }, inputName: "input", inputShape: shape, flops: elements * 2, bytes: elements * 2 * 4, outputShape: shape, totalElements: elements };
        },
      },
      {
        label: "medium",
        buildGraph(builder) {
          const shape = [1, 64, 64, 32];
          const input = builder.input("input", { dataType: "float32", shape });
          const output = builder.clamp(input, { minValue: 0, maxValue: 6 });
          const elements = 1 * 64 * 64 * 32;
          return { outputs: { output }, inputName: "input", inputShape: shape, flops: elements * 2, bytes: elements * 2 * 4, outputShape: shape, totalElements: elements };
        },
      },
      {
        label: "large",
        buildGraph(builder) {
          const shape = [1, 128, 128, 64];
          const input = builder.input("input", { dataType: "float32", shape });
          const output = builder.clamp(input, { minValue: 0, maxValue: 6 });
          const elements = 1 * 128 * 128 * 64;
          return { outputs: { output }, inputName: "input", inputShape: shape, flops: elements * 2, bytes: elements * 2 * 4, outputShape: shape, totalElements: elements };
        },
      },
      {
        label: "xl",
        buildGraph(builder) {
          const shape = [1, 256, 256, 32];
          const input = builder.input("input", { dataType: "float32", shape });
          const output = builder.clamp(input, { minValue: 0, maxValue: 6 });
          const elements = 1 * 256 * 256 * 32;
          return { outputs: { output }, inputName: "input", inputShape: shape, flops: elements * 2, bytes: elements * 2 * 4, outputShape: shape, totalElements: elements };
        },
      },
      {
        label: "xxl",
        buildGraph(builder) {
          const shape = [1, 512, 512, 32];
          const input = builder.input("input", { dataType: "float32", shape });
          const output = builder.clamp(input, { minValue: 0, maxValue: 6 });
          const elements = 1 * 512 * 512 * 32;
          return { outputs: { output }, inputName: "input", inputShape: shape, flops: elements * 2, bytes: elements * 2 * 4, outputShape: shape, totalElements: elements };
        },
      },
    ],
  },

  averagePool2d: {
    opType: "averagePool2d",
    sizes: [
      {
        label: "xs",
        buildGraph(builder) {
          const shape = [1, 16, 16, 8];
          const input = builder.input("input", { dataType: "float32", shape });
          const output = builder.averagePool2d(input, {
            windowDimensions: [3, 3], strides: [2, 2], padding: [1, 1, 1, 1], layout: "nhwc",
          });
          const outElements = 1 * 8 * 8 * 8;
          const inElements = 1 * 16 * 16 * 8;
          return { outputs: { output }, inputName: "input", inputShape: shape, flops: outElements * 3 * 3, bytes: (inElements + outElements) * 4, outputShape: [1, 8, 8, 8], totalElements: inElements };
        },
      },
      {
        label: "small",
        buildGraph(builder) {
          const shape = [1, 32, 32, 16];
          const input = builder.input("input", { dataType: "float32", shape });
          const output = builder.averagePool2d(input, {
            windowDimensions: [3, 3], strides: [2, 2], padding: [1, 1, 1, 1], layout: "nhwc",
          });
          const outElements = 1 * 16 * 16 * 16;
          const inElements = 1 * 32 * 32 * 16;
          return { outputs: { output }, inputName: "input", inputShape: shape, flops: outElements * 3 * 3, bytes: (inElements + outElements) * 4, outputShape: [1, 16, 16, 16], totalElements: inElements };
        },
      },
      {
        label: "medium",
        buildGraph(builder) {
          const shape = [1, 64, 64, 32];
          const input = builder.input("input", { dataType: "float32", shape });
          const output = builder.averagePool2d(input, {
            windowDimensions: [3, 3], strides: [2, 2], padding: [1, 1, 1, 1], layout: "nhwc",
          });
          const outElements = 1 * 32 * 32 * 32;
          const inElements = 1 * 64 * 64 * 32;
          return { outputs: { output }, inputName: "input", inputShape: shape, flops: outElements * 3 * 3, bytes: (inElements + outElements) * 4, outputShape: [1, 32, 32, 32], totalElements: inElements };
        },
      },
      {
        label: "large",
        buildGraph(builder) {
          const shape = [1, 128, 128, 64];
          const input = builder.input("input", { dataType: "float32", shape });
          const output = builder.averagePool2d(input, {
            windowDimensions: [3, 3], strides: [2, 2], padding: [1, 1, 1, 1], layout: "nhwc",
          });
          const outElements = 1 * 64 * 64 * 64;
          const inElements = 1 * 128 * 128 * 64;
          return { outputs: { output }, inputName: "input", inputShape: shape, flops: outElements * 3 * 3, bytes: (inElements + outElements) * 4, outputShape: [1, 64, 64, 64], totalElements: inElements };
        },
      },
      {
        label: "xl",
        buildGraph(builder) {
          const shape = [1, 256, 256, 32];
          const input = builder.input("input", { dataType: "float32", shape });
          const output = builder.averagePool2d(input, {
            windowDimensions: [3, 3], strides: [2, 2], padding: [1, 1, 1, 1], layout: "nhwc",
          });
          const outElements = 1 * 128 * 128 * 32;
          const inElements = 1 * 256 * 256 * 32;
          return { outputs: { output }, inputName: "input", inputShape: shape, flops: outElements * 3 * 3, bytes: (inElements + outElements) * 4, outputShape: [1, 128, 128, 32], totalElements: inElements };
        },
      },
      {
        label: "xxl",
        buildGraph(builder) {
          const shape = [1, 512, 512, 32];
          const input = builder.input("input", { dataType: "float32", shape });
          const output = builder.averagePool2d(input, {
            windowDimensions: [3, 3], strides: [2, 2], padding: [1, 1, 1, 1], layout: "nhwc",
          });
          const outElements = 1 * 256 * 256 * 32;
          const inElements = 1 * 512 * 512 * 32;
          return { outputs: { output }, inputName: "input", inputShape: shape, flops: outElements * 3 * 3, bytes: (inElements + outElements) * 4, outputShape: [1, 256, 256, 32], totalElements: inElements };
        },
      },
    ],
  },
};
