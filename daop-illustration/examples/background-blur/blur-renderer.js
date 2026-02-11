// src/demo/background-blur/blur-renderer.js

/**
 * BlurRenderer — image processing for background blur demo.
 *
 * Handles: image loading → resize → model input prep → mask application → display
 */
export class BlurRenderer {
  /**
   * @param {HTMLCanvasElement} canvas - Display canvas
   */
  constructor(canvas) {
    this.canvas = canvas;
    this.ctx = canvas.getContext("2d");
    this._originalImage = null;
  }

  /**
   * Load an image from URL into the canvas.
   */
  async loadImage(url) {
    return new Promise((resolve, reject) => {
      const img = new Image();
      img.crossOrigin = "anonymous";
      img.onload = () => {
        this._originalImage = img;
        this.canvas.width = img.width;
        this.canvas.height = img.height;
        this.ctx.drawImage(img, 0, 0);
        resolve(img);
      };
      img.onerror = reject;
      img.src = url;
    });
  }

  /**
   * Get the current image as a Float32Array suitable for model input.
   * Resizes to modelWidth × modelHeight and normalizes to [0, 1].
   *
   * @param {number} modelWidth
   * @param {number} modelHeight
   * @param {string} layout - "nchw" or "nhwc"
   * @returns {Float32Array}
   */
  getModelInput(modelWidth, modelHeight, layout = "nchw") {
    if (!this._originalImage) throw new Error("No image loaded");

    // Create offscreen canvas for resize
    const offscreen = document.createElement("canvas");
    offscreen.width = modelWidth;
    offscreen.height = modelHeight;
    const offCtx = offscreen.getContext("2d");
    offCtx.drawImage(this._originalImage, 0, 0, modelWidth, modelHeight);

    const imageData = offCtx.getImageData(0, 0, modelWidth, modelHeight);
    const { data } = imageData; // RGBA uint8

    const size = modelWidth * modelHeight;
    const float32 = new Float32Array(1 * 3 * size);

    if (layout === "nchw") {
      // [1, 3, H, W]
      for (let i = 0; i < size; i++) {
        float32[i] = data[i * 4] / 255.0;             // R
        float32[size + i] = data[i * 4 + 1] / 255.0;  // G
        float32[2 * size + i] = data[i * 4 + 2] / 255.0; // B
      }
    } else {
      // [1, H, W, 3]
      for (let i = 0; i < size; i++) {
        float32[i * 3] = data[i * 4] / 255.0;
        float32[i * 3 + 1] = data[i * 4 + 1] / 255.0;
        float32[i * 3 + 2] = data[i * 4 + 2] / 255.0;
      }
    }

    return float32;
  }

  /**
   * Apply segmentation mask to blur the background.
   *
   * @param {Float32Array} mask - 256x256 segmentation mask (0=bg, 1=fg)
   * @param {number} blurRadius - CSS blur radius in px
   */
  applyBlur(mask, blurRadius = 15) {
    if (!this._originalImage) throw new Error("No image loaded");

    const { width, height } = this.canvas;

    // 1. Draw blurred version
    const blurCanvas = document.createElement("canvas");
    blurCanvas.width = width;
    blurCanvas.height = height;
    const blurCtx = blurCanvas.getContext("2d");
    blurCtx.filter = `blur(${blurRadius}px)`;
    blurCtx.drawImage(this._originalImage, 0, 0, width, height);

    // 2. Scale mask to original image size
    const maskCanvas = document.createElement("canvas");
    maskCanvas.width = 256;
    maskCanvas.height = 256;
    const maskCtx = maskCanvas.getContext("2d");
    const maskImageData = maskCtx.createImageData(256, 256);

    for (let i = 0; i < 256 * 256; i++) {
      const val = Math.round(mask[i] * 255);
      maskImageData.data[i * 4] = val;
      maskImageData.data[i * 4 + 1] = val;
      maskImageData.data[i * 4 + 2] = val;
      maskImageData.data[i * 4 + 3] = 255;
    }
    maskCtx.putImageData(maskImageData, 0, 0);

    // Scale mask to image size
    const scaledMaskCanvas = document.createElement("canvas");
    scaledMaskCanvas.width = width;
    scaledMaskCanvas.height = height;
    const scaledMaskCtx = scaledMaskCanvas.getContext("2d");
    scaledMaskCtx.drawImage(maskCanvas, 0, 0, width, height);

    // 3. Composite: foreground (original) where mask=1, background (blurred) where mask=0
    this.ctx.drawImage(blurCanvas, 0, 0); // Start with blurred

    // Use mask as alpha for original image
    const origCanvas = document.createElement("canvas");
    origCanvas.width = width;
    origCanvas.height = height;
    const origCtx = origCanvas.getContext("2d");
    origCtx.drawImage(this._originalImage, 0, 0, width, height);

    const origData = origCtx.getImageData(0, 0, width, height);
    const scaledMask = scaledMaskCtx.getImageData(0, 0, width, height);

    // Apply mask alpha
    for (let i = 0; i < origData.data.length; i += 4) {
      origData.data[i + 3] = scaledMask.data[i]; // Use R channel of mask as alpha
    }
    origCtx.putImageData(origData, 0, 0);

    this.ctx.drawImage(origCanvas, 0, 0);
  }

  /** Reset to original image */
  reset() {
    if (this._originalImage) {
      this.ctx.drawImage(this._originalImage, 0, 0);
    }
  }
}
