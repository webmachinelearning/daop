// src/polyfill.js

import { DAOPContext } from "./daop-context.js";
import { DAOPGraphBuilder } from "./daop-graph-builder.js";

/**
 * DAOP Polyfill — intercepts WebNN API to add DAOP extensions.
 *
 * When initialized:
 * 1. Checks for native WebNN (navigator.ml). If missing → returns error.
 * 2. Saves references to native ML/MLGraphBuilder.
 * 3. Replaces window.MLGraphBuilder with DAOPGraphBuilder.
 * 4. Wraps navigator.ml.createContext() to return DAOPContext.
 */

let _initialized = false;
let _nativeML = null;
let _NativeMLGraphBuilder = null;

/**
 * Detect if native WebNN is available.
 *
 * Checks for the presence of navigator.ml and window.MLGraphBuilder.
 * This function only tests the NATIVE API — if DAOP has already been
 * initialized it still returns true (the native references are saved
 * internally).
 */
export function detectWebNNSupport() {
  if (typeof navigator === "undefined") {
    return { supported: false, reason: "No navigator object" };
  }
  if (!navigator.ml) {
    return { supported: false, reason: "navigator.ml not available" };
  }
  if (typeof MLGraphBuilder === "undefined") {
    return { supported: false, reason: "MLGraphBuilder not available" };
  }
  return { supported: true };
}

/**
 * Initialize the DAOP polyfill layer.
 *
 * @returns {{ok: boolean, error?: string}}
 */
export function initDAOP() {
  if (_initialized) {
    return { ok: true };
  }

  const support = detectWebNNSupport();
  if (!support.supported) {
    return {
      ok: false,
      error: `WebNN is not available: ${support.reason}. ` +
        `Please install a WebNN-capable browser: https://webnn.io/en/learn/get-started/installation`,
    };
  }

  // Save native references
  _nativeML = navigator.ml;
  _NativeMLGraphBuilder = window.MLGraphBuilder;

  // Wrap navigator.ml.createContext to return DAOPContext
  const originalCreateContext = _nativeML.createContext.bind(_nativeML);

  const wrappedML = {
    async createContext(options = {}) {
      const nativeContext = await originalCreateContext(options);
      return new DAOPContext(nativeContext, options);
    },
    __daopPolyfill: true,
  };

  // Replace globals
  // navigator.ml is a read-only getter on the Navigator prototype —
  // a plain assignment throws. Use Object.defineProperty to override.
  Object.defineProperty(navigator, "ml", {
    value: wrappedML,
    writable: true,
    configurable: true,
  });
  window.MLGraphBuilder = DAOPGraphBuilder;

  _initialized = true;
  return { ok: true };
}

/**
 * Get a reference to the native (un-wrapped) MLGraphBuilder.
 * Used internally by DAOPGraph.compile() for replay.
 */
export function getNativeMLGraphBuilder() {
  return _NativeMLGraphBuilder || window.MLGraphBuilder;
}

/**
 * Get a reference to the native ML API.
 */
export function getNativeML() {
  return _nativeML || navigator.ml;
}
