// src/qos/interpolation/time-model.js

import { polyFit, polyEval } from "./poly-fit.js";

const STORAGE_KEY = "daop_time_models";
const DEFAULT_POLY_DEGREE = 1;

/**
 * TimeModelDatabase — stores measured time data points per operator
 * and fits polynomial curves for direct time prediction.
 *
 * Stores raw (inputSize -> time) measurements and uses polynomial
 * regression to predict times for unseen sizes.
 */
class TimeModelDatabase {
  constructor() {
    /** @type {Object<string, {points: Array, coeffs: number[]|null}>} */
    this.models = {};
    this._loadFromLocalStorage();
  }

  _loadFromLocalStorage() {
    if (typeof localStorage !== "undefined") {
      const saved = localStorage.getItem(STORAGE_KEY);
      if (saved) {
        try {
          this.models = JSON.parse(saved);
        } catch (e) {
          console.error("[DAOP TimeModel] Failed to parse from localStorage", e);
        }
      }
    }
  }

  _saveToLocalStorage() {
    if (typeof localStorage !== "undefined") {
      localStorage.setItem(STORAGE_KEY, JSON.stringify(this.models));
    }
  }

  /**
   * Add a measured data point for an operator.
   *
   * @param {string} opType - Operator name (e.g., "conv2d")
   * @param {Object} point - { totalElements, medianMs, label, inputShape, ... }
   */
  addDataPoint(opType, point) {
    if (!this.models[opType]) {
      this.models[opType] = { points: [], coeffs: null };
    }
    const model = this.models[opType];
    // Replace existing point at same totalElements
    model.points = model.points.filter(p => p.totalElements !== point.totalElements);
    model.points.push(point);
    model.points.sort((a, b) => a.totalElements - b.totalElements);
    // Invalidate fitted curve
    model.coeffs = null;
    this._saveToLocalStorage();
  }

  /**
   * Fit polynomial curves for all ops that have data points.
   * Call this after all benchmarks are complete.
   *
   * @param {number} degree - Polynomial degree (default 2)
   */
  fitAll(degree = DEFAULT_POLY_DEGREE) {
    for (const [opType, model] of Object.entries(this.models)) {
      if (model.points && model.points.length >= 2) {
        this.fitOp(opType, degree);
      }
    }
    this._saveToLocalStorage();
  }

  /**
   * Fit polynomial for a single op.
   *
   * Uses log-log space: x = log(totalElements), y = log(medianMs).
   *
   * To handle noise at small sizes (where dispatch overhead dominates and
   * can produce U-shaped data), the fitter:
   * 1. Finds the point with the minimum medianMs.
   * 2. Clamps all points to the left of it to that minimum value.
   * 3. Fits a degree-1 polynomial (power law) using only the points from
   *    the minimum onward — the clamped left-side points are excluded.
   * 4. Stores `clampBelowLogX` and `clampLogY` so predict() can return
   *    the flat clamp value for inputs smaller than the minimum point.
   */
  fitOp(opType, degree = DEFAULT_POLY_DEGREE) {
    const model = this.models[opType];
    if (!model || !model.points || model.points.length < 2) return;

    // Find the index of the point with the smallest medianMs
    let minIdx = 0;
    for (let i = 1; i < model.points.length; i++) {
      if (model.points[i].medianMs < model.points[minIdx].medianMs) {
        minIdx = i;
      }
    }

    const minMs = model.points[minIdx].medianMs;
    const clampLogX = Math.log(model.points[minIdx].totalElements);
    const clampLogY = Math.log(Math.max(1e-6, minMs));

    // Clamp left-side points to the minimum value (mutate in place)
    for (let i = 0; i < minIdx; i++) {
      model.points[i].medianMs = minMs;
    }

    // Fit using only points from minIdx onward (right side of the minimum)
    const fitPoints = model.points.slice(minIdx);
    const xs = fitPoints.map(p => Math.log(p.totalElements));
    const ys = fitPoints.map(p => Math.log(Math.max(1e-6, p.medianMs)));

    model.coeffs = (fitPoints.length >= 2) ? polyFit(xs, ys, degree) : null;
    model.clampBelowLogX = (minIdx > 0) ? clampLogX : null;
    model.clampLogY = (minIdx > 0) ? clampLogY : null;
    model.fitDegree = degree;
    model.fittedAt = Date.now();
  }

  /**
   * Predict execution time (ms) for an operator at a given input size.
   *
   * @param {string} opType
   * @param {number} totalElements - Total elements of primary input tensor
   * @returns {number} Predicted time in ms
   */
  predict(opType, totalElements) {
    const model = this.models[opType];
    if (!model) {
      return 0.1;
    }

    if (model.coeffs) {
      const logX = Math.log(Math.max(1, totalElements));

      // Left-side clamp: if input is at or below the minimum-time point,
      // return the clamped floor value instead of extrapolating
      if (model.clampBelowLogX != null && logX <= model.clampBelowLogX) {
        return Math.max(0.001, Math.exp(model.clampLogY));
      }

      const logPredicted = polyEval(model.coeffs, logX);
      return Math.max(0.001, Math.exp(logPredicted));
    }

    return this._linearInterpolate(model.points, totalElements);
  }

  /**
   * Piecewise linear interpolation fallback.
   */
  _linearInterpolate(points, totalElements) {
    if (!points || points.length === 0) return 0.1;
    if (points.length === 1) return points[0].medianMs;

    if (totalElements <= points[0].totalElements) return points[0].medianMs;
    if (totalElements >= points[points.length - 1].totalElements) {
      return points[points.length - 1].medianMs;
    }

    for (let i = 1; i < points.length; i++) {
      if (points[i].totalElements >= totalElements) {
        const lo = points[i - 1];
        const hi = points[i];
        const t = (totalElements - lo.totalElements) / (hi.totalElements - lo.totalElements);
        return lo.medianMs + t * (hi.medianMs - lo.medianMs);
      }
    }
    return points[points.length - 1].medianMs;
  }

  /**
   * Get the model for an op (points + fitted coefficients).
   */
  getModel(opType) {
    return this.models[opType] || null;
  }

  /**
   * Get all models.
   */
  getAllModels() {
    return { ...this.models };
  }

  /**
   * Check if benchmark data exists.
   */
  hasBenchmarkData() {
    return Object.keys(this.models).length > 0 &&
      Object.values(this.models).some(m => m.points && m.points.length > 0);
  }

  /**
   * Reset all models.
   */
  resetToDefaults() {
    this.models = {};
    this._saveToLocalStorage();
  }
}

export const timeModelDatabase = new TimeModelDatabase();
