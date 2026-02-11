// src/qos/interpolation/poly-fit.js

/**
 * Polynomial least-squares regression.
 *
 * Fits y = c0 + c1*x + c2*x^2 + ... + cn*x^n to minimize squared error.
 * Uses the normal equations approach (sufficient for degree 2-3 with <20 points).
 */

/**
 * Fit a polynomial of given degree to (x, y) data points.
 *
 * @param {number[]} xs - Independent variable values
 * @param {number[]} ys - Dependent variable values
 * @param {number} degree - Polynomial degree (2 or 3 recommended)
 * @returns {number[]} Coefficients [c0, c1, c2, ...] where y = c0 + c1*x + c2*x^2 + ...
 */
export function polyFit(xs, ys, degree = 2) {
  if (xs.length !== ys.length) {
    throw new Error("polyFit: xs and ys must have same length");
  }
  if (xs.length <= degree) {
    // Not enough points for this degree — fall back to lower degree
    degree = Math.max(1, xs.length - 1);
  }

  const n = xs.length;
  const m = degree + 1; // number of coefficients

  // Build normal equations: A^T A c = A^T y
  // ATA[i][j] = sum(x^(i+j)), ATy[i] = sum(y * x^i)
  const ATA = Array.from({ length: m }, () => new Array(m).fill(0));
  const ATy = new Array(m).fill(0);

  // Pre-compute x^p for each data point (p = 0..2*degree)
  for (let k = 0; k < n; k++) {
    const xPows = new Array(2 * degree + 1);
    xPows[0] = 1;
    for (let p = 1; p < xPows.length; p++) {
      xPows[p] = xPows[p - 1] * xs[k];
    }
    for (let i = 0; i < m; i++) {
      ATy[i] += ys[k] * xPows[i];
      for (let j = i; j < m; j++) {
        ATA[i][j] += xPows[i + j];
      }
    }
  }

  // Fill symmetric lower triangle
  for (let i = 1; i < m; i++) {
    for (let j = 0; j < i; j++) {
      ATA[i][j] = ATA[j][i];
    }
  }

  // Solve via Gaussian elimination with partial pivoting
  return _solveLinearSystem(ATA, ATy);
}

/**
 * Evaluate a polynomial at a given x.
 *
 * @param {number[]} coeffs - [c0, c1, c2, ...] from polyFit
 * @param {number} x - Value to evaluate at
 * @returns {number} y = c0 + c1*x + c2*x^2 + ...
 */
export function polyEval(coeffs, x) {
  let result = 0;
  let xPow = 1;
  for (const c of coeffs) {
    result += c * xPow;
    xPow *= x;
  }
  return result;
}

/**
 * Solve Ax = b using Gaussian elimination with partial pivoting.
 * Modifies A and b in place.
 *
 * @param {number[][]} A - Square matrix
 * @param {number[]} b - Right-hand side
 * @returns {number[]} Solution vector x
 */
function _solveLinearSystem(A, b) {
  const n = A.length;

  // Forward elimination with partial pivoting
  for (let col = 0; col < n; col++) {
    // Find pivot
    let maxVal = Math.abs(A[col][col]);
    let maxRow = col;
    for (let row = col + 1; row < n; row++) {
      if (Math.abs(A[row][col]) > maxVal) {
        maxVal = Math.abs(A[row][col]);
        maxRow = row;
      }
    }

    // Swap rows
    if (maxRow !== col) {
      [A[col], A[maxRow]] = [A[maxRow], A[col]];
      [b[col], b[maxRow]] = [b[maxRow], b[col]];
    }

    // Eliminate
    const pivot = A[col][col];
    if (Math.abs(pivot) < 1e-12) continue; // singular — skip
    for (let row = col + 1; row < n; row++) {
      const factor = A[row][col] / pivot;
      for (let j = col; j < n; j++) {
        A[row][j] -= factor * A[col][j];
      }
      b[row] -= factor * b[col];
    }
  }

  // Back substitution
  const x = new Array(n).fill(0);
  for (let row = n - 1; row >= 0; row--) {
    let sum = b[row];
    for (let j = row + 1; j < n; j++) {
      sum -= A[row][j] * x[j];
    }
    x[row] = Math.abs(A[row][row]) > 1e-12 ? sum / A[row][row] : 0;
  }

  return x;
}
