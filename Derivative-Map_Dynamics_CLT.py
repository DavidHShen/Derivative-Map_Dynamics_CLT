#!/usr/bin/env python3
"""
DMD-CLT companion implementation:
Derivative-Map Dynamics of Finite-N Central-Limit Shape Collapse

This script numerically tracks the finite-N shape collapse of standardized iid sums
through density derivatives, stationary points, inflection points, sign words, and
simple Gaussian-skeleton distance metrics.

Main outputs
------------
outputs_DMDCLT/
  metrics_*.csv
  figure1_derivative_quadrants.png
  figure2_gaussian_derivative_skeleton.png
  figure3_gaussian_transition_points.png
  shape_collapse_*.png
  derivative_maps_*.png
  skeleton_distance_*.png

Requirements
------------
Python >= 3.9
numpy, pandas, matplotlib

No SciPy is required. The script uses density-thresholded zero detection,
numerical audit metrics, and convergence/self-test outputs for stricter
scientific-computing use.

"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

Array = np.ndarray


# -----------------------------------------------------------------------------
# Distribution model: finite Gaussian mixture before standardization.
# -----------------------------------------------------------------------------

@dataclass(frozen=True)
class GaussianMixture1D:
    """A one-dimensional Gaussian mixture law for X.

    X ~ sum_j weights[j] * N(means[j], variances[j]).

    The standardized iid sum is
        Z_N = (X_1 + ... + X_N - N mu) / (sigma sqrt(N)).

    Its characteristic function is
        phi_ZN(t) = exp(-i t sqrt(N) mu/sigma) * phi_X(t/(sigma sqrt(N)))^N.
    """

    name: str
    weights: Array
    means: Array
    variances: Array

    def __post_init__(self) -> None:
        w = np.asarray(self.weights, dtype=float)
        m = np.asarray(self.means, dtype=float)
        v = np.asarray(self.variances, dtype=float)
        if not (len(w) == len(m) == len(v)):
            raise ValueError("weights, means, and variances must have the same length")
        if np.any(w < 0):
            raise ValueError("mixture weights must be nonnegative")
        if not np.isclose(w.sum(), 1.0):
            object.__setattr__(self, "weights", w / w.sum())
        if np.any(v <= 0):
            raise ValueError("Gaussian component variances must be positive")
        object.__setattr__(self, "weights", np.asarray(self.weights, dtype=float))
        object.__setattr__(self, "means", m)
        object.__setattr__(self, "variances", v)

    @property
    def mean(self) -> float:
        return float(np.sum(self.weights * self.means))

    @property
    def variance(self) -> float:
        mu = self.mean
        second = np.sum(self.weights * (self.variances + self.means**2))
        return float(second - mu**2)

    @property
    def std(self) -> float:
        return math.sqrt(self.variance)

    def phi_x(self, u: Array) -> Array:
        """Characteristic function of X at real frequencies u."""
        u = np.asarray(u, dtype=float)
        terms = []
        for w, m, v in zip(self.weights, self.means, self.variances):
            terms.append(w * np.exp(1j * m * u - 0.5 * v * u**2))
        return np.sum(np.vstack(terms), axis=0)

    def phi_standardized_sum(self, t: Array, n: int) -> Array:
        """Characteristic function of Z_N."""
        if n < 1:
            raise ValueError("n must be positive")
        sigma = self.std
        mu = self.mean
        u = t / (sigma * math.sqrt(n))
        return np.exp(-1j * t * math.sqrt(n) * mu / sigma) * self.phi_x(u) ** n



# -----------------------------------------------------------------------------
# Exact Gaussian-mixture convolution for the built-in examples.
# -----------------------------------------------------------------------------

def integer_compositions(total: int, parts: int) -> List[Tuple[int, ...]]:
    """Yield all nonnegative integer tuples of length ``parts`` summing to total."""
    if parts == 1:
        return [(total,)]
    out: List[Tuple[int, ...]] = []
    for first in range(total + 1):
        for rest in integer_compositions(total - first, parts - 1):
            out.append((first,) + rest)
    return out


def exact_mixture_standardized_sum_derivatives(
    law: GaussianMixture1D,
    x_grid: Array,
    n: int,
    max_components: int = 100000,
) -> Tuple[Array, Array, Array]:
    """Compute f, f', f'' exactly for a finite Gaussian-mixture convolution.

    For a k-component Gaussian mixture, the N-fold convolution is a Gaussian
    mixture over multinomial count vectors.  This is much faster and more
    accurate than Fourier quadrature for the built-in examples and provides a
    strong numerical benchmark for the Fourier implementation.
    """
    k = len(law.weights)
    counts = integer_compositions(n, k)
    if len(counts) > max_components:
        raise RuntimeError(
            f"Exact mixture expansion would create {len(counts)} components; "
            f"increase max_components or use --method fourier."
        )
    x = np.asarray(x_grid, dtype=float)
    f = np.zeros_like(x)
    f1 = np.zeros_like(x)
    f2 = np.zeros_like(x)
    base_mu = law.mean
    base_sigma = law.std
    log_fact_n = math.lgamma(n + 1)
    log_weights = np.log(law.weights)
    for c_tuple in counts:
        c = np.asarray(c_tuple, dtype=float)
        log_w = log_fact_n - sum(math.lgamma(int(ci) + 1) for ci in c_tuple) + float(np.dot(c, log_weights))
        # Skip components below double-precision relevance.
        if log_w < -745:
            continue
        w = math.exp(log_w)
        mean_sum = float(np.dot(c, law.means))
        var_sum = float(np.dot(c, law.variances))
        mean_z = (mean_sum - n * base_mu) / (base_sigma * math.sqrt(n))
        var_z = var_sum / (base_sigma**2 * n)
        sd_z = math.sqrt(var_z)
        u = (x - mean_z) / sd_z
        g = np.exp(-0.5 * u**2) / (math.sqrt(2.0 * math.pi) * sd_z)
        g1 = -((x - mean_z) / var_z) * g
        g2 = (((x - mean_z) ** 2) / (var_z**2) - 1.0 / var_z) * g
        f += w * g
        f1 += w * g1
        f2 += w * g2
    return f, f1, f2

# -----------------------------------------------------------------------------
# Fourier inversion and derivative reconstruction.
# -----------------------------------------------------------------------------

def fourier_density_derivatives(
    phi: Callable[[Array], Array],
    x_grid: Array,
    t_max: float = 120.0,
    num_t: int = 6001,
) -> Tuple[Array, Array, Array]:
    """Compute f, f', f'' from characteristic function by direct Fourier inversion.

    Convention:
        phi(t) = E exp(i t X),
        f(x) = (1 / 2pi) integral exp(-i t x) phi(t) dt.

    Then:
        f'(x)  = (1 / 2pi) integral (-i t) exp(-i t x) phi(t) dt,
        f''(x) = (1 / 2pi) integral (-i t)^2 exp(-i t x) phi(t) dt.

    Direct quadrature is slower than FFT but transparent and stable for the
    moderate grids used in the paper's demonstration.
    """
    if num_t % 2 == 0:
        num_t += 1

    t = np.linspace(-t_max, t_max, num_t)
    dt = t[1] - t[0]
    ph = phi(t)

    # Trapezoid weights.
    weights = np.ones_like(t)
    weights[0] = 0.5
    weights[-1] = 0.5
    base = ph * weights * dt / (2.0 * math.pi)

    # Evaluate in x-chunks to avoid storing one very large complex matrix.
    b0 = base
    b1 = (-1j * t) * base
    b2 = ((-1j * t) ** 2) * base
    f = np.empty_like(x_grid, dtype=float)
    f1 = np.empty_like(x_grid, dtype=float)
    f2 = np.empty_like(x_grid, dtype=float)
    chunk_size = 256
    for start in range(0, len(x_grid), chunk_size):
        stop = min(start + chunk_size, len(x_grid))
        kernel = np.exp(-1j * np.outer(x_grid[start:stop], t))
        f[start:stop] = np.real(kernel @ b0)
        f1[start:stop] = np.real(kernel @ b1)
        f2[start:stop] = np.real(kernel @ b2)

    # Tiny numerical density values can occur from truncation.
    f[np.abs(f) < 1e-12] = 0.0
    return f, f1, f2


def normal_pdf(x: Array) -> Array:
    return np.exp(-0.5 * x**2) / math.sqrt(2.0 * math.pi)


def normal_pdf_derivatives(x: Array) -> Tuple[Array, Array, Array]:
    f = normal_pdf(x)
    f1 = -x * f
    f2 = (x**2 - 1.0) * f
    return f, f1, f2


# -----------------------------------------------------------------------------
# Zero detection, sign words, and metrics.
# -----------------------------------------------------------------------------

def interp_values(x: Array, y: Array, points: Array) -> Array:
    if len(points) == 0:
        return np.array([], dtype=float)
    return np.interp(points, x, y)


def contiguous_true_segments(mask: Array) -> List[Tuple[int, int]]:
    """Return half-open index segments on which a Boolean mask is true."""
    mask = np.asarray(mask, dtype=bool)
    segments: List[Tuple[int, int]] = []
    i = 0
    n = len(mask)
    while i < n:
        while i < n and not mask[i]:
            i += 1
        if i >= n:
            break
        j = i + 1
        while j < n and mask[j]:
            j += 1
        if j - i >= 2:
            segments.append((i, j))
        i = j
    return segments


def zero_crossings(
    x: Array,
    y: Array,
    tol: float = 1e-9,
    valid: Optional[Array] = None,
    min_separation: Optional[float] = None,
) -> Array:
    """Locate zeros by bracketed sign changes on numerically significant regions.

    Near-zero derivative values in far tails can create artificial roots near the
    edge of the analysis window. This routine uses a validity mask, normally
    generated from a density threshold, and accepts near-zero grid points only
    when adjacent values bracket a sign change inside the same significant-density
    segment.
    """

    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if valid is None:
        valid_mask = np.isfinite(x) & np.isfinite(y)
    else:
        valid_mask = np.asarray(valid, dtype=bool) & np.isfinite(x) & np.isfinite(y)

    roots: List[float] = []
    dx = float(np.median(np.diff(x))) if len(x) > 1 else 0.0
    if min_separation is None:
        min_separation = max(2.0 * dx, 1e-12)

    for start, stop in contiguous_true_segments(valid_mask):
        xs = x[start:stop]
        ys = y[start:stop]
        if len(xs) < 2:
            continue

        # Sign-change interpolation, with near-zero endpoint handling.
        for i in range(len(xs) - 1):
            y0 = ys[i]
            y1 = ys[i + 1]
            if not np.isfinite(y0) or not np.isfinite(y1):
                continue
            if abs(y0) <= tol and abs(y1) <= tol:
                # Flat near-zero patches are not reliable roots for derivative maps.
                continue
            if abs(y0) <= tol:
                if i > 0 and np.sign(ys[i - 1]) * np.sign(y1) < 0:
                    roots.append(float(xs[i]))
                continue
            if abs(y1) <= tol:
                if i + 2 < len(ys) and np.sign(y0) * np.sign(ys[i + 2]) < 0:
                    roots.append(float(xs[i + 1]))
                continue
            if y0 * y1 < 0:
                root = xs[i] - y0 * (xs[i + 1] - xs[i]) / (y1 - y0)
                roots.append(float(root))

    if not roots:
        return np.array([], dtype=float)
    roots_sorted = sorted(roots)

    # Deduplicate nearby roots.
    dedup = [roots_sorted[0]]
    for r in roots_sorted[1:]:
        if abs(r - dedup[-1]) > min_separation:
            dedup.append(r)
        else:
            dedup[-1] = 0.5 * (dedup[-1] + r)
    return np.array(dedup, dtype=float)

def classify_stationary_points(
    x: Array,
    f1: Array,
    f2: Array,
    roots_f1: Array,
    curvature_tol: float = 1e-5,
) -> Tuple[Array, Array, Array]:
    """Return maxima, minima, and flat/ambiguous stationary roots."""
    curv = interp_values(x, f2, roots_f1)
    maxima = roots_f1[curv < -curvature_tol]
    minima = roots_f1[curv > curvature_tol]
    ambiguous = roots_f1[np.abs(curv) <= curvature_tol]
    return maxima, minima, ambiguous


def quadrant_label(f1_val: float, f2_val: float, eps: float = 1e-10) -> str:
    s1 = 1 if f1_val >= -eps else -1
    s2 = 1 if f2_val >= -eps else -1
    if s1 > 0 and s2 > 0:
        return "I(+,+)"
    if s1 > 0 and s2 < 0:
        return "IV(+,-)"
    if s1 < 0 and s2 < 0:
        return "III(-,-)"
    return "II(-,+)"


def sign_word(
    x: Array,
    f1: Array,
    f2: Array,
    roots_f1: Array,
    roots_f2: Array,
    window: Tuple[float, float] = (-4.0, 4.0),
) -> str:
    """Build a coarse derivative-map sign word over a finite window."""
    lo, hi = window
    breakpoints = [lo]
    breakpoints.extend(float(r) for r in roots_f1 if lo < r < hi)
    breakpoints.extend(float(r) for r in roots_f2 if lo < r < hi)
    breakpoints.append(hi)
    pts = sorted(set(round(b, 8) for b in breakpoints))

    labels: List[str] = []
    for a, b in zip(pts[:-1], pts[1:]):
        if b - a <= 1e-8:
            continue
        mid = 0.5 * (a + b)
        f1_mid = float(np.interp(mid, x, f1))
        f2_mid = float(np.interp(mid, x, f2))
        lab = quadrant_label(f1_mid, f2_mid)
        if not labels or labels[-1] != lab:
            labels.append(lab)
    return " -> ".join(labels)


def skeleton_distance(
    stationary_roots: Array,
    inflection_roots: Array,
    extra_penalty: float = 1.0,
) -> float:
    """Simple distance from Gaussian skeleton {0}, {-1, +1}.

    The metric is intentionally diagnostic, not canonical:
      nearest stationary-root distance to 0
      + nearest inflection-root distances to -1 and +1
      + penalty for extra stationary/inflection roots.
    """
    dist = 0.0
    if len(stationary_roots) == 0:
        dist += extra_penalty
    else:
        dist += float(np.min(np.abs(stationary_roots - 0.0)))
        dist += extra_penalty * max(0, len(stationary_roots) - 1)

    for target in (-1.0, 1.0):
        if len(inflection_roots) == 0:
            dist += extra_penalty
        else:
            dist += float(np.min(np.abs(inflection_roots - target)))
    dist += extra_penalty * max(0, len(inflection_roots) - 2)
    return dist



def trapz(y: Array, x: Array) -> float:
    """Version-stable trapezoidal integral wrapper."""
    if hasattr(np, "trapezoid"):
        return float(np.trapezoid(y, x))
    return float(np.trapz(y, x))



def gaussian_l2_errors(x: Array, f: Array, f1: Array, f2: Array) -> Tuple[float, float, float]:
    g, g1, g2 = normal_pdf_derivatives(x)
    return (
        float(np.sqrt(trapz((f - g) ** 2, x))),
        float(np.sqrt(trapz((f1 - g1) ** 2, x))),
        float(np.sqrt(trapz((f2 - g2) ** 2, x))),
    )


def reconstruction_audit(x: Array, f: Array, f1: Array, f2: Array) -> Dict[str, float]:
    """Basic numerical-integrity checks for reconstructed densities."""
    f_pos = np.maximum(f, 0.0)
    mass = trapz(f, x)
    positive_mass = trapz(f_pos, x)
    negative_mass = trapz(np.maximum(-f, 0.0), x)
    mean = trapz(x * f, x)
    second = trapz((x**2) * f, x)
    variance = second - mean**2
    return {
        "mass": mass,
        "positive_mass": positive_mass,
        "negative_mass": negative_mass,
        "mean_numerical": mean,
        "variance_numerical": variance,
        "mass_error": abs(mass - 1.0),
        "mean_error": abs(mean),
        "variance_error": abs(variance - 1.0),
        "min_density": float(np.min(f)),
        "max_density": float(np.max(f)),
        "endpoint_abs_density_max": float(max(abs(f[0]), abs(f[-1]))),
        "endpoint_abs_first_derivative_max": float(max(abs(f1[0]), abs(f1[-1]))),
        "endpoint_abs_second_derivative_max": float(max(abs(f2[0]), abs(f2[-1]))),
    }


def fourier_tail_indicators(phi: Callable[[Array], Array], t_max: float, num_t: int) -> Dict[str, float]:
    """Report high-frequency truncation indicators near the Fourier boundary."""
    if num_t % 2 == 0:
        num_t += 1
    t = np.linspace(-t_max, t_max, num_t)
    ph = phi(t)
    edge = max(10, int(0.02 * len(t)))
    edge_mask = np.zeros_like(t, dtype=bool)
    edge_mask[:edge] = True
    edge_mask[-edge:] = True
    out: Dict[str, float] = {}
    for k in (0, 1, 2):
        out[f"fourier_edge_max_order{k}"] = float(np.max((np.abs(t) ** k) * np.abs(ph) * edge_mask))
    return out


# -----------------------------------------------------------------------------
# Diagnostics and plotting.
# -----------------------------------------------------------------------------

def compute_diagnostics(
    law: GaussianMixture1D,
    n_values: Sequence[int],
    x_grid: Array,
    t_max: float,
    num_t: int,
    central_window: Tuple[float, float],
    density_rel_floor: float = 1e-8,
    density_abs_floor: float = 1e-12,
    derivative_rel_tol: float = 5e-5,
    derivative_abs_tol: float = 1e-9,
    method: str = "exact-mixture",
) -> Tuple[pd.DataFrame, Dict[int, Dict[str, Array]]]:
    rows = []
    curves: Dict[int, Dict[str, Array]] = {}

    lo, hi = central_window
    mask = (x_grid >= lo) & (x_grid <= hi)
    xw = x_grid[mask]

    for n in n_values:
        phi_n = lambda t, n=n: law.phi_standardized_sum(t, n)
        if method == "exact-mixture":
            f, f1, f2 = exact_mixture_standardized_sum_derivatives(law, x_grid, n)
            fourier_audit = {
                "fourier_edge_max_order0": np.nan,
                "fourier_edge_max_order1": np.nan,
                "fourier_edge_max_order2": np.nan,
            }
        elif method == "fourier":
            f, f1, f2 = fourier_density_derivatives(
                phi_n,
                x_grid,
                t_max=t_max,
                num_t=num_t,
            )
            fourier_audit = fourier_tail_indicators(phi_n, t_max=t_max, num_t=num_t)
        else:
            raise ValueError("method must be 'exact-mixture' or 'fourier'")
        fw, f1w, f2w = f[mask], f1[mask], f2[mask]

        # Scientific-computing guardrail: count derivative zeros only where the
        # reconstructed density is numerically meaningful.  This implements the
        # tail-threshold instruction in the paper and avoids fake tail roots.
        max_fw = float(np.nanmax(np.maximum(fw, 0.0)))
        density_floor = max(density_abs_floor, density_rel_floor * max_fw)
        valid = np.isfinite(fw) & np.isfinite(f1w) & np.isfinite(f2w) & (fw >= density_floor)
        if np.count_nonzero(valid) < 3:
            valid = np.isfinite(fw) & np.isfinite(f1w) & np.isfinite(f2w)

        # Adaptive tolerances based on derivative magnitudes within the valid region.
        f1_scale = float(np.nanmax(np.abs(f1w[valid]))) if np.any(valid) else float(np.nanmax(np.abs(f1w)))
        f2_scale = float(np.nanmax(np.abs(f2w[valid]))) if np.any(valid) else float(np.nanmax(np.abs(f2w)))
        f1_tol = max(derivative_abs_tol, derivative_rel_tol * f1_scale)
        f2_tol = max(derivative_abs_tol, derivative_rel_tol * f2_scale)

        dx = float(np.median(np.diff(xw)))
        roots_f1 = zero_crossings(xw, f1w, tol=f1_tol, valid=valid, min_separation=2.5 * dx)
        roots_f2 = zero_crossings(xw, f2w, tol=f2_tol, valid=valid, min_separation=2.5 * dx)
        maxima, minima, ambiguous = classify_stationary_points(xw, f1w, f2w, roots_f1)

        if np.any(valid):
            x_valid = xw[valid]
            word_window = (float(x_valid[0]), float(x_valid[-1]))
        else:
            word_window = central_window
        word = sign_word(xw, f1w, f2w, roots_f1, roots_f2, window=word_window)
        d_skel = skeleton_distance(roots_f1, roots_f2)
        e0, e1, e2 = gaussian_l2_errors(xw, fw, f1w, f2w)
        audit = reconstruction_audit(x_grid, f, f1, f2)

        rows.append(
            {
                "distribution": law.name,
                "N": n,
                "num_stationary": len(roots_f1),
                "num_modes": len(maxima),
                "num_valleys": len(minima),
                "num_ambiguous_stationary": len(ambiguous),
                "num_inflections": len(roots_f2),
                "stationary_roots": format_roots(roots_f1),
                "mode_roots": format_roots(maxima),
                "valley_roots": format_roots(minima),
                "inflection_roots": format_roots(roots_f2),
                "skeleton_distance": d_skel,
                "L2_density_to_gaussian": e0,
                "L2_first_derivative_to_gaussian": e1,
                "L2_second_derivative_to_gaussian": e2,
                "density_floor_used": density_floor,
                "valid_grid_points": int(np.count_nonzero(valid)),
                "zero_tol_first_derivative": f1_tol,
                "zero_tol_second_derivative": f2_tol,
                "sign_word_window_lo": word_window[0],
                "sign_word_window_hi": word_window[1],
                **audit,
                **fourier_audit,
                "sign_word": word,
            }
        )
        curves[n] = {
            "x": x_grid,
            "f": f,
            "f1": f1,
            "f2": f2,
            "roots_f1": roots_f1,
            "roots_f2": roots_f2,
            "maxima": maxima,
            "minima": minima,
            "density_floor": np.array([density_floor]),
        }
    return pd.DataFrame(rows), curves

def format_roots(roots: Array) -> str:
    if len(roots) == 0:
        return ""
    return ";".join(f"{r:.5f}" for r in roots)


def make_intro_figures(out_dir: Path) -> None:
    """Generate the three motivating figures referenced in DMD-CLT."""
    out_dir.mkdir(parents=True, exist_ok=True)

    # Figure 1: derivative-map quadrants as a simple plane partition.
    fig, ax = plt.subplots(figsize=(6.0, 4.0))
    ax.axhline(0.0, linewidth=1.0)
    ax.axvline(0.0, linewidth=1.0)
    ax.text(0.55, 0.55, "I\n$f'>0, f''>0$", ha="center", va="center", fontsize=11)
    ax.text(0.55, -0.55, "IV\n$f'>0, f''<0$", ha="center", va="center", fontsize=11)
    ax.text(-0.55, -0.55, "III\n$f'<0, f''<0$", ha="center", va="center", fontsize=11)
    ax.text(-0.55, 0.55, "II\n$f'<0, f''>0$", ha="center", va="center", fontsize=11)
    ax.set_xlabel("first derivative sign $f'$")
    ax.set_ylabel("second derivative sign $f''$")
    ax.set_xlim(-1.0, 1.0)
    ax.set_ylim(-1.0, 1.0)
    ax.set_title("Derivative-map quadrants")
    fig.tight_layout()
    fig.savefig(out_dir / "figure1_derivative_quadrants.png", dpi=200)
    plt.close(fig)

    # Figure 2: Gaussian f, f', f''.
    x = np.linspace(-4.0, 4.0, 1000)
    g, g1, g2 = normal_pdf_derivatives(x)
    fig, ax = plt.subplots(figsize=(6.0, 4.0))
    ax.plot(x, g, label=r"$\phi$")
    ax.plot(x, g1, label=r"$\phi'$")
    ax.plot(x, g2, label=r"$\phi''$")
    ax.axhline(0.0, linewidth=0.8)
    ax.axvline(-1.0, linestyle="--", linewidth=0.8)
    ax.axvline(0.0, linestyle="--", linewidth=0.8)
    ax.axvline(1.0, linestyle="--", linewidth=0.8)
    ax.set_title("Gaussian derivative skeleton")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "figure2_gaussian_derivative_skeleton.png", dpi=200)
    plt.close(fig)

    # Figure 3: transition points on Gaussian density.
    fig, ax = plt.subplots(figsize=(6.0, 4.0))
    ax.plot(x, g, label=r"$\phi$")
    for xx in (-1.0, 0.0, 1.0):
        ax.axvline(xx, linestyle="--", linewidth=0.8, color="0.82", zorder=0)
    for xx, lab, offset in [
        (-1.0, "$-1$ inflection", (10, 10)),
        (0.0, "$0$ stationary", (10, -18)),
        (1.0, "$+1$ inflection", (10, 10)),
    ]:
        yy = float(normal_pdf(np.array([xx]))[0])
        ax.scatter([xx], [yy], s=35, zorder=3)
        ax.annotate(lab, (xx, yy), xytext=offset, textcoords="offset points")
    ax.text(-3.1, 0.03, "I", fontsize=12)
    ax.text(-0.62, 0.215, "IV", fontsize=12)
    ax.text(0.38, 0.215, "III", fontsize=12)
    ax.text(2.5, 0.03, "II", fontsize=12)
    ax.set_title("Gaussian density transition points")
    fig.tight_layout()
    fig.savefig(out_dir / "figure3_gaussian_transition_points.png", dpi=200)
    plt.close(fig)


def plot_shape_collapse(
    law: GaussianMixture1D,
    curves: Dict[int, Dict[str, Array]],
    selected_n: Sequence[int],
    out_dir: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(8.0, 5.0))
    for n in selected_n:
        if n not in curves:
            continue
        x = curves[n]["x"]
        f = curves[n]["f"]
        ax.plot(x, f, label=f"N={n}")
    x = next(iter(curves.values()))["x"]
    ax.plot(
        x,
        normal_pdf(x),
        linestyle="--",
        color="black",
        linewidth=1.6,
        label="Gaussian",
        zorder=10,
    )
    ax.set_xlim(-5.0, 5.0)
    ax.set_title(f"Finite-N density shape collapse: {law.name}")
    ax.set_xlabel("standardized coordinate x")
    ax.set_ylabel("density")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / f"shape_collapse_{safe_name(law.name)}.png", dpi=200)
    plt.close(fig)


def plot_derivative_maps(
    law: GaussianMixture1D,
    curves: Dict[int, Dict[str, Array]],
    selected_n: Sequence[int],
    out_dir: Path,
) -> None:
    for n in selected_n:
        if n not in curves:
            continue
        d = curves[n]
        x, f, f1, f2 = d["x"], d["f"], d["f1"], d["f2"]
        fig, ax = plt.subplots(figsize=(8.0, 5.0))
        ax.plot(x, f, label="$f_N$")
        ax.plot(x, f1, label="$f_N'$")
        ax.plot(x, f2, label="$f_N''$")
        ax.axhline(0.0, linewidth=0.8)
        for r in d["roots_f1"]:
            ax.axvline(r, linestyle="--", linewidth=0.7)
        for r in d["roots_f2"]:
            ax.axvline(r, linestyle=":", linewidth=0.7)
        ax.set_xlim(-5.0, 5.0)
        ax.set_title(f"Derivative map, {law.name}, N={n}")
        ax.set_xlabel("standardized coordinate x")
        ax.legend()
        fig.tight_layout()
        fig.savefig(out_dir / f"derivative_maps_{safe_name(law.name)}_N{n}.png", dpi=200)
        plt.close(fig)


def plot_metrics(df: pd.DataFrame, law_name: str, out_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.0, 4.5))
    ax.plot(df["N"], df["skeleton_distance"], marker="o")
    ax.set_xscale("log")
    ax.set_xlabel("N")
    ax.set_ylabel("diagnostic distance")
    ax.set_title(f"Gaussian-skeleton distance: {law_name}")
    fig.tight_layout()
    fig.savefig(out_dir / f"skeleton_distance_{safe_name(law_name)}.png", dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.0, 4.5))
    ax.plot(df["N"], df["num_modes"], marker="o", label="modes")
    ax.plot(df["N"], df["num_inflections"], marker="s", label="inflections")
    ax.set_xscale("log")
    ax.set_xlabel("N")
    ax.set_ylabel("count on central window")
    ax.set_title(f"Mode and inflection counts: {law_name}")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / f"counts_{safe_name(law_name)}.png", dpi=200)
    plt.close(fig)


def safe_name(name: str) -> str:
    return "".join(ch.lower() if ch.isalnum() else "_" for ch in name).strip("_")


# -----------------------------------------------------------------------------
# Example distributions aligned with DMD-CLT's numerical section.
# -----------------------------------------------------------------------------

def example_distributions() -> List[GaussianMixture1D]:
    return [
        GaussianMixture1D(
            name="symmetric_two_gaussian_mixture",
            weights=np.array([0.5, 0.5]),
            means=np.array([-2.0, 2.0]),
            variances=np.array([0.16, 0.16]),
        ),
        GaussianMixture1D(
            name="asymmetric_three_gaussian_mixture",
            weights=np.array([0.55, 0.30, 0.15]),
            means=np.array([-1.6, 0.5, 3.0]),
            variances=np.array([0.12, 0.25, 0.20]),
        ),
        GaussianMixture1D(
            name="smoothed_discrete_multimass",
            weights=np.array([0.20, 0.35, 0.30, 0.15]),
            means=np.array([-3.0, -0.8, 1.2, 3.5]),
            variances=np.array([0.04, 0.04, 0.04, 0.04]),
        ),
    ]


# -----------------------------------------------------------------------------
# Scientific-computing self-tests.
# -----------------------------------------------------------------------------

def run_gaussian_self_test(
    x_grid: Array,
    t_max: float,
    num_t: int,
    central_window: Tuple[float, float],
    density_rel_floor: float,
    density_abs_floor: float,
    derivative_rel_tol: float,
    derivative_abs_tol: float,
) -> Dict[str, object]:
    """Reconstruct the standard normal from its characteristic function.

    This is a useful regression test because the exact density derivatives and
    zero skeleton are known: one stationary point at 0 and inflections at +-1.
    """
    phi = lambda t: np.exp(-0.5 * np.asarray(t, dtype=float) ** 2)
    f, f1, f2 = fourier_density_derivatives(phi, x_grid, t_max=t_max, num_t=num_t)
    lo, hi = central_window
    mask = (x_grid >= lo) & (x_grid <= hi)
    xw, fw, f1w, f2w = x_grid[mask], f[mask], f1[mask], f2[mask]
    density_floor = max(density_abs_floor, density_rel_floor * float(np.max(np.maximum(fw, 0.0))))
    valid = fw >= density_floor
    f1_tol = max(derivative_abs_tol, derivative_rel_tol * float(np.max(np.abs(f1w[valid]))))
    f2_tol = max(derivative_abs_tol, derivative_rel_tol * float(np.max(np.abs(f2w[valid]))))
    dx = float(np.median(np.diff(xw)))
    roots_f1 = zero_crossings(xw, f1w, tol=f1_tol, valid=valid, min_separation=2.5 * dx)
    roots_f2 = zero_crossings(xw, f2w, tol=f2_tol, valid=valid, min_separation=2.5 * dx)
    e0, e1, e2 = gaussian_l2_errors(xw, fw, f1w, f2w)
    audit = reconstruction_audit(x_grid, f, f1, f2)
    status = (
        len(roots_f1) == 1
        and len(roots_f2) == 2
        and (abs(roots_f1[0]) < 5.0 * dx if len(roots_f1) else False)
        and (np.max(np.abs(np.sort(roots_f2) - np.array([-1.0, 1.0]))) < 5.0 * dx if len(roots_f2) == 2 else False)
        and audit["mass_error"] < 5e-4
    )
    return {
        "status": "PASS" if status else "CHECK",
        "stationary_roots": format_roots(roots_f1),
        "inflection_roots": format_roots(roots_f2),
        "L2_density_to_exact_gaussian": e0,
        "L2_first_derivative_to_exact_gaussian": e1,
        "L2_second_derivative_to_exact_gaussian": e2,
        **audit,
    }


def write_audit_report(out_dir: Path, self_test: Dict[str, object], combined: pd.DataFrame) -> None:
    """Write a human-readable numerical audit report."""
    lines: List[str] = []
    lines.append("DMD-CLT scientific-computing audit")
    lines.append("")
    lines.append("Gaussian reconstruction self-test")
    for k, v in self_test.items():
        lines.append(f"  {k}: {v}")
    lines.append("")
    lines.append("Worst reconstruction diagnostics over reported runs")
    for col in [
        "mass_error",
        "mean_error",
        "variance_error",
        "negative_mass",
        "endpoint_abs_density_max",
        "endpoint_abs_first_derivative_max",
        "endpoint_abs_second_derivative_max",
        "fourier_edge_max_order0",
        "fourier_edge_max_order1",
        "fourier_edge_max_order2",
    ]:
        if col in combined.columns:
            lines.append(f"  max {col}: {combined[col].max():.6e}")
    lines.append("")
    lines.append(
        "Interpretation: zero counts are density-window and tolerance dependent. "
        "Large Fourier-edge indicators, mass errors, or negative density mass should "
        "trigger a finer grid, larger t_max, or a more conservative analysis window."
    )
    (out_dir / "SC_numerical_audit.txt").write_text("\n".join(lines), encoding="utf-8")


# -----------------------------------------------------------------------------
# Main command-line interface.
# -----------------------------------------------------------------------------

def parse_n_values(text: str) -> List[int]:
    vals = []
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        vals.append(int(part))
    vals = sorted(set(vals))
    if not vals or vals[0] < 1:
        raise ValueError("N values must be positive integers")
    return vals


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Derivative-map diagnostics for finite-N central-limit shape collapse."
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("outputs_DMDCLT"),
        help="Output directory for figures and CSV metrics.",
    )
    parser.add_argument(
        "--n-values",
        type=str,
        default="1,2,3,4,5,8,12,20,35,60",
        help="Comma-separated N values.",
    )
    parser.add_argument(
        "--x-max",
        type=float,
        default=7.0,
        help="Use x grid [-x-max, x-max].",
    )
    parser.add_argument(
        "--num-x",
        type=int,
        default=1401,
        help="Number of x-grid points.",
    )
    parser.add_argument(
        "--t-max",
        type=float,
        default=120.0,
        help="Fourier integration range [-t-max, t-max].",
    )
    parser.add_argument(
        "--num-t",
        type=int,
        default=6001,
        help="Number of Fourier grid points. Odd value recommended.",
    )
    parser.add_argument(
        "--central-window",
        type=float,
        default=4.5,
        help="Diagnostics are counted on [-central-window, central-window].",
    )
    parser.add_argument(
        "--density-rel-floor",
        type=float,
        default=1e-8,
        help="Relative density floor for derivative-zero counting; prevents tail roots.",
    )
    parser.add_argument(
        "--density-abs-floor",
        type=float,
        default=1e-12,
        help="Absolute density floor for derivative-zero counting.",
    )
    parser.add_argument(
        "--derivative-rel-tol",
        type=float,
        default=5e-5,
        help="Relative derivative tolerance for zero bracketing on the valid density window.",
    )
    parser.add_argument(
        "--derivative-abs-tol",
        type=float,
        default=1e-9,
        help="Absolute derivative tolerance for zero bracketing.",
    )
    parser.add_argument(
        "--method",
        choices=["exact-mixture", "fourier"],
        default="exact-mixture",
        help="Use exact Gaussian-mixture convolution for built-in examples, or direct Fourier quadrature.",
    )
    parser.add_argument(
        "--skip-self-test",
        action="store_true",
        help="Skip the Gaussian reconstruction self-test.",
    )
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    make_intro_figures(args.out_dir)

    n_values = parse_n_values(args.n_values)
    x_grid = np.linspace(-args.x_max, args.x_max, args.num_x)
    central_window = (-args.central_window, args.central_window)
    selected_n = [n for n in [1, 2, 3, 5, 12, 35, 60] if n in n_values]
    if not selected_n:
        selected_n = n_values[: min(5, len(n_values))]

    all_metrics = []
    for law in example_distributions():
        print(f"Running diagnostics for {law.name}...")
        print(f"  mean={law.mean:.6f}, std={law.std:.6f}")
        df, curves = compute_diagnostics(
            law=law,
            n_values=n_values,
            x_grid=x_grid,
            t_max=args.t_max,
            num_t=args.num_t,
            central_window=central_window,
            density_rel_floor=args.density_rel_floor,
            density_abs_floor=args.density_abs_floor,
            derivative_rel_tol=args.derivative_rel_tol,
            derivative_abs_tol=args.derivative_abs_tol,
            method=args.method,
        )
        csv_path = args.out_dir / f"metrics_{safe_name(law.name)}.csv"
        df.to_csv(csv_path, index=False)
        all_metrics.append(df)

        plot_shape_collapse(law, curves, selected_n, args.out_dir)
        plot_derivative_maps(law, curves, selected_n, args.out_dir)
        plot_metrics(df, law.name, args.out_dir)

    combined = pd.concat(all_metrics, ignore_index=True)
    combined.to_csv(args.out_dir / "metrics_all_distributions.csv", index=False)

    if not args.skip_self_test:
        self_test = run_gaussian_self_test(
            x_grid=x_grid,
            t_max=args.t_max,
            num_t=args.num_t,
            central_window=central_window,
            density_rel_floor=args.density_rel_floor,
            density_abs_floor=args.density_abs_floor,
            derivative_rel_tol=args.derivative_rel_tol,
            derivative_abs_tol=args.derivative_abs_tol,
        )
    else:
        self_test = {"status": "SKIPPED"}
    write_audit_report(args.out_dir, self_test, combined)

    readme = args.out_dir / "README_outputs.txt"
    readme.write_text(
        "Derivative-map CLT diagnostics outputs.\n\n"
        "The metrics CSV files report stationary roots, mode roots, valley roots, "
        "inflection roots, derivative-map sign words, Gaussian L2 errors, and a simple "
        "Gaussian-skeleton distance. The PNG figures visualize the motivating Gaussian "
        "skeleton, finite-N density collapse, derivative maps, and diagnostic counts. "
        "SC_numerical_audit.txt reports reconstruction self-tests and numerical-integrity checks.\n",
        encoding="utf-8",
    )
    print(f"Done. Outputs written to: {args.out_dir.resolve()}")


if __name__ == "__main__":
    main()
