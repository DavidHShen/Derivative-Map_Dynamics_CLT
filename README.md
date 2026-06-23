# Derivative-Map Dynamics of Central-Limit Shape Collapse (DMD-CLT)

This repository contains a companion Python implementation for **Derivative-Map Dynamics of Finite- $N$ Central-Limit Shape Collapse**. The code numerically tracks finite- $N$ shape collapse of standardized iid sums through density derivatives, stationary points, inflection points, derivative-map sign words, and Gaussian-skeleton distance diagnostics.

The implementation is designed for reproducible numerical experiments attached to the mathematical framework in `DMD-CLT`.

## Mathematical object

Let $X_1,X_2,\ldots$ be iid copies of a random variable $X$ with mean $\mu$ and variance $\sigma^2>0$. Define the standardized iid sum

$$
Z_N = \frac{X_1+\cdots+X_N-N\mu}{\sigma\sqrt{N}}.
$$

When $Z_N$ has density $f_N$, the diagnostic skeleton is

$$
\mathcal S_N = \{x: f_N'(x)=0\},
\qquad
\mathcal I_N = \{x: f_N''(x)=0\}.
$$

Here $\mathcal S_N$ records stationary points, and $\mathcal I_N$ records inflection points. The standard Gaussian endpoint has

$$
\mathcal S_\infty=\{0\},
\qquad
\mathcal I_\infty=\{-1,1\},
$$

with derivative-map sign word

$$
I \to IV \to III \to II.
$$

## Fourier mechanism

For the standardized sum,

$$
\varphi_{Z_N}(t)=\exp\!\left(-i t\sqrt{N}\mu/\sigma\right)
\varphi_X\!\left(\frac{t}{\sigma\sqrt N}\right)^N.
$$

Fourier inversion reconstructs density derivatives by

$$
f_N^{(k)}(x)=\frac{1}{2\pi}\int_{-\infty}^{\infty}
(-it)^k e^{-itx}\varphi_{Z_N}(t)\,dt,
\qquad k=0,1,2.
$$

The script also implements an exact finite Gaussian-mixture convolution route for the built-in Gaussian-mixture examples. This exact route avoids Fourier-truncation error for those examples and provides a stable benchmark.

## Repository contents

```text
Derivative-Map_Dynamics_CLT.py      Main companion implementation
README.md                           GitHub overview and run instructions
requirements.txt                    Python dependencies
.gitignore                          Generated-output and cache exclusions
examples/run_default.sh             Default execution example
examples/run_fourier_mode.sh        Fourier-quadrature execution example
Derivative-Map_Dynamics_CLT.pdf     Mathematical paper source
```

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

On Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Basic run

```bash
python Derivative-Map_Dynamics_CLT.py
```

Default outputs are written to:

```text
outputs_DMDCLT/
```

## Recommended reproducible run

```bash
python Derivative-Map_Dynamics_CLT.py \
  --out-dir outputs_DMDCLT \
  --n-values 1,2,3,4,5,8,12,20,35,60 \
  --x-max 7.0 \
  --num-x 1401 \
  --central-window 4.5 \
  --density-rel-floor 1e-8 \
  --density-abs-floor 1e-12 \
  --derivative-rel-tol 5e-5 \
  --derivative-abs-tol 1e-9 \
  --method exact-mixture
```

## Optional Fourier mode

```bash
python Derivative-Map_Dynamics_CLT.py \
  --out-dir outputs_DMDCLT_fourier \
  --method fourier \
  --t-max 120 \
  --num-t 6001
```

## Main outputs

The output directory contains:

```text
metrics_*.csv
metrics_all_distributions.csv
SC_numerical_audit.txt
README_outputs.txt
figure1_derivative_quadrants.png
figure2_gaussian_derivative_skeleton.png
figure3_gaussian_transition_points.png
shape_collapse_*.png
derivative_maps_*.png
skeleton_distance_*.png
counts_*.png
```

The CSV files report stationary roots, mode roots, valley roots, inflection roots, sign words, Gaussian $L^2$ errors, tolerance values, density floors, and reconstruction-audit metrics.

## Numerical safeguards

The implementation uses several scientific-computing safeguards:

1. Density-thresholded zero detection avoids artificial stationary or inflection roots in far tails.
2. Bracketed sign changes are used for zero detection instead of accepting every near-zero grid value as a root.
3. A Gaussian reconstruction self-test verifies one stationary point near $0$ and two inflection points near $\pm1$.
4. The audit file records mass error, mean error, variance error, negative density mass, endpoint density and derivative magnitudes, and Fourier-edge indicators.
5. Exact-mixture and Fourier routes are kept separate so that built-in examples have a stable benchmark while preserving the characteristic-function reconstruction path.

All root counts and sign words are finite-window, grid-dependent, and tolerance-dependent diagnostics. They are not asserted as exact global topological counts.

## Built-in examples

The script evaluates three representative laws:

| Example | Description |
|---|---|
| `symmetric_two_gaussian_mixture` | Symmetric two-component Gaussian mixture |
| `asymmetric_three_gaussian_mixture` | Asymmetric three-component Gaussian mixture |
| `smoothed_discrete_multimass` | Narrow Gaussian smoothing of a multi-mass discrete law |

## Workflow

```text
Gaussian-mixture input
        |
        v
standardized iid sum Z_N
        |
        v
exact-mixture or Fourier reconstruction of f_N, f_N', f_N''
        |
        v
density-thresholded zero detection
        |
        v
stationary roots, mode roots, valley roots, inflection roots
        |
        v
sign words, Gaussian-skeleton distance, numerical audit metrics
        |
        v
CSV tables and diagnostic figures
```

## Citation

```text
Shen, David Hongkai. Derivative-Map Dynamics of Finite-N Central-Limit Shape Collapse: Stationary--Inflection Skeletons, Characteristic Functions, and Finite-Window Mode-Death Thresholds. June 2026.
```
