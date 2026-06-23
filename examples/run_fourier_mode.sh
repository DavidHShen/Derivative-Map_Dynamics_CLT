#!/usr/bin/env bash
set -euo pipefail

python ../Derivative-Map_Dynamics_CLT.py \
  --out-dir ../outputs_DMDCLT_fourier \
  --method fourier \
  --t-max 120 \
  --num-t 6001
