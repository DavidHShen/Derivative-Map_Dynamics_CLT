#!/usr/bin/env bash
set -euo pipefail

python ../Derivative-Map_Dynamics_CLT.py \
  --out-dir ../outputs_DMDCLT \
  --n-values 1,2,3,4,5,8,12,20,35,60 \
  --x-max 7.0 \
  --num-x 1401 \
  --central-window 4.5 \
  --method exact-mixture
