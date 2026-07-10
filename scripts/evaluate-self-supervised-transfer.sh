#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-data/derived/models/self_supervised_transformer_transfer}" \
SSL_EPOCHS="${SSL_EPOCHS:-6}" \
SUPERVISED_EPOCHS="${SUPERVISED_EPOCHS:-12}" \
MAX_PRETRAIN_WINDOWS="${MAX_PRETRAIN_WINDOWS:-2048}" \
  ./scripts/evaluate-self-supervised-transformer.sh
