#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export MODAL_BUILD_VALIDATION=ignore
exec modal run -m modal_eval.orchestrator::sync_results "$@"
