#!/usr/bin/env bash
# Wrapper for modal run that tolerates IDE auto-save during long image builds.
set -euo pipefail
cd "$(dirname "$0")/.."
export MODAL_BUILD_VALIDATION=ignore
exec modal run -m modal_eval.orchestrator::run_eval "$@"
