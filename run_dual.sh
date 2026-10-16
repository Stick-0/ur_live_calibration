#!/usr/bin/env bash
set -euo pipefail

bundle_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export TORCH_HOME="$bundle_dir/torch_cache"
export PYTHONPATH="$bundle_dir${PYTHONPATH:+:$PYTHONPATH}"

if [[ $# -eq 0 ]]; then
    set -- --help
fi

exec "${ROBOTPOSE_PYTHON:-python3}" -m urcal.inference.infer_live_dual "$@"
