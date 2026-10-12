#!/usr/bin/env bash
set -euo pipefail

bundle_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export TORCH_HOME="$bundle_dir/torch_cache"
entry_dir="$bundle_dir"
if [[ "${ROBOTPOSE_REFERENCE:-0}" == "1" ]]; then
    entry_dir="$bundle_dir/reference"
fi

if [[ $# -eq 0 ]]; then
    set -- --help
fi

exec "${ROBOTPOSE_PYTHON:-python3}" "$entry_dir/infer_live.py" \
    --checkpoint "$bundle_dir/trained_models/v3m2a3/best.pt" "$@"
