#!/usr/bin/env bash
set -euo pipefail

bundle_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export TORCH_HOME="$bundle_dir/torch_cache"
export PYTHONPATH="$bundle_dir${PYTHONPATH:+:$PYTHONPATH}"
if [[ $# -eq 0 ]]; then
    set -- --help
fi
if [[ "${ROBOTPOSE_REFERENCE:-0}" == "1" ]]; then
    exec "${ROBOTPOSE_PYTHON:-python3}" "$bundle_dir/reference/infer_live_ur10e.py" \
        --checkpoint "$bundle_dir/trained_models/ur10e_2/best.pt" "$@"
fi
exec "${ROBOTPOSE_PYTHON:-python3}" -m urcal.inference.infer_live_ur10e \
    --checkpoint "$bundle_dir/trained_models/ur10e_2/best.pt" "$@"
