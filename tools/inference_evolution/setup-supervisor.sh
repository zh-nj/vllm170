#!/usr/bin/env bash
set -euo pipefail

# The supervisor imports only Python's standard library. Keep its interpreter
# outside the checkout and outside the inference service's vLLM environment.
source_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
repo_dir=$(cd -- "${source_dir}/../.." && pwd)
exec "${repo_dir}/.venv/bin/python" "${source_dir}/install.py" "$@"
