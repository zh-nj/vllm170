# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Shared endpoint defaults for the DeepSeek V4 benchmark harnesses."""

import os

URL = os.getenv("DSV4_BENCH_URL", "http://127.0.0.1:8098/v1/completions")
MODEL = os.getenv("DSV4_BENCH_MODEL", "dsv4s")
