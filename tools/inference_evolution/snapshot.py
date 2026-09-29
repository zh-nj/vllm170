# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Capture this vLLM editable checkout as an independent runtime capsule.

Linux/uv CPython layout only. GPU model weights, CUDA toolkit and host drivers
remain shared dependencies; do not modify them during experiments. Unknown
editable paths require operator inspection rather than silent substitution.
"""

import argparse
import os
import shutil
import subprocess
import time
from pathlib import Path

from controller import Controller, atomic, identifier


def copy(source, destination):
    destination.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["cp", "-a", "--reflink=auto", str(source), str(destination)], check=True
    )


def snapshot(root, repo, name, pid, external):
    identifier(name)
    folder = root / "releases" / name
    folder.mkdir(mode=0o700)
    source = folder / "source"
    source.mkdir()
    archive = folder / "source.tar"
    subprocess.run(
        ["git", "archive", "--output", str(archive), "HEAD"], cwd=repo, check=True
    )
    subprocess.run(["tar", "-xf", str(archive), "-C", str(source)], check=True)
    archive.unlink()
    # Include ignored native binaries, without modifying any original mmap file.
    shutil.rmtree(source / "vllm")
    copy(repo / "vllm", source / "vllm")
    copy(repo / ".venv", folder / "env")
    python_base = (repo / ".venv/bin/python").resolve().parent.parent
    copy(python_base, folder / "python")
    binary = folder / "env/bin/python"
    binary.unlink()
    binary.symlink_to("../../python/bin/python3.12")
    cfg = folder / "env/pyvenv.cfg"
    cfg.write_text(
        "\n".join(
            f"home = {folder / 'python/bin'}" if line.startswith("home =") else line
            for line in cfg.read_text().splitlines()
        )
        + "\n"
    )
    rewrites = {
        str(repo / ".venv"): str(folder / "env"),
        str(python_base): str(folder / "python"),
        str(repo): str(source),
    }
    for original in external:
        dest = folder / "external" / original.name
        copy(original, dest)
        rewrites[str(original)] = str(dest)
    patchelf = shutil.which("patchelf")
    if not patchelf:
        raise RuntimeError("patchelf is required to relocate native library paths")
    for parent in (source / "vllm", folder / "env", folder / "external"):
        for path in parent.rglob("*.so*"):
            if not path.is_file() or path.is_symlink():
                continue
            result = subprocess.run(
                [patchelf, "--print-rpath", str(path)], capture_output=True, text=True
            )
            if result.returncode:
                continue
            old_path = result.stdout.strip()
            new_path = old_path
            for old, new in rewrites.items():
                new_path = new_path.replace(old, new)
            if new_path != old_path:
                subprocess.run(
                    [patchelf, "--set-rpath", new_path, str(path)], check=True
                )
    site = folder / "env/lib/python3.12/site-packages"
    for path in [*site.glob("*.pth"), *site.glob("__editable__*finder.py")]:
        content = path.read_text()
        for old, new in rewrites.items():
            content = content.replace(old, new)
        path.write_text(content)
        for line in content.splitlines():
            if line.startswith("/") and not Path(line).is_relative_to(folder):
                raise ValueError(f"Uncaptured editable path: {path.name}: {line}")
    # Console script shebangs must not reach back into the original environment.
    for path in (folder / "env/bin").iterdir():
        if path.is_file() and not path.is_symlink():
            data = path.read_bytes()
            if data.startswith(b"#!") and str(repo / ".venv").encode() in data[:256]:
                path.write_bytes(
                    data.replace(
                        str(repo / ".venv").encode(), str(folder / "env").encode()
                    )
                )
    proc = Path(f"/proc/{pid}")
    argv = proc.joinpath("cmdline").read_bytes().decode().rstrip("\0").split("\0")
    if argv[:2] != [str(repo / ".venv/bin/python"), str(repo / ".venv/bin/vllm")]:
        raise ValueError("Expected this checkout's vllm console entrypoint")
    argv[:2] = [str(folder / "env/bin/python"), str(folder / "env/bin/vllm")]
    original_env = dict(
        e.split("=", 1)
        for e in proc.joinpath("environ").read_bytes().decode().split("\0")
        if e
    )
    env = {
        k: v
        for k, v in original_env.items()
        if k.startswith(
            ("VLLM_", "NCCL_", "CUDA_", "HF_", "OMP_", "TORCH_", "TRITON_", "TILELANG_")
        )
    }
    cache = root / "runtime" / name
    cache.mkdir(parents=True)
    env.update(
        PATH=f"{folder / 'env/bin'}:/usr/local/cuda/bin:/usr/bin:/bin",
        LD_LIBRARY_PATH=f"{site / 'torch/lib'}:/usr/local/cuda/lib64",
        PYTHONDONTWRITEBYTECODE="1",
        PYTHONNOUSERSITE="1",
        VLLM_CACHE_ROOT=str(cache / "vllm"),
        TRITON_CACHE_DIR=str(cache / "triton"),
        TILELANG_CACHE_DIR=str(cache / "tilelang"),
        TORCH_EXTENSIONS_DIR=str(cache / "torch_extensions"),
        FLASHINFER_WORKSPACE_BASE=str(cache / "flashinfer"),
    )
    # TileLang and Triton binaries are reusable seeds. Each release writes its
    # own cache; vLLM compilation graphs are rebuilt for relocated source paths.
    for original, dest in [
        (Path.home() / ".cache/triton", cache / "triton"),
        (Path.home() / ".tilelang/cache", cache / "tilelang"),
    ]:
        if original.exists():
            copy(original, dest)
    shutil.copy2(Path(__file__).with_name("smoke.py"), folder / "smoke.py")
    port = argv[argv.index("--port") + 1]
    model = argv[argv.index("--served-model-name") + 1]
    base = f"http://127.0.0.1:{port}"
    spec = {
        "id": name,
        "cwd": str(source),
        "argv": argv,
        "env": env,
        "probe": {"base_url": base, "model": model, "prompt": "Hello"},
        "build": [],
        "checks": [
            [
                str(folder / "env/bin/python"),
                str(folder / "smoke.py"),
                "--base",
                base,
                "--model",
                model,
            ]
        ],
        "seal": ["source", "env", "python", "smoke.py"]
        + (["external"] if external else []),
        "startup_timeout": 1800,
        "job_timeout": 420,
    }
    atomic(
        folder / "provenance.json",
        {
            "captured_at": time.time(),
            "pid": pid,
            "repo": str(repo),
            "commit": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=repo, text=True
            ).strip(),
            "shared": ["model weights", "CUDA toolkit", "GPU driver", "OS libraries"],
            "cache_policy": "Per-release mutable JIT caches; outside artifact seal",
        },
    )
    controller = Controller(root)
    controller.store.register(spec)
    controller.seal(name)
    print(folder, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--repo", required=True, type=Path)
    parser.add_argument("--id", required=True)
    parser.add_argument("--pid", required=True, type=int)
    parser.add_argument("--external", action="append", type=Path, default=[])
    args = parser.parse_args()
    os.umask(0o077)
    snapshot(
        args.root.resolve(),
        args.repo.resolve(),
        args.id,
        args.pid,
        [p.resolve() for p in args.external],
    )
