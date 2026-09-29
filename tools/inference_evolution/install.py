# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Install a pinned CPU supervisor outside the inference checkout."""

import argparse
import hashlib
import json
import os
import shutil
import subprocess
from pathlib import Path

from controller import Store, atomic, identifier


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--prefix", type=Path, default=Path.home() / ".local/share/inference-evolution"
    )
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--instance", default="dsv41f")
    parser.add_argument("--dsh", type=Path)
    parser.add_argument("--start", action="store_true")
    args = parser.parse_args()
    identifier(args.instance)
    root = args.root.expanduser().resolve()
    prefix = args.prefix.expanduser().resolve()
    unit = f"inference-evolution-{args.instance}.service"
    unit_path = Path.home() / ".config/systemd/user" / unit
    if unit_path.exists() or (root.exists() and any(root.iterdir())):
        raise ValueError("Existing state/unit is preserved; choose a fresh instance")
    source = Path(__file__).parent.resolve()
    repo = source.parents[1]
    if prefix.is_relative_to(repo) or root.is_relative_to(repo):
        raise ValueError("Install and state directories must be outside the checkout")
    if any(c in str(prefix) + str(root) for c in '\n%\\"'):
        raise ValueError("Unsupported character in installation path")
    prefix.mkdir(parents=True, exist_ok=True)
    python = prefix / ".venv/bin/python"
    if not python.exists():
        subprocess.run(
            ["uv", "venv", "--python", "3.12", str(prefix / ".venv")], check=True
        )
    runtime_files = [
        source / name
        for name in (
            "controller.py",
            "dsh_bridge.py",
            "dsh.patch.yml",
            "adopt.py",
            "snapshot.py",
            "smoke.py",
        )
    ]
    skill = repo / ".agents/skills/self-evolve-inference"
    files = runtime_files + sorted(p for p in skill.rglob("*") if p.is_file())
    version = hashlib.sha256(b"".join(p.read_bytes() for p in files)).hexdigest()[:16]
    package = prefix / "versions" / version
    if not package.exists():
        package.mkdir(parents=True)
        for file in runtime_files:
            shutil.copy2(file, package / file.name)
        shutil.copytree(skill, package / "skills/self-evolve-inference")
    config = {
        "poll_seconds": 10,
        "probe_timeout": 20,
        "failure_threshold": 3,
        "stable_restarts": 1,
    }
    if args.dsh:
        pnpm = shutil.which("pnpm")
        node = shutil.which("node")
        if not pnpm or not node:
            raise ValueError("DSH requires pnpm and node on PATH")
        profile = (
            Path(os.environ.get("DSH_HOME", str(Path.home() / ".dsh")))
            / "profiles/inference-evolution/package.json"
        )
        profile_args = (
            [] if profile.exists() else ["--from-default-profile", "headless"]
        )
        subprocess.run(
            [pnpm, "dsh", "--profile", "inference-evolution", *profile_args, "--help"],
            cwd=args.dsh,
            check=True,
            stdout=subprocess.DEVNULL,
        )
        config["resume"] = {
            "argv": [
                str(python),
                str(package / "dsh_bridge.py"),
                "--config",
                str(root / "dsh.json"),
                "{prompt}",
            ],
            "cwd": str(root),
            "timeout": 600,
            "max_attempts": 1,
        }
    if (root / "state.json").exists():
        raise ValueError(
            "Existing state is preserved; use a new instance for installation"
        )
    Store(root).initialize(config)
    if args.dsh:
        overlay = (package / "dsh.patch.yml").read_text()
        overlay += (
            "\n- id: fs-sandbox\n  config:\n    cwd: "
            + json.dumps(str(root / "notes"))
            + "\n- id: skill-filesystem\n  config:\n    includeDefaultRoots: false"
            + "\n    customSkillDirs:\n      - "
            + json.dumps(str(package / "skills"))
            + "\n"
        )
        (root / "dsh.patch.yml").write_text(overlay)
        atomic(
            root / "dsh.json",
            {
                "argv": [
                    pnpm,
                    "dsh",
                    "--profile",
                    "inference-evolution",
                    "--patch",
                    str(root / "dsh.patch.yml"),
                    "--json",
                ],
                "cwd": str(args.dsh.resolve()),
            },
        )
    unit = f"inference-evolution-{args.instance}.service"
    unit_path = Path.home() / ".config/systemd/user" / unit
    if unit_path.exists():
        raise ValueError(f"Existing unit is preserved: {unit_path}")
    unit_path.parent.mkdir(parents=True, exist_ok=True)
    env_path = os.environ.get("PATH", "/usr/bin:/bin")
    unit_path.write_text(
        "[Unit]\nDescription=CPU inference evolution supervisor\n"
        "StartLimitIntervalSec=120\nStartLimitBurst=5\n\n"
        "[Service]\nType=simple\n"
        f'WorkingDirectory={root}\nEnvironment="PATH={env_path}"\n'
        f'ExecStart="{python}" "{package / "controller.py"}" --root "{root}" run\n'
        "Restart=on-failure\nRestartSec=5\nUMask=0077\nNoNewPrivileges=yes\n"
        "\n[Install]\nWantedBy=default.target\n"
    )
    # Give the skill a stable, file-backed locator in the installed notes workspace.
    atomic(
        root / "installation.json",
        {
            "root": str(root),
            "python": str(python),
            "package": str(package),
            "unit": unit,
            "version": version,
        },
    )
    subprocess.run(["systemctl", "--user", "daemon-reload"], check=True)
    if args.start:
        subprocess.run(["systemctl", "--user", "enable", "--now", unit], check=True)
    print(json.dumps({"unit": unit, "root": str(root), "package": str(package)}))


if __name__ == "__main__":
    main()
