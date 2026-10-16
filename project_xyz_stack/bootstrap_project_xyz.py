from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

UPSTREAMS = {
    "xllm": "https://github.com/ifm-ai/xllm.git",
    "pretraining-data-toolkit": "https://github.com/ifm-ai/pretraining-data-toolkit.git",
    "xbridges": "https://github.com/ifm-ai/xbridges.git",
    "RL360": "https://github.com/ifm-ai/RL360.git",
    "process_entry": "https://github.com/ifm-ai/process_entry.git",
    "LCQA": "https://github.com/ifm-ai/LCQA.git",
}


def run(*cmd: str, cwd: Path | None = None) -> str:
    print("+", " ".join(cmd))
    p = subprocess.run(cmd, cwd=cwd, text=True, check=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    return p.stdout.strip()


def main() -> None:
    ap = argparse.ArgumentParser(description="Clone optional upstream training backends used by Project XYZ")
    ap.add_argument("--update", action="store_true", help="fast-forward clean existing clones")
    args = ap.parse_args()

    root = Path(__file__).resolve().parent / "_upstream" / "project_xyz_backends"
    root.mkdir(parents=True, exist_ok=True)
    lock = {}

    for name, url in UPSTREAMS.items():
        dst = root / name
        if not dst.exists():
            run("git", "clone", "--filter=blob:none", url, str(dst))
        elif args.update:
            dirty = run("git", "status", "--porcelain", cwd=dst)
            if dirty:
                raise RuntimeError(f"Refusing to update dirty upstream clone: {dst}")
            run("git", "pull", "--ff-only", cwd=dst)
        sha = run("git", "rev-parse", "HEAD", cwd=dst)
        remote = run("git", "remote", "get-url", "origin", cwd=dst)
        lock[name] = {"remote": remote, "commit": sha}

    lock_path = root / "PROJECT_XYZ_UPSTREAM_LOCK.json"
    lock_path.write_text(json.dumps(lock, indent=2), encoding="utf-8")
    print(f"\nPinned Project XYZ upstream revisions written to {lock_path}")
    print("Do not commit _upstream; the lock file is local provenance for the Project XYZ run you execute.")


if __name__ == "__main__":
    main()
