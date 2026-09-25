"""Audit tracked source boundaries and fixed external dependency metadata."""

import json
from pathlib import Path
import re
import subprocess

ROOT = Path(__file__).resolve().parents[1]


def main():
    names = subprocess.check_output(["git", "ls-files", "-z"], cwd=ROOT).decode().split("\0")
    errors = []
    for name in filter(None, names):
        path = ROOT / name
        if name.startswith("third_party/"):
            continue
        if any(part in {"output", "temp", "__pycache__", "agent", ".vscode"} for part in path.relative_to(ROOT).parts):
            errors.append(f"Generated/internal path: {name}")
        if path.suffix in {".pth", ".pt", ".pkl", ".npy", ".npz", ".so"} or path.stat().st_size > 2_000_000:
            errors.append(f"Asset or oversized tracked file: {name}")
        if path.is_file():
            text = path.read_text(errors="replace")
            # Match machine-specific configuration values, not general documentation examples.
            for pattern in (r"/home/(?:ymr|mingrui)/", r"/data/(?:mingrui|dataset)/", r"\.\./(?:BimanBODex|HumanGraspData)/", r"git@github\.com:"):
                if re.search(pattern, text):
                    errors.append(f"Private path/remote pattern in {name}")
            if re.search(r"(?:gh[pousr]_[A-Za-z0-9]{30,}|^-----BEGIN [A-Z ]*PRIVATE KEY-----$)", text, re.MULTILINE):
                errors.append(f"Credential-like content in {name}")
    deps = json.loads((ROOT / "releases/dependencies.json").read_text())["dependencies"]
    for dep in deps:
        if not dep["url"].startswith("https://github.com/") or not re.fullmatch(r"[0-9a-f]{40}", dep["commit"]):
            errors.append(f"Invalid dependency lock: {dep['name']}")
        entry = subprocess.check_output(["git", "ls-files", "--stage", f"third_party/{dep['name']}"], cwd=ROOT, text=True)
        if not entry.startswith(f"160000 {dep['commit']} "):
            errors.append(f"Missing or mismatched dependency gitlink: {dep['name']}")
    print(json.dumps({"tracked_entries": len(list(filter(None, names))), "errors": errors}, indent=2))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
