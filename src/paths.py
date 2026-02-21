from __future__ import annotations
from pathlib import Path

def repo_root() -> Path:
    return Path(__file__).resolve().parent.parent

def p(*parts: str) -> Path:
    return repo_root().joinpath(*parts)

def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)

def update_latest_symlink(runs_dir: Path, run_dir: Path) -> None:
    latest = runs_dir / "latest"
    if latest.is_symlink() or latest.exists():
        latest.unlink()
    latest.symlink_to(run_dir, target_is_directory=True)
