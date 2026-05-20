from __future__ import annotations

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
ALLOWED_SPREADSHEET_ROOT = REPO_ROOT / "data" / "camels_us" / "selected"
DISALLOWED_SUFFIXES = {".docx", ".zip", ".pt", ".pth", ".pkl", ".pickle"}


def is_within(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except ValueError:
        return False


def main() -> None:
    problems: list[str] = []
    for path in REPO_ROOT.rglob("*"):
        if not path.is_file():
            continue
        if ".git" in path.parts or "__pycache__" in path.parts:
            continue
        suffix = path.suffix.lower()
        if suffix in DISALLOWED_SUFFIXES:
            problems.append(f"disallowed file type: {path.relative_to(REPO_ROOT)}")
        if suffix in {".xls", ".xlsx"} and not is_within(path, ALLOWED_SPREADSHEET_ROOT):
            problems.append(f"spreadsheet outside public CAMELS subset: {path.relative_to(REPO_ROOT)}")
    if problems:
        for problem in problems:
            print(problem)
        raise SystemExit(1)
    print("Release audit passed: no disallowed files detected.")


if __name__ == "__main__":
    main()
