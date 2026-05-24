from __future__ import annotations

from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile


def make_submission_zip(project_root: Path, out_dir: Path, zip_name: str = "submission.zip") -> Path:
    """打包 outputs 和工程代码为 submission.zip。"""
    project_root = Path(project_root).resolve()
    out_dir = Path(out_dir).resolve()
    zip_path = out_dir / zip_name
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    include_suffixes = {".py", ".md", ".txt", ".csv", ".json", ".sh", ".png", ".jpg", ".jpeg"}

    with ZipFile(zip_path, "w", compression=ZIP_DEFLATED) as zf:
        if out_dir.exists():
            for path in out_dir.rglob("*"):
                if path == zip_path or path.is_dir() or path.suffix.lower() not in include_suffixes:
                    continue
                zf.write(path, path.relative_to(project_root.parent))

        for path in project_root.rglob("*"):
            if path == zip_path or path.is_dir() or path.suffix.lower() not in include_suffixes:
                continue
            try:
                path.relative_to(out_dir)
                continue
            except ValueError:
                pass
            zf.write(path, path.relative_to(project_root.parent))
    return zip_path
