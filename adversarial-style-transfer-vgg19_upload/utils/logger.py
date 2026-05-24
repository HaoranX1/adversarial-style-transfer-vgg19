from __future__ import annotations

import json
from pathlib import Path
from typing import Dict


class SimpleLogger:
    """简单文本日志器，同时输出到终端和文件。"""

    def __init__(self, log_path: Path):
        self.log_path = Path(log_path)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self._file = self.log_path.open("w", encoding="utf-8")

    def log(self, message: str) -> None:
        print(message, flush=True)
        self._file.write(message + "\n")
        self._file.flush()

    def log_dict(self, payload: Dict[str, object]) -> None:
        self.log(json.dumps(payload, ensure_ascii=False))

    def close(self) -> None:
        self._file.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()
