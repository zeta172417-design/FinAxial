from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping


class SwanLabRun:
    """Small strict wrapper around SwanLab, imported only by experiment jobs."""

    def __init__(
        self,
        *,
        enabled: bool,
        project: str,
        name: str,
        logdir: str | Path,
        config: Mapping[str, Any],
        tags: list[str] | None = None,
        mode: str = "online",
    ) -> None:
        self.enabled = bool(enabled)
        self.run = None
        if not self.enabled:
            return
        import swanlab

        normalized_mode = "online" if mode == "cloud" else mode
        Path(logdir).mkdir(parents=True, exist_ok=True)
        self.run = swanlab.init(
            project=project,
            name=name,
            log_dir=str(logdir),
            mode=normalized_mode,
            config=dict(config),
            tags=tags or [],
            resume="never",
            reinit=True,
        )

    def log(self, values: Mapping[str, Any], *, step: int | None = None) -> None:
        if self.run is not None:
            self.run.log(dict(values), step=step)

    def finish(self, state: str = "success", error: str | None = None) -> None:
        if self.run is not None:
            self.run.finish(state=state, error=error)
            self.run = None

    def __enter__(self) -> "SwanLabRun":
        return self

    def __exit__(self, exc_type, exc, traceback) -> bool:
        if exc is None:
            self.finish("success")
        else:
            self.finish("crashed", str(exc))
        return False
