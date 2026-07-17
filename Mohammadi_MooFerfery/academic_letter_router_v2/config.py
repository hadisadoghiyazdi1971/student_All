from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(override=False)


@dataclass(frozen=True)
class ProjectPaths:
    root: Path
    data_dir: Path
    taxonomy_dir: Path
    cache_dir: Path
    runtime_dir: Path
    settings_db: Path

    @classmethod
    def from_environment(cls) -> "ProjectPaths":
        root = Path(os.getenv("APP_ROOT", Path(__file__).resolve().parent)).resolve()
        data_dir = Path(os.getenv("DATA_DIR", root / "data")).resolve()
        taxonomy_dir = Path(os.getenv("TAXONOMY_DIR", data_dir / "taxonomy")).resolve()
        cache_dir = Path(os.getenv("CACHE_DIR", root / "cache")).resolve()
        runtime_dir = Path(os.getenv("RUNTIME_DIR", root / "runtime")).resolve()
        settings_db = Path(os.getenv("SETTINGS_DB_PATH", runtime_dir / "settings.db")).resolve()

        cache_dir.mkdir(parents=True, exist_ok=True)
        runtime_dir.mkdir(parents=True, exist_ok=True)
        return cls(
            root=root,
            data_dir=data_dir,
            taxonomy_dir=taxonomy_dir,
            cache_dir=cache_dir,
            runtime_dir=runtime_dir,
            settings_db=settings_db,
        )


PATHS = ProjectPaths.from_environment()
SETTINGS_PROFILE = os.getenv("APP_SETTINGS_PROFILE", "default").strip() or "default"
