"""One-time migration of uploaded files from the original AgentDemo project."""

import sys
import json
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR))

from knowledge_base.store import ingest_file


SOURCE_DIR = PROJECT_DIR.parent / "AgentDemo" / "chroma_demo_data" / "uploads"
SOURCE_CATALOG = SOURCE_DIR.parent / "catalog.json"


def main() -> None:
    if not SOURCE_DIR.is_dir():
        raise SystemExit(f"Source directory does not exist: {SOURCE_DIR}")
    catalog = json.loads(SOURCE_CATALOG.read_text(encoding="utf-8")) if SOURCE_CATALOG.exists() else []
    catalog_by_stored_name = {item.get("stored_name"): item for item in catalog if isinstance(item, dict)}
    for path in sorted(SOURCE_DIR.iterdir()):
        if not path.is_file():
            continue
        source = catalog_by_stored_name.get(path.name, {})
        filename = source.get("file_name", path.name)
        content_type = source.get("content_type") or None
        item = ingest_file(filename, content_type, path.read_bytes())
        print(f"migrated {filename}: {item['chunk_count']} chunks")


if __name__ == "__main__":
    main()
