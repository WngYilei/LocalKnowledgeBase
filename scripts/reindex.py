"""Rebuild a collection from the stored source files using the configured provider."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR))

from knowledge_base.store import UPLOAD_DIR, _load_catalog, ingest_file  # noqa: E402


def main() -> None:
    catalog = _load_catalog()
    if not catalog:
        print("No catalog entries found.")
        return

    failures = 0
    for item in catalog:
        stored_name = Path(str(item.get("stored_name", ""))).name
        path = (UPLOAD_DIR / stored_name).resolve()
        if UPLOAD_DIR.resolve() not in path.parents or not path.is_file():
            print(f"missing source: {stored_name}", file=sys.stderr)
            failures += 1
            continue
        try:
            result = ingest_file(
                str(item.get("file_name") or stored_name),
                item.get("content_type") or None,
                path.read_bytes(),
            )
            print(f"reindexed {result['file_name']}: {result['chunk_count']} chunks")
        except Exception as exc:
            print(f"failed {stored_name}: {exc}", file=sys.stderr)
            failures += 1

    if failures:
        raise SystemExit(f"Reindex completed with {failures} failure(s).")


if __name__ == "__main__":
    main()
