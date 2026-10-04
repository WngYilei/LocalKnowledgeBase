import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from knowledge_base import store


class FakeProvider:
    model_name = "fake-local-model"
    dimension = 3

    def embed_documents(self, texts):
        return [[float(len(text)), 1.0, 0.0] for text in texts]

    def embed_query(self, text):
        return [float(len(text)), 1.0, 0.0]


class FakeCollection:
    def __init__(self):
        self.upserted = []

    def upsert(self, **kwargs):
        self.upserted.append(kwargs)

    def count(self):
        return sum(len(item["ids"]) for item in self.upserted)


class StoreSecurityTests(unittest.TestCase):
    def test_search_results_are_redacted_and_metadata_is_allowlisted(self):
        rows = [
            {
                "id": "chunk-1",
                "content": "联系人手机号 13812345678，邮箱 user@example.com",
                "distance": 0.12,
                "metadata": {
                    "file_hash": "abc",
                    "file_name": "张三-合同.md",
                    "stored_path": "/private/raw/secret.pdf",
                    "source_kind": "markdown",
                    "page_number": 2,
                    "chunk_index": 1,
                },
            }
        ]

        result = store.sanitize_search_results(rows)[0]

        self.assertNotIn("13812345678", result["content"])
        self.assertNotIn("user@example.com", result["content"])
        self.assertNotIn("stored_path", result["metadata"])
        self.assertNotIn("/private/raw/secret.pdf", str(result))
        self.assertEqual(2, result["page_number"])

    def test_ingestion_stores_sanitized_text_and_versioned_metadata(self):
        with tempfile.TemporaryDirectory() as directory:
            data_dir = Path(directory)
            upload_dir = data_dir / "uploads"
            catalog_path = data_dir / "catalog.json"
            collection = FakeCollection()

            overrides = {
                "DATA_DIR": data_dir,
                "UPLOAD_DIR": upload_dir,
                "CATALOG_PATH": catalog_path,
                "INITIALIZED": True,
            }
            with patch.multiple(store, **overrides), patch.object(store, "get_embedding_provider", return_value=FakeProvider()), patch.object(
                store, "open_collection", return_value=collection
            ):
                item = store.ingest_file(
                    "customer.md",
                    "text/markdown",
                    "客户手机号：13812345678\n邮箱：user@example.com".encode(),
                )

            record = collection.upserted[0]
            self.assertEqual(1, len(record["documents"]))
            self.assertNotIn("13812345678", record["documents"][0])
            self.assertNotIn("user@example.com", record["documents"][0])
            self.assertNotIn("stored_path", record["metadatas"][0])
            self.assertEqual("fake-local-model", record["metadatas"][0]["embedding_model"])
            self.assertTrue((upload_dir / Path(item["file_hash"][:12] + "_customer.md")).exists())

            catalog = json.loads(catalog_path.read_text(encoding="utf-8"))
            self.assertEqual("uploads/" + catalog[0]["stored_name"], catalog[0]["stored_path"])

    def test_file_path_rejects_invalid_hash(self):
        self.assertIsNone(store.get_file_path("../../etc/passwd"))
        self.assertIsNone(store.get_file_path("not-a-sha256"))

    def test_search_rejects_unbounded_top_k(self):
        with patch.object(store, "ensure_initialized"), patch.object(store, "open_collection") as open_collection:
            with self.assertRaises(ValueError):
                store.search_documents("test", 51)
            open_collection.assert_not_called()


if __name__ == "__main__":
    unittest.main()
