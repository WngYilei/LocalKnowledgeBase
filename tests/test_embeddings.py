import os
import sys
import types
import unittest
from unittest.mock import patch

from knowledge_base.embeddings import HashEmbeddingProvider, LocalEmbeddingProvider, get_embedding_provider


class EmbeddingTests(unittest.TestCase):
    def test_hash_provider_is_deterministic_and_normalized(self):
        provider = HashEmbeddingProvider()
        first = provider.embed_query("本地知识库")
        second = provider.embed_query("本地知识库")

        self.assertEqual(provider.dimension, len(first))
        self.assertEqual(first, second)
        self.assertAlmostEqual(sum(value * value for value in first) ** 0.5, 1.0)

    def test_hash_provider_supports_batches_and_empty_batch(self):
        provider = HashEmbeddingProvider()
        self.assertEqual([], provider.embed_documents([]))
        vectors = provider.embed_documents(["a", "b"])
        self.assertEqual(2, len(vectors))
        self.assertEqual(provider.dimension, len(vectors[0]))

    def test_factory_uses_hash_provider_for_test_profile(self):
        get_embedding_provider.cache_clear()
        with patch.dict(os.environ, {"KNOWLEDGE_BASE_EMBEDDING_PROVIDER": "hash"}, clear=False):
            provider = get_embedding_provider()
        self.assertIsInstance(provider, HashEmbeddingProvider)
        get_embedding_provider.cache_clear()

    def test_factory_rejects_unknown_provider(self):
        get_embedding_provider.cache_clear()

    def test_local_provider_uses_a_downloaded_model_without_network(self):
        class FakeVectors:
            def tolist(self):
                return [[0.6, 0.8]]

        class FakeModel:
            def __init__(self, path, **kwargs):
                self.path = path
                self.kwargs = kwargs

            def get_sentence_embedding_dimension(self):
                return 2

            def encode(self, texts, **kwargs):
                self.texts = texts
                self.encode_kwargs = kwargs
                return FakeVectors()

        fake_module = types.SimpleNamespace(SentenceTransformer=FakeModel)
        with patch.dict(sys.modules, {"sentence_transformers": fake_module}):
            provider = LocalEmbeddingProvider("/models/test-model", batch_size=4, device="cpu")
            vectors = provider.embed_documents(["hello"])

        self.assertEqual("test-model", provider.model_name)
        self.assertEqual(2, provider.dimension)
        self.assertEqual([[0.6, 0.8]], vectors)
        self.assertEqual([0.6, 0.8], provider.embed_query("hello"))
        with patch.dict(os.environ, {"KNOWLEDGE_BASE_EMBEDDING_PROVIDER": "unknown"}, clear=False):
            with self.assertRaises(ValueError):
                get_embedding_provider()
        get_embedding_provider.cache_clear()


if __name__ == "__main__":
    unittest.main()
