import unittest

from knowledge_base.store import chunk_text, desensitize_text


class StoreTests(unittest.TestCase):
    def test_desensitize_common_pii(self):
        text = "手机号 13812345678，身份证 11010519491231002X，邮箱 user@example.com，银行卡 622202123456789012。"
        masked = desensitize_text(text)
        self.assertIn("138****5678", masked)
        self.assertIn("110105********002X", masked)
        self.assertIn("u***@example.com", masked)
        self.assertIn("6222**********9012", masked)
        self.assertNotIn("13812345678", masked)


    def test_chunk_text_has_overlap(self):
        chunks = chunk_text("a" * 1900)
        self.assertEqual(len(chunks), 3)
        self.assertEqual(chunks[0][-120:], chunks[1][:120])


if __name__ == "__main__":
    unittest.main()
