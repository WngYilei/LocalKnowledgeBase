import unittest

from knowledge_base.sanitizer import SANITIZER_VERSION, contains_common_pii, sanitize_text


class SanitizerTests(unittest.TestCase):
    def test_masks_common_pii(self):
        text = "手机号 13812345678，身份证 11010519491231002X，邮箱 user@example.com，银行卡 622202123456789012。"
        masked, findings = sanitize_text(text)

        self.assertEqual(SANITIZER_VERSION, "pii-v2")
        self.assertIn("138****5678", masked)
        self.assertIn("110105********002X", masked)
        self.assertIn("u***@example.com", masked)
        self.assertIn("6222**********9012", masked)
        self.assertFalse(contains_common_pii(masked))
        self.assertGreaterEqual(sum(item.count for item in findings), 4)

    def test_masks_secrets_and_ip_addresses(self):
        text = "Bearer abcdefghijklmnopqrstuvwxyz123456 192.168.1.10 sk-abcdefghijklmnopqrstuvwxyz123456"
        masked, findings = sanitize_text(text)

        self.assertNotIn("abcdefghijklmnopqrstuvwxyz123456", masked)
        self.assertIn("<IP_REDACTED>", masked)
        self.assertIn("<SECRET_REDACTED>", masked)
        self.assertEqual({item.kind for item in findings}, {"bearer_token", "ipv4", "secret"})

    def test_does_not_mask_short_numbers_as_bank_cards(self):
        text = "订单号 123456789012345，版本 2026.10.03"
        masked, findings = sanitize_text(text)

        self.assertEqual(text, masked)
        self.assertEqual([], findings)

    def test_repeated_identifiers_are_counted(self):
        text = "13812345678 13812345678"
        masked, findings = sanitize_text(text)

        self.assertEqual("138****5678 138****5678", masked)
        self.assertEqual(2, next(item.count for item in findings if item.kind == "mobile"))


if __name__ == "__main__":
    unittest.main()
