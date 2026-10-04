"""PII and secret sanitization used at both ingestion and retrieval boundaries."""

from __future__ import annotations

import re
from dataclasses import dataclass


SANITIZER_VERSION = "pii-v2"

ID_CARD_18 = re.compile(
    r"(?<![0-9A-Za-z])(\d{6})(?:18|19|20)\d{2}(?:0[1-9]|1[0-2])"
    r"(?:0[1-9]|[12]\d|3[01])(\d{3}[\dXx])(?![0-9A-Za-z])"
)
ID_CARD_15 = re.compile(
    r"(?<![0-9A-Za-z])(\d{6})\d{2}(?:0[1-9]|1[0-2])"
    r"(?:0[1-9]|[12]\d|3[01])(\d{3})(?![0-9A-Za-z])"
)
MOBILE_PHONE = re.compile(r"(?<!\d)(1[3-9]\d)([- ]?)(\d{4})([- ]?)(\d{4})(?!\d)")
EMAIL = re.compile(
    r"(?i)(?<![A-Za-z0-9._%+-])([A-Za-z0-9._%+-])"
    r"[A-Za-z0-9._%+-]*(@[A-Za-z0-9.-]+\.[A-Za-z]{2,})"
    r"(?![A-Za-z0-9._%+-])"
)
BANK_CARD = re.compile(r"(?<![0-9A-Za-z])(?:\d[ -]?){16,19}(?![0-9A-Za-z])")
IPV4 = re.compile(r"(?<![\d.])(?:\d{1,3}\.){3}\d{1,3}(?![\d.])")
BEARER_TOKEN = re.compile(r"(?i)(\bBearer\s+)[A-Za-z0-9._~+/=-]{20,}")
COMMON_SECRET = re.compile(
    r"(?i)(?<![A-Za-z0-9])(?:sk-[A-Za-z0-9_-]{20,}|AKIA[0-9A-Z]{16})"
    r"(?![A-Za-z0-9])"
)


@dataclass(frozen=True)
class SanitizationFinding:
    kind: str
    count: int


def _replace(pattern: re.Pattern[str], text: str, replacement: str, kind: str, findings: list[SanitizationFinding]) -> str:
    count = len(pattern.findall(text))
    if count:
        findings.append(SanitizationFinding(kind, count))
        text = pattern.sub(replacement, text)
    return text


def sanitize_text(text: str) -> tuple[str, list[SanitizationFinding]]:
    """Return sanitized text and the categories that were replaced."""
    findings: list[SanitizationFinding] = []
    count = len(ID_CARD_18.findall(text))
    text = ID_CARD_18.sub(lambda m: f"{m.group(1)}********{m.group(2)}", text)
    if count:
        findings.append(SanitizationFinding("id_card_18", count))

    count = len(ID_CARD_15.findall(text))
    text = ID_CARD_15.sub(lambda m: f"{m.group(1)}******{m.group(2)}", text)
    if count:
        findings.append(SanitizationFinding("id_card_15", count))

    text = _replace(MOBILE_PHONE, text, r"\1\2****\4\5", "mobile", findings)
    text = _replace(EMAIL, text, r"\1***\2", "email", findings)

    def mask_card(match: re.Match[str]) -> str:
        digits = re.sub(r"\D", "", match.group(0))
        if not 16 <= len(digits) <= 19:
            return match.group(0)
        return f"{digits[:4]}{'*' * (len(digits) - 8)}{digits[-4:]}"

    card_matches = [match for match in BANK_CARD.finditer(text) if 16 <= len(re.sub(r"\D", "", match.group(0))) <= 19]
    if card_matches:
        findings.append(SanitizationFinding("bank_card", len(card_matches)))
        text = BANK_CARD.sub(mask_card, text)

    text = _replace(IPV4, text, "<IP_REDACTED>", "ipv4", findings)
    text = _replace(BEARER_TOKEN, text, r"\1<SECRET_REDACTED>", "bearer_token", findings)
    text = _replace(COMMON_SECRET, text, "<SECRET_REDACTED>", "secret", findings)
    return text, findings


def desensitize_text(text: str) -> str:
    """Backward-compatible convenience wrapper."""
    return sanitize_text(text)[0]


def contains_common_pii(text: str) -> bool:
    """Best-effort safety check used before returning data to callers."""
    patterns = (ID_CARD_18, ID_CARD_15, MOBILE_PHONE, EMAIL, BANK_CARD, BEARER_TOKEN, COMMON_SECRET)
    return any(pattern.search(text) for pattern in patterns)
