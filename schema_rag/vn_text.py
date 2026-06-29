"""Shared Vietnamese text normalization (có dấu -> không dấu).

This is the single source of truth for turning Vietnamese text into a normalized
ASCII không-dấu form used across retrieval (alias matching, BM25, entity fuzzy
matching) and SQL identifier validation.

Design choice: we use unicodedata NFKD + combining-mark stripping + an explicit
``đ -> d`` rule instead of pulling in ``unidecode``. NFKD already removes every
Vietnamese tone/vowel mark, ``đ`` is the only base letter NFKD leaves intact, and
keeping zero extra dependencies matches the rest of this project.
"""
from __future__ import annotations

import re
import unicodedata

__all__ = [
    "normalize_vietnamese_text",
    "normalize_identifier",
    "has_diacritics",
    "tokenize",
]


def normalize_vietnamese_text(value: object) -> str:
    """Lowercase, strip Vietnamese diacritics, and collapse to space-joined tokens.

    Examples::

        "Khách hàng đang hoạt động" -> "khach hang dang hoat dong"
        "trang_thai_khach_hang"     -> "trang thai khach hang"
    """
    text = unicodedata.normalize("NFKD", str(value or ""))
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    text = text.lower()
    text = text.replace("đ", "d")
    # Underscores are separators too, so schema identifiers (khach_hang) and natural
    # Vietnamese phrases (khách hàng) collapse to the same space-joined không-dấu form.
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return " ".join(text.split())


def normalize_identifier(identifier: object) -> str:
    """Normalized form of a schema identifier (table/column).

    Behaviourally identical to :func:`normalize_vietnamese_text`; kept as a named
    helper so call sites read clearly (``normalize_identifier("don_hang_ban")`` ->
    ``"don hang ban"``).
    """
    return normalize_vietnamese_text(identifier)


def has_diacritics(text: object) -> bool:
    """True when *text* contains any Vietnamese diacritic or the letter đ/Đ.

    Used by the SQL validator to reject identifiers that are not plain không-dấu
    snake_case. It deliberately only looks at the characters themselves, so it can
    be applied to identifier tokens (not string literals, which may legitimately
    contain dấu).
    """
    s = str(text or "")
    if not s:
        return False
    decomposed = unicodedata.normalize("NFKD", s)
    if any(unicodedata.combining(ch) for ch in decomposed):
        return True
    return any(ch in "đĐ" for ch in s)


def tokenize(value: object) -> list[str]:
    """Normalized whitespace tokens, ready for BM25 / n-gram alias lookup."""
    normalized = normalize_vietnamese_text(value)
    return normalized.split() if normalized else []
