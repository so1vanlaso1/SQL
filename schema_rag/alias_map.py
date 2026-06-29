"""Exact alias / synonym map: normalized Vietnamese phrase -> schema identifiers.

Built offline from the schema catalog (table names, column names, and the curated
``aliases`` lists in schema_def). At query time we normalize the question and look
for any alias phrase that appears in it; matches get a strong boost during candidate
ranking (plan §11.1 "exact alias matching should have strong priority").

The map is a plain JSON dict so it is easy to inspect, diff, and hand-edit::

    {
      "khach hang":   [{"type": "table",  "identifier": "khach_hang",            "source": "schema"}],
      "cua hang":     [{"type": "table",  "identifier": "khach_hang",            "source": "alias"}],
      "ngay dat hang":[{"type": "column", "identifier": "don_hang_ban.ngay_dat_hang", "table": "don_hang_ban", "source": "schema"}]
    }
"""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List

from . import config, schema_catalog
from .vn_text import normalize_identifier, normalize_vietnamese_text

# Single-token keys shorter/noisier than this are skipped to avoid matching noise.
_MIN_KEY_LEN = 3


def _add(out: Dict[str, List[dict]], key: str, entry: dict) -> None:
    key = key.strip()
    if len(key) < _MIN_KEY_LEN:
        return
    bucket = out.setdefault(key, [])
    # De-dup identical (type, identifier) entries; keep the first source seen.
    sig = (entry.get("type"), entry.get("identifier"))
    if any((e.get("type"), e.get("identifier")) == sig for e in bucket):
        return
    bucket.append(entry)


def build_alias_map(catalog: dict | None = None) -> Dict[str, List[dict]]:
    """Build the normalized alias map from the schema catalog."""
    catalog = catalog or schema_catalog.load_catalog()
    out: Dict[str, List[dict]] = {}
    for table, meta in catalog.get("tables", {}).items():
        table_entry = {"type": "table", "identifier": table, "source": "schema"}
        _add(out, normalize_identifier(table), table_entry)
        _add(out, normalize_identifier(table.replace("_", " ")), table_entry)
        for alias in meta.get("aliases", []) or []:
            _add(out, normalize_vietnamese_text(alias), {"type": "table", "identifier": table, "source": "alias"})

        for column in meta.get("columns", {}):
            qualified = f"{table}.{column}"
            col_entry = {"type": "column", "identifier": qualified, "table": table, "source": "schema"}
            _add(out, normalize_identifier(column), col_entry)
            _add(out, normalize_identifier(column.replace("_", " ")), col_entry)
    return out


def save_alias_map(alias_map: Dict[str, List[dict]], path: Path | None = None) -> Path:
    path = Path(path or config.ALIAS_MAP_PATH)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(alias_map, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def build_and_save(catalog: dict | None = None, path: Path | None = None) -> Path:
    out = save_alias_map(build_alias_map(catalog), path)
    load_alias_map.cache_clear()
    return out


@lru_cache(maxsize=1)
def load_alias_map(path: str | None = None) -> Dict[str, List[dict]]:
    p = Path(path or config.ALIAS_MAP_PATH)
    if not p.exists():
        return {}
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def lookup(query: str, alias_map: Dict[str, List[dict]] | None = None) -> List[dict]:
    """Return alias hits whose phrase appears (whole-word) in the question.

    Each hit carries the matched phrase's token length as ``phrase_len`` so callers
    can weight longer, more specific phrases (e.g. "khach hang dang hoat dong")
    above single tokens.
    """
    alias_map = alias_map if alias_map is not None else load_alias_map()
    if not alias_map:
        return []
    normalized = normalize_vietnamese_text(query)
    if not normalized:
        return []
    padded = f" {normalized} "
    hits: List[dict] = []
    for phrase, entries in alias_map.items():
        if f" {phrase} " in padded:
            phrase_len = len(phrase.split())
            for entry in entries:
                hits.append({**entry, "phrase": phrase, "phrase_len": phrase_len})
    # Longer phrase matches first (more specific intent).
    hits.sort(key=lambda h: h["phrase_len"], reverse=True)
    return hits
