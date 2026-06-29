"""Reciprocal Rank Fusion for merging multiple ranked candidate lists.

RRF combines rankings from independent retrievers (vector, BM25, ...) without
needing their raw scores to be on the same scale - it only uses the *rank* of each
item in each list. See plan §11.5.

    score(item) = sum over lists of 1 / (k + rank_in_list)   (rank is 1-based)

Optional ``boosts`` add a flat bonus per item (used for strong signals like exact
alias matches that should outrank ordinary semantic hits).
"""
from __future__ import annotations

from typing import Dict, Iterable, List


def rrf_score(rank: int, k: int = 60) -> float:
    """RRF contribution for a 1-based rank position."""
    return 1.0 / (k + rank)


def fuse(
    rankings: Iterable[List[str]],
    k: int = 60,
    boosts: Dict[str, float] | None = None,
) -> Dict[str, float]:
    """Fuse several best-first ranked lists of item keys into one score map.

    Items appearing only in ``boosts`` (e.g. an exact alias match that was not in
    any retriever's top-k) still receive their boost, so strong lexical signals are
    never dropped.
    """
    scores: Dict[str, float] = {}
    for ranking in rankings:
        for position, item in enumerate(ranking, start=1):
            scores[item] = scores.get(item, 0.0) + rrf_score(position, k)
    if boosts:
        for item, bonus in boosts.items():
            scores[item] = scores.get(item, 0.0) + bonus
    return scores
