"""Foreign-key graph + join-path expansion.

This is the part the prompt-author insisted on: the vector retriever only PICKS
candidate tables. The actual JOINs come from real DB relationships modelled here.

We build an undirected graph whose nodes are tables and whose edges are FK links.
Given a set of seed tables, we connect them with shortest FK paths and return:
  * the expanded set of tables (seeds + any bridging tables)
  * the concrete join conditions (a.col = b.col) to put in the prompt
"""
from __future__ import annotations

from collections import defaultdict, deque
from typing import Dict, List, Optional, Set, Tuple

from . import schema_def

# edge key: frozenset({tableA, tableB}) -> (tableA.colA, tableB.colB)
_Edge = Tuple[str, str, str, str]  # (t1, c1, t2, c2)


def _build_adjacency() -> Tuple[Dict[str, Set[str]], Dict[frozenset, _Edge]]:
    adj: Dict[str, Set[str]] = defaultdict(set)
    edges: Dict[frozenset, _Edge] = {}
    for fk in schema_def.all_foreign_keys():
        a, b = fk["from_table"], fk["to_table"]
        if a == b:
            continue
        adj[a].add(b)
        adj[b].add(a)
        key = frozenset((a, b))
        # keep the first edge we see for a table pair (good enough for a star-ish schema)
        edges.setdefault(key, (a, fk["from_column"], b, fk["to_column"]))
    return adj, edges


_ADJ, _EDGES = _build_adjacency()


def neighbors(table: str) -> Set[str]:
    return set(_ADJ.get(table, set()))


def shortest_path(src: str, dst: str) -> Optional[List[str]]:
    """BFS shortest path of table names from src to dst (inclusive), or None."""
    if src == dst:
        return [src]
    seen = {src}
    queue: deque[List[str]] = deque([[src]])
    while queue:
        path = queue.popleft()
        for nxt in sorted(_ADJ.get(path[-1], ())):
            if nxt in seen:
                continue
            new_path = path + [nxt]
            if nxt == dst:
                return new_path
            seen.add(nxt)
            queue.append(new_path)
    return None


def join_condition(t1: str, t2: str) -> Optional[str]:
    """Return 't1.c1 = t2.c2' for a directly-connected table pair, else None."""
    edge = _EDGES.get(frozenset((t1, t2)))
    if not edge:
        return None
    a, ca, b, cb = edge
    return f"{a}.{ca} = {b}.{cb}"


def expand(seed_tables: List[str], max_tables: int = 12) -> Dict[str, object]:
    """Connect the seed tables via FK shortest paths.

    Returns dict with:
      tables        - ordered list of tables (seeds first, then bridges)
      join_edges    - list of {"left","right","on"} join conditions among final tables
      added_bridges - tables added purely to connect the seeds
    """
    seeds = list(dict.fromkeys(seed_tables))  # de-dup, keep order
    selected: List[str] = list(seeds)
    selected_set: Set[str] = set(seeds)

    # Connect every seed to the first seed (anchor); union all bridging tables.
    if len(seeds) > 1:
        anchor = seeds[0]
        for other in seeds[1:]:
            path = shortest_path(anchor, other)
            if not path:
                continue  # disconnected component - leave as-is
            for tbl in path:
                if tbl not in selected_set:
                    selected.append(tbl)
                    selected_set.add(tbl)

    # Safety cap.
    if len(selected) > max_tables:
        selected = selected[:max_tables]
        selected_set = set(selected)

    bridges = [t for t in selected if t not in seeds]

    # Collect every FK edge whose BOTH endpoints are in the final selection.
    join_edges: List[dict] = []
    seen_pairs: Set[frozenset] = set()
    for key, (a, ca, b, cb) in _EDGES.items():
        if a in selected_set and b in selected_set and key not in seen_pairs:
            seen_pairs.add(key)
            join_edges.append({"left": a, "right": b, "on": f"{a}.{ca} = {b}.{cb}"})

    return {"tables": selected, "join_edges": join_edges, "added_bridges": bridges}
