"""Compact SQLite-backed chat session memory."""
from __future__ import annotations

import json
import sqlite3
import time
import uuid
from pathlib import Path
from typing import Any

from . import config


def _connect(db_path: Path | None = None) -> sqlite3.Connection:
    con = sqlite3.connect(db_path or config.DB_PATH)
    con.row_factory = sqlite3.Row
    return con


def init(db_path: Path | None = None) -> None:
    with _connect(db_path) as con:
        con.execute(
            """
            CREATE TABLE IF NOT EXISTS sys_chat_sessions (
                session_id TEXT PRIMARY KEY,
                title TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
            """
        )
        con.execute(
            """
            CREATE TABLE IF NOT EXISTS sys_chat_messages (
                message_id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                request_id TEXT,
                selected_tables_json TEXT,
                sql TEXT,
                row_count INTEGER,
                status TEXT,
                created_at TEXT NOT NULL,
                FOREIGN KEY (session_id) REFERENCES sys_chat_sessions(session_id)
            )
            """
        )
        con.execute("CREATE INDEX IF NOT EXISTS idx_sys_chat_messages_session ON sys_chat_messages(session_id, message_id)")


def _now() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def create_session(title: str | None = None) -> dict[str, Any]:
    init()
    session_id = f"chat_{uuid.uuid4().hex[:12]}"
    now = _now()
    title = (title or "Cuộc trò chuyện mới").strip()[:80] or "Cuộc trò chuyện mới"
    with _connect() as con:
        con.execute(
            "INSERT INTO sys_chat_sessions(session_id, title, created_at, updated_at) VALUES (?, ?, ?, ?)",
            (session_id, title, now, now),
        )
    return {"session_id": session_id, "title": title, "created_at": now, "updated_at": now}


def touch_title(session_id: str, title: str) -> None:
    with _connect() as con:
        con.execute(
            "UPDATE sys_chat_sessions SET title = ?, updated_at = ? WHERE session_id = ?",
            (title[:80], _now(), session_id),
        )


def ensure_session(session_id: str | None = None, title: str | None = None) -> str:
    init()
    if not session_id:
        return create_session(title)["session_id"]
    with _connect() as con:
        row = con.execute("SELECT session_id FROM sys_chat_sessions WHERE session_id = ?", (session_id,)).fetchone()
    if row:
        return session_id
    return create_session(title)["session_id"]


def list_sessions(limit: int = 30) -> list[dict[str, Any]]:
    init()
    with _connect() as con:
        rows = con.execute(
            """
            SELECT session_id, title, created_at, updated_at
            FROM sys_chat_sessions
            ORDER BY updated_at DESC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()
    return [dict(row) for row in rows]


def add_user_message(session_id: str, content: str) -> None:
    session_id = ensure_session(session_id, content)
    now = _now()
    with _connect() as con:
        con.execute(
            """
            INSERT INTO sys_chat_messages(session_id, role, content, status, created_at)
            VALUES (?, 'user', ?, 'ok', ?)
            """,
            (session_id, content, now),
        )
        con.execute("UPDATE sys_chat_sessions SET updated_at = ? WHERE session_id = ?", (now, session_id))
        count = con.execute(
            "SELECT COUNT(*) FROM sys_chat_messages WHERE session_id = ? AND role = 'user'",
            (session_id,),
        ).fetchone()[0]
        if int(count) == 1:
            con.execute(
                "UPDATE sys_chat_sessions SET title = ? WHERE session_id = ?",
                (content.strip()[:80] or "Cuộc trò chuyện mới", session_id),
            )


def add_assistant_message(
    session_id: str,
    content: str,
    request_id: str = "",
    selected_tables: list[str] | None = None,
    sql: str | None = None,
    row_count: int | None = None,
    status: str = "ok",
) -> None:
    session_id = ensure_session(session_id)
    now = _now()
    with _connect() as con:
        con.execute(
            """
            INSERT INTO sys_chat_messages(
                session_id, role, content, request_id, selected_tables_json, sql, row_count, status, created_at
            )
            VALUES (?, 'assistant', ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                session_id,
                content,
                request_id,
                json.dumps(selected_tables or [], ensure_ascii=False),
                sql,
                row_count,
                status,
                now,
            ),
        )
        con.execute("UPDATE sys_chat_sessions SET updated_at = ? WHERE session_id = ?", (now, session_id))


def messages(session_id: str, limit: int = 100) -> list[dict[str, Any]]:
    init()
    with _connect() as con:
        rows = con.execute(
            """
            SELECT message_id, role, content, request_id, selected_tables_json, sql, row_count, status, created_at
            FROM sys_chat_messages
            WHERE session_id = ?
            ORDER BY message_id ASC
            LIMIT ?
            """,
            (session_id, limit),
        ).fetchall()
    out: list[dict[str, Any]] = []
    for row in rows:
        item = dict(row)
        try:
            item["selected_tables"] = json.loads(item.pop("selected_tables_json") or "[]")
        except json.JSONDecodeError:
            item["selected_tables"] = []
        out.append(item)
    return out


def compact_history(session_id: str | None, turns: int | None = None) -> str:
    if not session_id:
        return ""
    turns = turns or config.CHAT_HISTORY_TURNS
    rows = messages(session_id, limit=turns * 2 + 4)[-(turns * 2) :]
    blocks: list[str] = []
    for row in rows:
        if row["role"] == "user":
            blocks.append(f"Người dùng: {row['content']}")
        else:
            bits = [f"Trợ lý: {row['content']}"]
            if row.get("selected_tables"):
                bits.append("Bảng đã dùng: " + ", ".join(row["selected_tables"]))
            if row.get("sql"):
                bits.append("SQL đã chạy: " + " ".join(str(row["sql"]).split())[:500])
            if row.get("row_count") is not None:
                bits.append(f"Số dòng trả về: {row['row_count']}")
            if row.get("status") and row["status"] != "ok":
                bits.append(f"Trạng thái: {row['status']}")
            blocks.append(" | ".join(bits))
    return "\n".join(blocks)
