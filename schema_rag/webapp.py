"""Small local chat UI for testing the schema-RAG retriever."""
from __future__ import annotations

import json
import sqlite3
from dataclasses import asdict
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

from . import config, pipeline, schema_def


HTML = r"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Schema RAG Chat</title>
  <style>
    :root {
      color-scheme: light;
      --bg: #f6f7f9;
      --panel: #ffffff;
      --ink: #17202a;
      --muted: #667085;
      --line: #d9dee7;
      --accent: #1f7a5a;
      --accent-2: #245a8d;
      --warn: #9a5b00;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      background: var(--bg);
      color: var(--ink);
      letter-spacing: 0;
    }
    .app {
      min-height: 100vh;
      display: grid;
      grid-template-columns: minmax(280px, 360px) 1fr;
    }
    aside {
      border-right: 1px solid var(--line);
      background: #eef2f6;
      padding: 18px;
      overflow: auto;
    }
    main {
      min-width: 0;
      display: grid;
      grid-template-rows: auto 1fr auto;
      height: 100vh;
    }
    header {
      padding: 16px 20px;
      border-bottom: 1px solid var(--line);
      background: var(--panel);
      display: flex;
      justify-content: space-between;
      gap: 12px;
      align-items: center;
    }
    h1, h2, h3 { margin: 0; font-weight: 650; }
    h1 { font-size: 18px; }
    h2 { font-size: 14px; margin-bottom: 10px; }
    h3 { font-size: 13px; margin-bottom: 8px; }
    .subtle { color: var(--muted); font-size: 13px; }
    .schema-list {
      display: grid;
      gap: 8px;
      margin-top: 12px;
    }
    .table-row {
      border: 1px solid var(--line);
      background: var(--panel);
      padding: 9px 10px;
      border-radius: 6px;
      display: grid;
      gap: 3px;
    }
    .table-row code { font-size: 12px; color: var(--accent-2); }
    .messages {
      overflow: auto;
      padding: 18px 20px 24px;
      display: grid;
      align-content: start;
      gap: 14px;
    }
    .message {
      max-width: 1120px;
      border: 1px solid var(--line);
      background: var(--panel);
      border-radius: 8px;
      padding: 14px;
    }
    .user {
      margin-left: auto;
      background: #eaf4ef;
      border-color: #bddbcb;
      max-width: 760px;
    }
    .grid {
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 12px;
      margin-top: 12px;
    }
    .section {
      border-top: 1px solid var(--line);
      padding-top: 12px;
      margin-top: 12px;
    }
    .pillset {
      display: flex;
      flex-wrap: wrap;
      gap: 6px;
    }
    .pill {
      border: 1px solid #b7c8d9;
      background: #f2f7fb;
      color: #123b5d;
      border-radius: 999px;
      padding: 4px 8px;
      font-size: 12px;
      white-space: nowrap;
    }
    .seed { border-color: #aecfbf; background: #eef8f2; color: #14513a; }
    pre {
      margin: 0;
      white-space: pre-wrap;
      overflow-wrap: anywhere;
      font-size: 12px;
      line-height: 1.45;
      background: #f7f8fa;
      border: 1px solid var(--line);
      border-radius: 6px;
      padding: 10px;
      max-height: 280px;
      overflow: auto;
    }
    table {
      width: 100%;
      border-collapse: collapse;
      font-size: 12px;
      background: #fff;
    }
    th, td {
      border-bottom: 1px solid var(--line);
      text-align: left;
      padding: 7px;
      vertical-align: top;
      overflow-wrap: anywhere;
    }
    th { color: var(--muted); font-weight: 600; }
    .composer {
      border-top: 1px solid var(--line);
      background: var(--panel);
      padding: 14px 20px;
      display: grid;
      grid-template-columns: 1fr auto;
      gap: 10px;
    }
    textarea {
      width: 100%;
      resize: vertical;
      min-height: 48px;
      max-height: 160px;
      border: 1px solid var(--line);
      border-radius: 7px;
      padding: 12px;
      font: inherit;
      font-size: 14px;
    }
    button {
      border: 0;
      border-radius: 7px;
      background: var(--accent);
      color: white;
      min-width: 96px;
      padding: 0 16px;
      font: inherit;
      font-weight: 650;
      cursor: pointer;
    }
    button:disabled { opacity: .55; cursor: wait; }
    .metric-list {
      display: grid;
      gap: 6px;
      font-size: 13px;
    }
    .metric-list b { color: var(--accent); }
    @media (max-width: 880px) {
      .app { grid-template-columns: 1fr; }
      aside { display: none; }
      main { height: 100vh; }
      .grid { grid-template-columns: 1fr; }
      .composer { grid-template-columns: 1fr; }
      button { min-height: 42px; }
    }
  </style>
</head>
<body>
  <div class="app">
    <aside>
      <h2>Vietnamese demo schema</h2>
      <div class="subtle">20 tables based on FMCG sales, visits, customers, distributors, routes, delivery and returns.</div>
      <div id="schemaList" class="schema-list"></div>
    </aside>
    <main>
      <header>
        <div>
          <h1>Schema RAG Chat</h1>
          <div class="subtle">Ask a business question. The response shows every chosen table and a summary of matching data.</div>
        </div>
        <div class="subtle" id="status">Ready</div>
      </header>
      <section id="messages" class="messages"></section>
      <form id="form" class="composer">
        <textarea id="question" placeholder="Example: Which distributors have customers with falling order frequency?"></textarea>
        <button id="send" type="submit">Send</button>
      </form>
    </main>
  </div>
  <script>
    const schemaList = document.querySelector("#schemaList");
    const messages = document.querySelector("#messages");
    const form = document.querySelector("#form");
    const question = document.querySelector("#question");
    const send = document.querySelector("#send");
    const statusEl = document.querySelector("#status");

    function escapeHtml(value) {
      return String(value ?? "").replace(/[&<>"']/g, ch => ({
        "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;"
      }[ch]));
    }

    function renderTable(rows) {
      if (!rows || rows.length === 0) return "<div class='subtle'>No sample rows.</div>";
      const cols = Object.keys(rows[0]);
      return `<table><thead><tr>${cols.map(c => `<th>${escapeHtml(c)}</th>`).join("")}</tr></thead>` +
        `<tbody>${rows.map(r => `<tr>${cols.map(c => `<td>${escapeHtml(r[c])}</td>`).join("")}</tr>`).join("")}</tbody></table>`;
    }

    function addUser(text) {
      const node = document.createElement("article");
      node.className = "message user";
      node.textContent = text;
      messages.appendChild(node);
      messages.scrollTop = messages.scrollHeight;
    }

    function addAssistant(data) {
      const node = document.createElement("article");
      node.className = "message";
      const expanded = data.retrieval.expanded_tables.map(t => `<span class="pill">${escapeHtml(t)}</span>`).join("");
      const seeds = data.retrieval.seed_tables.map(t => `<span class="pill seed">${escapeHtml(t)}</span>`).join("");
      const joins = data.retrieval.join_edges.length
        ? data.retrieval.join_edges.map(e => escapeHtml(e.on)).join("\n")
        : "(no FK joins needed)";
      const tableBlocks = data.table_summaries.map(t => `
        <div class="section">
          <h3>${escapeHtml(t.table)} <span class="subtle">(${t.row_count} rows)</span></h3>
          <div class="subtle">${escapeHtml(t.description)}</div>
          <div class="metric-list" style="margin-top:8px">${t.metrics.map(m => `<div>${escapeHtml(m)}</div>`).join("")}</div>
          <div style="margin-top:8px">${renderTable(t.sample_rows)}</div>
        </div>`).join("");
      const plan = data.plan ? JSON.stringify(data.plan, null, 2) : "(no plan produced)";
      const sql = data.sql || "(no SQL produced)";
      const validationStatus = data.sql_validation ? (data.sql_validation.ok ? "OK" : "PROBLEMS") : "not run";
      const resultRows = data.result_rows || [];
      const resultTable = resultRows.length && data.result_columns.length
        ? renderTable(resultRows.map(row => Object.fromEntries(data.result_columns.map((c, i) => [c, row[i]]))))
        : `<div class='subtle'>${escapeHtml(data.answer || data.run_error || "No result rows.")}</div>`;
      node.innerHTML = `
        <div class="section">
          <h3>Answer</h3>
          <div>${escapeHtml(data.answer || data.run_error || "No executable SQL was produced.")}</div>
        </div>
        <div class="grid">
          <div class="section">
            <h3>Planner JSON</h3>
            <pre>${escapeHtml(plan)}</pre>
          </div>
          <div class="section">
            <h3>SQL <span class="subtle">validation: ${escapeHtml(validationStatus)}</span></h3>
            <pre>${escapeHtml(sql)}</pre>
          </div>
        </div>
        <div class="section">
          <h3>Result rows</h3>
          ${resultTable}
        </div>
        <div class="metric-list">
          ${data.database_summary.map(m => `<div>${m}</div>`).join("")}
        </div>
        <div class="grid">
          <div class="section">
            <h3>Seed tables from vector search</h3>
            <div class="pillset">${seeds}</div>
          </div>
          <div class="section">
            <h3>Expanded tables after FK graph</h3>
            <div class="pillset">${expanded}</div>
          </div>
        </div>
        <div class="section">
          <h3>Join paths</h3>
          <pre>${joins}</pre>
        </div>
        <div class="section">
          <h3>Data summary for chosen tables</h3>
          ${tableBlocks}
        </div>`;
      messages.appendChild(node);
      messages.scrollTop = messages.scrollHeight;
    }

    async function loadSchema() {
      const res = await fetch("/api/schema");
      const data = await res.json();
      schemaList.innerHTML = data.tables.map(t => `
        <div class="table-row">
          <code>${escapeHtml(t.name)}</code>
          <span class="subtle">${escapeHtml(t.columns.join(", "))}</span>
        </div>`).join("");
    }

    form.addEventListener("submit", async (event) => {
      event.preventDefault();
      const text = question.value.trim();
      if (!text) return;
      addUser(text);
      question.value = "";
      send.disabled = true;
      statusEl.textContent = "Retrieving";
      try {
        const res = await fetch("/api/chat", {
          method: "POST",
          headers: {"Content-Type": "application/json"},
          body: JSON.stringify({question: text})
        });
        const data = await res.json();
        if (!res.ok) throw new Error(data.error || "Request failed");
        addAssistant(data);
        statusEl.textContent = "Ready";
      } catch (err) {
        const node = document.createElement("article");
        node.className = "message";
        node.textContent = `Error: ${err.message}`;
        messages.appendChild(node);
        statusEl.textContent = "Error";
      } finally {
        send.disabled = false;
        question.focus();
      }
    });

    question.addEventListener("keydown", event => {
      if (event.key === "Enter" && !event.shiftKey) {
        event.preventDefault();
        form.requestSubmit();
      }
    });

    loadSchema();
  </script>
</body>
</html>
"""


def _connect() -> sqlite3.Connection:
    con = sqlite3.connect(config.DB_PATH)
    con.row_factory = sqlite3.Row
    return con


def _scalar(con: sqlite3.Connection, sql: str, default: Any = 0) -> Any:
    try:
        row = con.execute(sql).fetchone()
        return row[0] if row else default
    except sqlite3.Error:
        return default


def _database_summary() -> list[str]:
    with _connect() as con:
        total_tables = len(schema_def.TABLES)
        total_rows = sum(_scalar(con, f"SELECT COUNT(*) FROM {t['name']}") for t in schema_def.TABLES)
        customers = _scalar(con, "SELECT COUNT(*) FROM khach_hang")
        distributors = _scalar(con, "SELECT COUNT(*) FROM nha_phan_phoi")
        visits = _scalar(con, "SELECT COUNT(*) FROM lich_su_vieng_tham")
        orders = _scalar(con, "SELECT COUNT(*) FROM don_hang_ban")
        revenue = _scalar(con, "SELECT ROUND(SUM(tong_tien), 2) FROM don_hang_ban WHERE trang_thai = 'NORMAL'", 0)
    return [
        f"<b>{total_tables}</b> tables, <b>{total_rows}</b> rows in the Vietnamese FMCG demo database.",
        f"<b>{customers}</b> customers, <b>{distributors}</b> distributors, <b>{visits}</b> visits, <b>{orders}</b> sales orders.",
        f"Recorded normal-order revenue: <b>{revenue:,.0f}</b> VND.",
    ]


def _table_summary(con: sqlite3.Connection, table: str) -> dict:
    meta = schema_def.get_table(table)
    count = _scalar(con, f"SELECT COUNT(*) FROM {table}")
    rows = [dict(r) for r in con.execute(f"SELECT * FROM {table} LIMIT 3").fetchall()]
    metrics: list[str] = []

    cols = set(schema_def.columns_of(table))
    if "ngay_dat_hang" in cols:
        metrics.append(
            "date range: "
            + str(_scalar(con, f"SELECT MIN(ngay_dat_hang) || ' to ' || MAX(ngay_dat_hang) FROM {table}", "n/a"))
        )
    if "ngay_vieng_tham" in cols:
        metrics.append(
            "visit date range: "
            + str(_scalar(con, f"SELECT MIN(ngay_vieng_tham) || ' to ' || MAX(ngay_vieng_tham) FROM {table}", "n/a"))
        )
        top_result = con.execute(
            f"SELECT ket_qua, COUNT(*) AS n FROM {table} GROUP BY ket_qua ORDER BY n DESC LIMIT 3"
        ).fetchall()
        metrics.append("top visit results: " + ", ".join(f"{r['ket_qua']}={r['n']}" for r in top_result))
    if "tong_tien" in cols:
        sql = f"SELECT ROUND(SUM(tong_tien), 2) FROM {table} WHERE trang_thai = 'NORMAL'"
        metrics.append(f"total normal revenue: {_scalar(con, sql, 0):,.0f}")
    if "thanh_tien" in cols:
        metrics.append(f"line amount sum: {_scalar(con, f'SELECT ROUND(SUM(thanh_tien), 2) FROM {table}', 0):,.0f}")
    if "nha_phan_phoi_id" in cols:
        metrics.append(f"distinct distributors: {_scalar(con, f'SELECT COUNT(DISTINCT nha_phan_phoi_id) FROM {table}')}")
    if "khach_hang_id" in cols:
        metrics.append(f"distinct customers: {_scalar(con, f'SELECT COUNT(DISTINCT khach_hang_id) FROM {table}')}")
    if not metrics:
        metrics.append(f"columns: {', '.join(schema_def.columns_of(table))}")

    return {
        "table": table,
        "description": meta["description"],
        "row_count": count,
        "metrics": metrics,
        "sample_rows": rows,
    }


def chat(question: str) -> dict:
    res = pipeline.ask(question)
    r = res.retrieval
    with _connect() as con:
        summaries = [_table_summary(con, table) for table in r.expanded_tables]
    return {
        "request_id": res.request_id,
        "question": question,
        "answer": res.answer,
        "run_error": res.run_error,
        "plan": res.plan,
        "plan_validation": asdict(res.plan_validation) if res.plan_validation else None,
        "sql": res.sql,
        "sql_validation": asdict(res.validation) if res.validation else None,
        "result_columns": res.columns or [],
        "result_rows": res.rows or [],
        "warnings": res.warnings,
        "database_summary": _database_summary(),
        "retrieval": {
            "seed_tables": r.seed_tables,
            "expanded_tables": r.expanded_tables,
            "bridge_tables": r.bridge_tables,
            "join_edges": r.join_edges,
            "table_scores": r.table_scores,
        },
        "table_summaries": summaries,
    }


class Handler(BaseHTTPRequestHandler):
    def _json(self, status: int, payload: dict) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:  # noqa: N802
        if self.path == "/" or self.path == "/index.html":
            body = HTML.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        if self.path == "/api/schema":
            self._json(
                200,
                {
                    "tables": [
                        {"name": t["name"], "description": t["description"], "columns": [c["name"] for c in t["columns"]]}
                        for t in schema_def.TABLES
                    ]
                },
            )
            return
        self.send_error(404)

    def do_POST(self) -> None:  # noqa: N802
        if self.path != "/api/chat":
            self.send_error(404)
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            payload = json.loads(self.rfile.read(length).decode("utf-8"))
            question = str(payload.get("question", "")).strip()
            if not question:
                self._json(400, {"error": "Question is required."})
                return
            self._json(200, chat(question))
        except Exception as exc:  # noqa: BLE001
            self._json(500, {"error": f"{exc.__class__.__name__}: {exc}"})

    def log_message(self, fmt: str, *args: Any) -> None:
        print(f"[web] {self.address_string()} - {fmt % args}")


def serve(host: str = "127.0.0.1", port: int = 8000) -> None:
    if not Path(config.DB_PATH).exists():
        raise FileNotFoundError(f"Database not found at {config.DB_PATH}. Run: python -m schema_rag.cli setup")
    server = ThreadingHTTPServer((host, port), Handler)
    print(f"[web] open http://{host}:{port}")
    server.serve_forever()


if __name__ == "__main__":
    serve()
