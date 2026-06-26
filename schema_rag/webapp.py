"""Local Vietnamese chat and DBMS UI."""
from __future__ import annotations

import json
from dataclasses import asdict
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from . import chat_memory, config, db_admin, pipeline, retriever, schema_catalog


COMMON_CSS = r"""
:root{--bg:#f7f7f5;--panel:#fff;--ink:#16171d;--muted:#667085;--line:#d9dde5;--accent:#1f6f55;--soft:#edf6f1;--danger:#9b1c1c}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font-family:Inter,ui-sans-serif,system-ui,"Segoe UI",sans-serif}
a{color:inherit;text-decoration:none}button,input,textarea,select{font:inherit}button{border:0;border-radius:6px;background:var(--accent);color:#fff;padding:9px 13px;font-weight:650;cursor:pointer}button.secondary{background:#eef2f6;color:#17202a;border:1px solid var(--line)}button.danger{background:var(--danger)}button:disabled{opacity:.55;cursor:wait}
.top{height:56px;background:var(--panel);border-bottom:1px solid var(--line);display:flex;align-items:center;justify-content:space-between;padding:0 18px}.brand{font-weight:750}.nav{display:flex;gap:10px}.nav a{padding:7px 10px;border-radius:6px}.nav a.active{background:var(--soft);color:var(--accent)}
.subtle{color:var(--muted);font-size:13px}.pill{display:inline-flex;align-items:center;gap:6px;border:1px solid #bfd2c8;background:#f1f8f4;color:#174934;border-radius:999px;padding:4px 8px;font-size:12px;margin:2px}
table{width:100%;border-collapse:collapse;background:#fff;font-size:12px}th,td{border-bottom:1px solid var(--line);text-align:left;padding:7px;vertical-align:top;overflow-wrap:anywhere}th{color:var(--muted);font-weight:650}
pre{white-space:pre-wrap;overflow:auto;background:#f8fafc;border:1px solid var(--line);border-radius:6px;padding:10px;max-height:320px;font-size:12px}.panel{background:var(--panel);border:1px solid var(--line);border-radius:8px}
"""


CHAT_HTML = r"""<!doctype html><html lang="vi"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Chat dữ liệu</title><style>
__CSS__
.layout{height:calc(100vh - 56px);display:grid;grid-template-columns:300px 1fr}.sessions{border-right:1px solid var(--line);background:#eef2f6;padding:14px;overflow:auto}.session{display:block;width:100%;text-align:left;background:#fff;color:var(--ink);border:1px solid var(--line);margin:8px 0;padding:9px;border-radius:6px}.session.active{border-color:var(--accent);background:var(--soft)}
.chat{display:grid;grid-template-rows:1fr auto;min-width:0}.messages{overflow:auto;padding:18px;display:grid;align-content:start;gap:13px}.msg{max-width:980px;border:1px solid var(--line);background:#fff;border-radius:8px;padding:13px}.msg.user{margin-left:auto;background:#eaf4ef;border-color:#c5ddcf;max-width:720px}.composer{border-top:1px solid var(--line);background:#fff;padding:14px;display:grid;grid-template-columns:1fr auto;gap:10px}textarea{min-height:48px;max-height:150px;resize:vertical;border:1px solid var(--line);border-radius:7px;padding:11px}
.candidates{display:grid;gap:10px;margin-top:10px}.candidate{border:1px solid var(--line);border-radius:7px;padding:10px;background:#fff}.row{display:flex;gap:8px;align-items:center;justify-content:space-between;flex-wrap:wrap}.actions{display:flex;gap:8px;align-items:center;flex-wrap:wrap}.details{display:none;margin-top:10px}.details.open{display:block}.debug{display:none}.debug.open{display:block}.result{margin-top:10px}
@media(max-width:900px){.layout{grid-template-columns:1fr}.sessions{display:none}.composer{grid-template-columns:1fr}}
</style></head><body>
<div class="top"><div class="brand">Trợ lý dữ liệu</div><div class="nav"><a class="active" href="/chat">Chat</a><a href="/db">Quản trị DB</a></div></div>
<div class="layout"><aside class="sessions"><button id="newChat">Cuộc trò chuyện mới</button><div id="sessions"></div></aside><main class="chat"><section id="messages" class="messages"></section><form id="form" class="composer"><textarea id="question" placeholder="Nhập câu hỏi dữ liệu..."></textarea><button id="send">Gửi</button></form></main></div>
<script>
let sessionId=localStorage.getItem("schema_rag_session")||"";let pendingQuestion="";
const messages=document.querySelector("#messages"),sessionsEl=document.querySelector("#sessions"),form=document.querySelector("#form"),question=document.querySelector("#question"),send=document.querySelector("#send");
function esc(v){return String(v??"").replace(/[&<>"']/g,c=>({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}[c]))}
function tableHtml(cols,rows){if(!rows||!rows.length)return"<div class='subtle'>Không có dữ liệu mẫu.</div>";cols=cols||Object.keys(rows[0]);return`<table><thead><tr>${cols.map(c=>`<th>${esc(c)}</th>`).join("")}</tr></thead><tbody>${rows.map(r=>`<tr>${cols.map((c,i)=>`<td>${esc(Array.isArray(r)?r[i]:r[c])}</td>`).join("")}</tr>`).join("")}</tbody></table>`}
function addMsg(role,html){const n=document.createElement("article");n.className="msg "+(role==="user"?"user":"assistant");n.innerHTML=html;messages.appendChild(n);messages.scrollTop=messages.scrollHeight;return n}
async function api(url,opts){const r=await fetch(url,opts);const d=await r.json();if(!r.ok)throw new Error(d.error||"Lỗi yêu cầu");return d}
async function ensureSession(){if(sessionId)return sessionId;const s=await api("/api/chat/session",{method:"POST"});sessionId=s.session_id;localStorage.setItem("schema_rag_session",sessionId);await loadSessions();return sessionId}
async function loadSessions(){const data=await api("/api/chat/sessions");sessionsEl.innerHTML=data.sessions.map(s=>`<button class="session ${s.session_id===sessionId?"active":""}" data-id="${esc(s.session_id)}"><b>${esc(s.title)}</b><br><span class="subtle">${esc(s.updated_at)}</span></button>`).join("");sessionsEl.querySelectorAll(".session").forEach(b=>b.onclick=()=>{sessionId=b.dataset.id;localStorage.setItem("schema_rag_session",sessionId);loadSession()})}
async function loadSession(){messages.innerHTML="";if(!sessionId)return;const data=await api(`/api/chat/session?id=${encodeURIComponent(sessionId)}`);data.messages.forEach(m=>{if(m.role==="user")addMsg("user",esc(m.content));else renderAnswer(m)});await loadSessions()}
function renderAnswer(data){const cols=data.result_columns||[];const rows=data.result_rows||[];const sql=data.sql||"";const selected=data.selected_tables||[];addMsg("assistant",`<div>${esc(data.answer||data.content||"")}</div><div>${selected.map(t=>`<span class="pill">${esc(t)}</span>`).join("")}</div>${rows.length?`<div class="result">${tableHtml(cols,rows)}</div>`:""}<button class="secondary" onclick="this.nextElementSibling.classList.toggle('open')">Chi tiết</button><div class="debug"><pre>${esc(sql||"Không có SQL")}</pre></div>`)}
function renderCandidates(data){const n=addMsg("assistant",`<b>Xác nhận bảng sẽ dùng</b><div class="subtle">Chỉ các bảng joined (jt_) được dùng cho chat. Bấm Xem thêm để kiểm tra dữ liệu mẫu.</div><div class="candidates"></div><div class="actions" style="margin-top:10px"><button class="run">Chạy truy vấn</button></div>`);const box=n.querySelector(".candidates");box.innerHTML=data.tables.map(t=>`<div class="candidate"><div class="row"><label><input type="checkbox" checked value="${esc(t.name)}"> <b>${esc(t.name)}</b></label><span class="subtle">${esc(t.row_count)} dòng</span></div><div class="subtle">${esc(t.description||"")}</div><button class="secondary more" type="button">Xem thêm</button><div class="details">${tableHtml(t.columns,t.sample_rows)}</div></div>`).join("");box.querySelectorAll(".more").forEach(b=>b.onclick=()=>b.nextElementSibling.classList.toggle("open"));n.querySelector(".run").onclick=async()=>{const tables=[...box.querySelectorAll("input:checked")].map(i=>i.value);if(!tables.length){alert("Chọn ít nhất một bảng.");return}send.disabled=true;try{const res=await api("/api/chat/run",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({session_id:sessionId,question:pendingQuestion,selected_tables:tables})});renderAnswer(res);await loadSessions()}catch(e){addMsg("assistant",`<span style="color:#9b1c1c">${esc(e.message)}</span>`)}finally{send.disabled=false}}}
form.onsubmit=async e=>{e.preventDefault();const text=question.value.trim();if(!text)return;pendingQuestion=text;question.value="";addMsg("user",esc(text));send.disabled=true;try{await ensureSession();const data=await api("/api/chat/retrieve",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({session_id:sessionId,question:text})});renderCandidates(data)}catch(err){addMsg("assistant",`<span style="color:#9b1c1c">${esc(err.message)}</span>`)}finally{send.disabled=false;question.focus()}};
document.querySelector("#newChat").onclick=async()=>{const s=await api("/api/chat/session",{method:"POST"});sessionId=s.session_id;localStorage.setItem("schema_rag_session",sessionId);messages.innerHTML="";await loadSessions()};
question.addEventListener("keydown",e=>{if(e.key==="Enter"&&!e.shiftKey){e.preventDefault();form.requestSubmit()}});
(async()=>{await loadSessions();if(sessionId)await loadSession();else await ensureSession()})();
</script></body></html>"""


DB_HTML = r"""<!doctype html><html lang="vi"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Quản trị DB</title><style>
__CSS__
.layout{height:calc(100vh - 56px);display:grid;grid-template-columns:320px 1fr}.left{border-right:1px solid var(--line);background:#eef2f6;padding:14px;overflow:auto}.main{padding:16px;overflow:auto}.tablebtn{display:block;width:100%;text-align:left;background:#fff;color:var(--ink);border:1px solid var(--line);margin:7px 0;padding:9px;border-radius:6px}.grid{display:grid;grid-template-columns:1fr 1fr;gap:12px}.sql{width:100%;min-height:160px;border:1px solid var(--line);border-radius:7px;padding:10px;font-family:ui-monospace,Consolas,monospace}.section{margin-bottom:14px}
@media(max-width:900px){.layout{grid-template-columns:1fr}.left{max-height:260px}.grid{grid-template-columns:1fr}}
</style></head><body>
<div class="top"><div class="brand">Quản trị cơ sở dữ liệu</div><div class="nav"><a href="/chat">Chat</a><a class="active" href="/db">Quản trị DB</a></div></div>
<div class="layout"><aside class="left"><button id="reload" class="secondary">Tải lại</button><div id="tables"></div></aside><main class="main"><div class="section panel" style="padding:12px"><h3 id="title">Chọn bảng</h3><div id="schema"></div><div id="rows"></div></div><div class="section panel" style="padding:12px"><h3>SQL editor</h3><textarea id="sql" class="sql" placeholder="SELECT * FROM jt_don_hang_day_du LIMIT 20;"></textarea><div style="margin-top:8px"><button id="run">Chạy SQL</button></div><div id="sqlResult"></div></div></main></div>
<script>
const tablesEl=document.querySelector("#tables"),schemaEl=document.querySelector("#schema"),rowsEl=document.querySelector("#rows"),title=document.querySelector("#title"),sqlEl=document.querySelector("#sql"),sqlResult=document.querySelector("#sqlResult");
function esc(v){return String(v??"").replace(/[&<>"']/g,c=>({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}[c]))}
function tableHtml(cols,rows){if(!rows||!rows.length)return"<div class='subtle'>Không có dòng.</div>";return`<table><thead><tr>${cols.map(c=>`<th>${esc(c)}</th>`).join("")}</tr></thead><tbody>${rows.map(r=>`<tr>${cols.map((c,i)=>`<td>${esc(Array.isArray(r)?r[i]:r[c])}</td>`).join("")}</tr>`).join("")}</tbody></table>`}
async function api(url,opts){const r=await fetch(url,opts);const d=await r.json();if(!r.ok)throw new Error(d.error||"Lỗi yêu cầu");return d}
async function loadTables(){const d=await api("/api/db/tables");tablesEl.innerHTML=d.tables.map(t=>`<button class="tablebtn" data-name="${esc(t.name)}"><b>${esc(t.name)}</b> ${t.chat_enabled?"<span class='pill'>chat</span>":""}<br><span class="subtle">${esc(t.type)} · ${esc(t.row_count)} dòng</span></button>`).join("");tablesEl.querySelectorAll("button").forEach(b=>b.onclick=()=>loadTable(b.dataset.name))}
async function loadTable(name){title.textContent=name;const s=await api(`/api/db/schema?name=${encodeURIComponent(name)}`);schemaEl.innerHTML=tableHtml(["cid","name","type","notnull","dflt_value","pk"],s.columns.map(c=>[c.cid,c.name,c.type,c.notnull,c.dflt_value,c.pk]));const r=await api(`/api/db/table?name=${encodeURIComponent(name)}&limit=50`);rowsEl.innerHTML="<h3>Dữ liệu mẫu</h3>"+tableHtml(r.columns,r.rows)}
document.querySelector("#run").onclick=async()=>{sqlResult.innerHTML="<div class='subtle'>Đang chạy...</div>";try{const d=await api("/api/db/sql",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({sql:sqlEl.value})});let html=`<div class="subtle">Hoàn tất. Backup: ${esc(d.backup_path||"không có")}</div>`;if(d.changed_tables?.length)html+=`<div>Đã cập nhật skill/index cho: ${d.changed_tables.map(esc).join(", ")}</div>`;if(d.removed_tables?.length)html+=`<div>Đã xóa skill của: ${d.removed_tables.map(esc).join(", ")}</div>`;if(d.columns?.length)html+=tableHtml(d.columns,d.rows);sqlResult.innerHTML=html;await loadTables()}catch(e){sqlResult.innerHTML=`<div style="color:#9b1c1c">${esc(e.message)}</div>`}};
document.querySelector("#reload").onclick=loadTables;loadTables();
</script></body></html>"""


def _html(template: str) -> bytes:
    return template.replace("__CSS__", COMMON_CSS).encode("utf-8")


def _catalog_table_summary(table: str) -> dict:
    catalog = schema_catalog.load_catalog()
    meta = catalog["tables"][table]
    rows = db_admin.table_rows(table, limit=3)
    return {
        "name": table,
        "description": meta.get("description", ""),
        "row_count": meta.get("row_count", 0),
        "columns": list(meta.get("columns", {}).keys()),
        "sample_rows": [dict(zip(rows["columns"], row)) for row in rows["rows"]],
    }


def chat_retrieve(session_id: str | None, question: str) -> dict:
    sid = chat_memory.ensure_session(session_id, question)
    history = chat_memory.compact_history(sid)
    r = retriever.retrieve(question, history_context=history, joined_only=True)
    tables = [_catalog_table_summary(t) for t in r.seed_tables]
    return {"session_id": sid, "question": question, "tables": tables, "table_scores": r.table_scores}


def chat_run(session_id: str | None, question: str, selected_tables: list[str]) -> dict:
    sid = chat_memory.ensure_session(session_id, question)
    history = chat_memory.compact_history(sid)
    chat_memory.add_user_message(sid, question)
    res = pipeline.ask(question, selected_tables=selected_tables, history_context=history)
    status = "error" if res.run_error else "ok"
    chat_memory.add_assistant_message(
        sid,
        res.answer,
        request_id=res.request_id,
        selected_tables=res.retrieval.expanded_tables,
        sql=res.sql,
        row_count=len(res.rows or []),
        status=status,
    )
    return {
        "session_id": sid,
        "request_id": res.request_id,
        "answer": res.answer,
        "selected_tables": res.retrieval.expanded_tables,
        "sql": res.sql,
        "result_columns": res.columns or [],
        "result_rows": res.rows or [],
        "run_error": res.run_error,
        "plan": res.plan,
        "sql_validation": asdict(res.validation) if res.validation else None,
    }


class Handler(BaseHTTPRequestHandler):
    def _json(self, status: int, payload: dict) -> None:
        body = json.dumps(payload, ensure_ascii=False, default=str).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_html(self, body: bytes) -> None:
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _payload(self) -> dict:
        length = int(self.headers.get("Content-Length", "0"))
        if length <= 0:
            return {}
        return json.loads(self.rfile.read(length).decode("utf-8"))

    def do_GET(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        qs = parse_qs(parsed.query)
        try:
            if parsed.path in {"/", "/index.html"}:
                self.send_response(302)
                self.send_header("Location", "/chat")
                self.end_headers()
                return
            if parsed.path == "/chat":
                self._send_html(_html(CHAT_HTML))
                return
            if parsed.path == "/db":
                self._send_html(_html(DB_HTML))
                return
            if parsed.path == "/api/chat/sessions":
                self._json(200, {"sessions": chat_memory.list_sessions()})
                return
            if parsed.path == "/api/chat/session":
                sid = (qs.get("id") or [""])[0]
                self._json(200, {"session_id": sid, "messages": chat_memory.messages(sid)})
                return
            if parsed.path == "/api/db/tables":
                self._json(200, {"tables": db_admin.list_tables()})
                return
            if parsed.path == "/api/db/schema":
                self._json(200, db_admin.table_schema((qs.get("name") or [""])[0]))
                return
            if parsed.path == "/api/db/table":
                self._json(
                    200,
                    db_admin.table_rows(
                        (qs.get("name") or [""])[0],
                        limit=int((qs.get("limit") or ["50"])[0]),
                        offset=int((qs.get("offset") or ["0"])[0]),
                    ),
                )
                return
            self.send_error(404)
        except Exception as exc:  # noqa: BLE001
            self._json(500, {"error": f"{exc.__class__.__name__}: {exc}"})

    def do_POST(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        try:
            payload = self._payload()
            if parsed.path == "/api/chat/session":
                self._json(200, chat_memory.create_session())
                return
            if parsed.path == "/api/chat/retrieve":
                question = str(payload.get("question", "")).strip()
                if not question:
                    self._json(400, {"error": "Câu hỏi không được để trống."})
                    return
                self._json(200, chat_retrieve(payload.get("session_id"), question))
                return
            if parsed.path == "/api/chat/run":
                question = str(payload.get("question", "")).strip()
                selected = [str(t) for t in payload.get("selected_tables", [])]
                if not question or not selected:
                    self._json(400, {"error": "Cần câu hỏi và ít nhất một bảng đã chọn."})
                    return
                self._json(200, chat_run(payload.get("session_id"), question, selected))
                return
            if parsed.path == "/api/db/sql":
                self._json(200, db_admin.execute_sql(str(payload.get("sql", ""))))
                return
            if parsed.path == "/api/db/rebuild-skills":
                catalog = schema_catalog.extract_catalog()
                schema_catalog.save_catalog(catalog)
                jt_tables = [t for t in catalog["tables"] if t.startswith("jt_")]
                files = db_admin.refresh_skills_for_tables(jt_tables)
                self._json(200, {"skill_files": files})
                return
            self.send_error(404)
        except Exception as exc:  # noqa: BLE001
            self._json(500, {"error": f"{exc.__class__.__name__}: {exc}"})

    def log_message(self, fmt: str, *args) -> None:
        print(f"[web] {self.address_string()} - {fmt % args}")


def serve(host: str = "127.0.0.1", port: int = 8000) -> None:
    if not Path(config.DB_PATH).exists():
        raise FileNotFoundError(f"Database not found at {config.DB_PATH}. Run: python -m schema_rag.cli setup")
    chat_memory.init()
    server = ThreadingHTTPServer((host, port), Handler)
    print(f"[web] open http://{host}:{port}/chat")
    server.serve_forever()


if __name__ == "__main__":
    serve()
