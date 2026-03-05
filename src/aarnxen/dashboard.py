"""AarnXen Knowledge Base Dashboard — local web UI.

Run with: python -m aarnxen.dashboard
Opens at: http://localhost:8765
"""

import json
import sqlite3
import time
from datetime import datetime, timezone
from http.server import HTTPServer, BaseHTTPRequestHandler
from pathlib import Path
from urllib.parse import parse_qs, urlparse

DB_PATH = Path("~/.aarnxen/knowledge.db").expanduser()
PORT = 8765


def get_db():
    conn = sqlite3.connect(str(DB_PATH), timeout=5)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.row_factory = sqlite3.Row
    return conn


def fmt_time(ts):
    if not ts:
        return ""
    try:
        return datetime.fromtimestamp(float(ts), tz=timezone.utc).strftime("%Y-%m-%d %H:%M")
    except (ValueError, TypeError, OSError):
        return str(ts)


# ── API handlers ──────────────────────────────────────────────

def api_stats():
    conn = get_db()
    docs = conn.execute("SELECT COUNT(*) c FROM documents").fetchone()["c"]
    entities = conn.execute("SELECT COUNT(*) c FROM entities").fetchone()["c"]
    relations = conn.execute("SELECT COUNT(*) c FROM relations").fetchone()["c"]
    observations = conn.execute("SELECT COUNT(*) c FROM observations").fetchone()["c"]
    doc_types = conn.execute(
        "SELECT doc_type, COUNT(*) c FROM documents GROUP BY doc_type ORDER BY c DESC"
    ).fetchall()
    entity_types = conn.execute(
        "SELECT entity_type, COUNT(*) c FROM entities GROUP BY entity_type ORDER BY c DESC"
    ).fetchall()
    conn.close()
    return {
        "documents": docs, "entities": entities,
        "relations": relations, "observations": observations,
        "doc_types": {r["doc_type"]: r["c"] for r in doc_types},
        "entity_types": {r["entity_type"]: r["c"] for r in entity_types},
    }


def api_documents(params):
    conn = get_db()
    doc_type = params.get("type", [""])[0]
    search = params.get("q", [""])[0]
    limit = min(int(params.get("limit", ["100"])[0]), 500)

    if search:
        terms = search.split()
        fts_query = " ".join(f'"{t}"' for t in terms)
        rows = conn.execute(
            "SELECT d.id, d.title, snippet(documents_fts, 1, '<mark>', '</mark>', '...', 64) as snippet, "
            "d.doc_type, d.tags, d.source, d.created_at, d.importance "
            "FROM documents_fts f JOIN documents d ON d.rowid = f.rowid "
            "WHERE documents_fts MATCH ? ORDER BY rank LIMIT ?",
            (fts_query, limit),
        ).fetchall()
    elif doc_type:
        rows = conn.execute(
            "SELECT id, title, substr(content, 1, 200) as snippet, doc_type, tags, source, created_at, importance "
            "FROM documents WHERE doc_type = ? ORDER BY updated_at DESC LIMIT ?",
            (doc_type, limit),
        ).fetchall()
    else:
        rows = conn.execute(
            "SELECT id, title, substr(content, 1, 200) as snippet, doc_type, tags, source, created_at, importance "
            "FROM documents ORDER BY updated_at DESC LIMIT ?",
            (limit,),
        ).fetchall()
    conn.close()
    return [
        {"id": r["id"], "title": r["title"], "snippet": r["snippet"],
         "type": r["doc_type"], "tags": r["tags"], "source": r["source"],
         "created_at": fmt_time(r["created_at"]), "importance": r["importance"]}
        for r in rows
    ]


def api_document_detail(doc_id):
    conn = get_db()
    row = conn.execute(
        "SELECT id, title, content, doc_type, tags, source, created_at, updated_at, importance, access_count "
        "FROM documents WHERE id = ?", (doc_id,)
    ).fetchone()
    conn.close()
    if not row:
        return {"error": "Not found"}
    return {
        "id": row["id"], "title": row["title"], "content": row["content"],
        "type": row["doc_type"], "tags": row["tags"], "source": row["source"],
        "created_at": fmt_time(row["created_at"]),
        "updated_at": fmt_time(row["updated_at"]),
        "importance": row["importance"], "access_count": row["access_count"],
    }


def api_entities(params):
    conn = get_db()
    search = params.get("q", [""])[0]
    if search:
        rows = conn.execute(
            "SELECT id, name, entity_type, description FROM entities "
            "WHERE name LIKE ? OR description LIKE ? ORDER BY name LIMIT 200",
            (f"%{search}%", f"%{search}%"),
        ).fetchall()
    else:
        rows = conn.execute(
            "SELECT id, name, entity_type, description FROM entities ORDER BY name LIMIT 200"
        ).fetchall()
    conn.close()
    return [{"id": r["id"], "name": r["name"], "type": r["entity_type"],
             "description": r["description"]} for r in rows]


def api_entity_detail(entity_id):
    conn = get_db()
    entity = conn.execute(
        "SELECT id, name, entity_type, description, created_at FROM entities WHERE id = ?",
        (entity_id,),
    ).fetchone()
    if not entity:
        conn.close()
        return {"error": "Not found"}

    observations = conn.execute(
        "SELECT id, content, obs_type, session_id, created_at FROM observations "
        "WHERE entity_id = ? ORDER BY created_at DESC LIMIT 50",
        (entity_id,),
    ).fetchall()

    relations_out = conn.execute(
        "SELECT r.id, e.name as target, r.relation_type, r.confidence "
        "FROM relations r JOIN entities e ON e.id = r.to_entity "
        "WHERE r.from_entity = ?",
        (entity_id,),
    ).fetchall()

    relations_in = conn.execute(
        "SELECT r.id, e.name as source, r.relation_type, r.confidence "
        "FROM relations r JOIN entities e ON e.id = r.from_entity "
        "WHERE r.to_entity = ?",
        (entity_id,),
    ).fetchall()
    conn.close()

    return {
        "id": entity["id"], "name": entity["name"],
        "type": entity["entity_type"], "description": entity["description"],
        "created_at": fmt_time(entity["created_at"]),
        "observations": [
            {"id": o["id"], "content": o["content"], "obs_type": o["obs_type"],
             "session_id": o["session_id"], "created_at": fmt_time(o["created_at"])}
            for o in observations
        ],
        "relations_out": [
            {"id": r["id"], "target": r["target"], "type": r["relation_type"],
             "confidence": r["confidence"]} for r in relations_out
        ],
        "relations_in": [
            {"id": r["id"], "source": r["source"], "type": r["relation_type"],
             "confidence": r["confidence"]} for r in relations_in
        ],
    }


def api_observations(params):
    conn = get_db()
    search = params.get("q", [""])[0]
    obs_type = params.get("type", [""])[0]
    limit = min(int(params.get("limit", ["50"])[0]), 200)

    conditions = []
    bind = []
    if search:
        conditions.append("(o.content LIKE ? OR e.name LIKE ?)")
        bind.extend([f"%{search}%", f"%{search}%"])
    if obs_type:
        conditions.append("o.obs_type = ?")
        bind.append(obs_type)
    where = (" WHERE " + " AND ".join(conditions)) if conditions else ""
    bind.append(limit)

    rows = conn.execute(
        f"SELECT o.id, e.name as entity, o.content, o.obs_type, o.session_id, o.created_at "
        f"FROM observations o JOIN entities e ON e.id = o.entity_id"
        f"{where} ORDER BY o.created_at DESC LIMIT ?",
        bind,
    ).fetchall()
    conn.close()
    return [
        {"id": r["id"], "entity": r["entity"], "content": r["content"],
         "obs_type": r["obs_type"], "session_id": r["session_id"],
         "created_at": fmt_time(r["created_at"])}
        for r in rows
    ]


def api_relations():
    conn = get_db()
    rows = conn.execute(
        "SELECT r.id, e1.name as from_name, e2.name as to_name, r.relation_type, r.confidence "
        "FROM relations r "
        "JOIN entities e1 ON e1.id = r.from_entity "
        "JOIN entities e2 ON e2.id = r.to_entity "
        "ORDER BY r.created_at DESC LIMIT 200"
    ).fetchall()
    conn.close()
    return [{"id": r["id"], "from": r["from_name"], "to": r["to_name"],
             "type": r["relation_type"], "confidence": r["confidence"]} for r in rows]


def api_delete_document(doc_id):
    conn = get_db()
    conn.execute("DELETE FROM documents WHERE id = ?", (doc_id,))
    conn.commit()
    conn.close()
    return {"deleted": doc_id}


# ── HTML Dashboard ────────────────────────────────────────────

DASHBOARD_HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>AarnXen Knowledge Base</title>
<style>
:root {
  --bg: #0d1117; --surface: #161b22; --border: #30363d;
  --text: #e6edf3; --text-dim: #8b949e; --accent: #58a6ff;
  --green: #3fb950; --red: #f85149; --orange: #d29922; --purple: #bc8cff;
  --tag-bg: #1f2937;
}
* { margin: 0; padding: 0; box-sizing: border-box; }
body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Helvetica, Arial, sans-serif;
  background: var(--bg); color: var(--text); line-height: 1.5; }
a { color: var(--accent); text-decoration: none; }
a:hover { text-decoration: underline; }

/* Layout */
.app { display: flex; min-height: 100vh; }
.sidebar { width: 220px; background: var(--surface); border-right: 1px solid var(--border);
  padding: 16px; position: fixed; height: 100vh; overflow-y: auto; }
.main { margin-left: 220px; flex: 1; padding: 24px; max-width: 1200px; }

/* Sidebar */
.logo { font-size: 18px; font-weight: 700; color: var(--accent); margin-bottom: 24px;
  display: flex; align-items: center; gap: 8px; }
.logo span { font-size: 22px; }
.nav-item { display: block; padding: 8px 12px; border-radius: 6px; color: var(--text-dim);
  cursor: pointer; margin-bottom: 2px; font-size: 14px; transition: all 0.15s; }
.nav-item:hover { background: var(--border); color: var(--text); }
.nav-item.active { background: rgba(88,166,255,0.15); color: var(--accent); }
.nav-section { font-size: 11px; text-transform: uppercase; color: var(--text-dim);
  margin: 16px 0 6px 12px; letter-spacing: 0.5px; }

/* Cards */
.stats-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(160px, 1fr));
  gap: 12px; margin-bottom: 24px; }
.stat-card { background: var(--surface); border: 1px solid var(--border); border-radius: 8px;
  padding: 16px; }
.stat-card .number { font-size: 28px; font-weight: 700; color: var(--accent); }
.stat-card .label { font-size: 12px; color: var(--text-dim); text-transform: uppercase; }

/* Search */
.search-bar { display: flex; gap: 8px; margin-bottom: 16px; }
.search-bar input { flex: 1; padding: 8px 12px; background: var(--surface); border: 1px solid var(--border);
  border-radius: 6px; color: var(--text); font-size: 14px; outline: none; }
.search-bar input:focus { border-color: var(--accent); }
.search-bar select { padding: 8px; background: var(--surface); border: 1px solid var(--border);
  border-radius: 6px; color: var(--text); font-size: 13px; }
.btn { padding: 8px 16px; background: var(--accent); color: #fff; border: none; border-radius: 6px;
  cursor: pointer; font-size: 13px; font-weight: 500; }
.btn:hover { opacity: 0.9; }
.btn-sm { padding: 4px 10px; font-size: 12px; }
.btn-danger { background: var(--red); }
.btn-ghost { background: transparent; border: 1px solid var(--border); color: var(--text-dim); }

/* Table */
table { width: 100%; border-collapse: collapse; }
th { text-align: left; padding: 10px 12px; font-size: 12px; text-transform: uppercase;
  color: var(--text-dim); border-bottom: 1px solid var(--border); }
td { padding: 10px 12px; border-bottom: 1px solid var(--border); font-size: 13px;
  vertical-align: top; }
tr:hover { background: rgba(88,166,255,0.04); }

/* Tags */
.tag { display: inline-block; padding: 2px 8px; background: var(--tag-bg); border-radius: 12px;
  font-size: 11px; color: var(--text-dim); margin: 1px 2px; }
.tag-type { background: rgba(188,140,255,0.15); color: var(--purple); }
.tag-bugfix { background: rgba(248,81,73,0.15); color: var(--red); }
.tag-feature { background: rgba(63,185,80,0.15); color: var(--green); }
.tag-decision { background: rgba(210,153,34,0.15); color: var(--orange); }
.tag-discovery { background: rgba(88,166,255,0.15); color: var(--accent); }

/* Detail panel */
.detail-panel { background: var(--surface); border: 1px solid var(--border); border-radius: 8px;
  padding: 20px; margin-bottom: 16px; }
.detail-panel h2 { margin-bottom: 12px; font-size: 18px; }
.detail-panel .meta { font-size: 12px; color: var(--text-dim); margin-bottom: 12px; }
.detail-panel .content { white-space: pre-wrap; font-size: 13px; line-height: 1.6;
  background: var(--bg); padding: 12px; border-radius: 6px; max-height: 400px; overflow-y: auto; }
.detail-panel mark { background: rgba(210,153,34,0.3); color: var(--text); padding: 0 2px; }

/* Entity graph */
.relations-list { display: flex; flex-wrap: wrap; gap: 8px; margin-top: 8px; }
.relation-badge { padding: 4px 10px; background: var(--bg); border: 1px solid var(--border);
  border-radius: 16px; font-size: 12px; display: flex; align-items: center; gap: 4px; }
.relation-badge .arrow { color: var(--accent); }

/* Observation cards */
.obs-card { background: var(--surface); border: 1px solid var(--border); border-radius: 8px;
  padding: 12px 16px; margin-bottom: 8px; }
.obs-card .obs-meta { font-size: 11px; color: var(--text-dim); margin-bottom: 4px;
  display: flex; gap: 8px; align-items: center; }
.obs-card .obs-content { font-size: 13px; }

/* Toast */
.toast { position: fixed; bottom: 20px; right: 20px; padding: 12px 20px;
  background: var(--green); color: #fff; border-radius: 8px; font-size: 13px;
  z-index: 1000; display: none; animation: fadeIn 0.3s; }
@keyframes fadeIn { from { opacity: 0; transform: translateY(10px); } to { opacity: 1; } }

.page-title { font-size: 22px; font-weight: 700; margin-bottom: 16px; }
.empty { text-align: center; padding: 40px; color: var(--text-dim); }
.badge-count { background: var(--border); padding: 1px 6px; border-radius: 10px;
  font-size: 11px; margin-left: 6px; }
</style>
</head>
<body>
<div class="app">
  <div class="sidebar">
    <div class="logo"><span>&#9672;</span> AarnXen KB</div>
    <div class="nav-section">Overview</div>
    <div class="nav-item active" onclick="showPage('dashboard')">Dashboard</div>
    <div class="nav-section">Knowledge</div>
    <div class="nav-item" onclick="showPage('documents')">Documents</div>
    <div class="nav-item" onclick="showPage('entities')">Entities</div>
    <div class="nav-item" onclick="showPage('observations')">Observations</div>
    <div class="nav-item" onclick="showPage('relations')">Relations</div>
    <div class="nav-section">Tools</div>
    <div class="nav-item" onclick="showPage('search')">Search</div>
  </div>
  <div class="main" id="content">
    <div class="empty">Loading...</div>
  </div>
</div>
<div class="toast" id="toast"></div>

<script>
const API = '';
let currentPage = 'dashboard';

async function api(path) {
  const r = await fetch(API + '/api/' + path);
  return r.json();
}

function showToast(msg) {
  const t = document.getElementById('toast');
  t.textContent = msg; t.style.display = 'block';
  setTimeout(() => t.style.display = 'none', 2500);
}

function tagClass(type) {
  const map = {bugfix:'tag-bugfix', feature:'tag-feature', decision:'tag-decision',
    discovery:'tag-discovery', refactor:'tag-type', change:'tag-type'};
  return map[type] || '';
}

function escapeHtml(s) {
  if (!s) return '';
  // Allow <mark> tags from FTS snippets
  return s.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;')
    .replace(/&lt;mark&gt;/g,'<mark>').replace(/&lt;\/mark&gt;/g,'</mark>');
}

// ── Pages ──

async function renderDashboard() {
  const stats = await api('stats');
  const recentDocs = await api('documents?limit=5');
  const recentObs = await api('observations?limit=5');
  return `
    <div class="page-title">Knowledge Base Dashboard</div>
    <div class="stats-grid">
      <div class="stat-card"><div class="number">${stats.documents}</div><div class="label">Documents</div></div>
      <div class="stat-card"><div class="number">${stats.entities}</div><div class="label">Entities</div></div>
      <div class="stat-card"><div class="number">${stats.observations}</div><div class="label">Observations</div></div>
      <div class="stat-card"><div class="number">${stats.relations}</div><div class="label">Relations</div></div>
    </div>
    <div style="display:grid; grid-template-columns:1fr 1fr; gap:16px;">
      <div>
        <h3 style="margin-bottom:8px;">Document Types</h3>
        <div class="detail-panel" style="padding:12px;">
          ${Object.entries(stats.doc_types).map(([k,v]) =>
            `<div style="display:flex;justify-content:space-between;padding:4px 0;">
              <span class="tag ${tagClass(k)}">${k}</span><span>${v}</span></div>`
          ).join('')}
        </div>
      </div>
      <div>
        <h3 style="margin-bottom:8px;">Entity Types</h3>
        <div class="detail-panel" style="padding:12px;">
          ${Object.entries(stats.entity_types).map(([k,v]) =>
            `<div style="display:flex;justify-content:space-between;padding:4px 0;">
              <span class="tag tag-type">${k}</span><span>${v}</span></div>`
          ).join('')}
        </div>
      </div>
    </div>
    <h3 style="margin:16px 0 8px;">Recent Documents</h3>
    <table>
      <tr><th>Title</th><th>Type</th><th>Date</th></tr>
      ${recentDocs.map(d => `
        <tr onclick="showDocDetail('${d.id}')" style="cursor:pointer">
          <td>${escapeHtml(d.title)}</td>
          <td><span class="tag ${tagClass(d.type)}">${d.type}</span></td>
          <td style="color:var(--text-dim)">${d.created_at}</td>
        </tr>`).join('')}
    </table>
    <h3 style="margin:16px 0 8px;">Recent Observations</h3>
    ${recentObs.map(o => `
      <div class="obs-card">
        <div class="obs-meta">
          <span class="tag ${tagClass(o.obs_type)}">${o.obs_type}</span>
          <span>${escapeHtml(o.entity)}</span>
          <span>${o.created_at}</span>
        </div>
        <div class="obs-content">${escapeHtml(o.content).substring(0, 200)}${o.content.length > 200 ? '...' : ''}</div>
      </div>
    `).join('')}
  `;
}

async function renderDocuments(params = '') {
  const docs = await api('documents' + (params ? '?' + params : ''));
  return `
    <div class="page-title">Documents <span class="badge-count">${docs.length}</span></div>
    <div class="search-bar">
      <input id="doc-search" placeholder="Search documents..." onkeydown="if(event.key==='Enter')searchDocs()">
      <select id="doc-type">
        <option value="">All types</option>
        <option value="note">note</option><option value="bugfix">bugfix</option>
        <option value="feature">feature</option><option value="decision">decision</option>
        <option value="discovery">discovery</option><option value="change">change</option>
        <option value="refactor">refactor</option><option value="fact">fact</option>
        <option value="code">code</option><option value="reference">reference</option>
      </select>
      <button class="btn" onclick="searchDocs()">Search</button>
    </div>
    <table>
      <tr><th>ID</th><th>Title</th><th>Type</th><th>Tags</th><th>Date</th></tr>
      ${docs.map(d => `
        <tr onclick="showDocDetail('${d.id}')" style="cursor:pointer">
          <td style="color:var(--text-dim);font-family:monospace">${d.id}</td>
          <td>${escapeHtml(d.title)}</td>
          <td><span class="tag ${tagClass(d.type)}">${d.type}</span></td>
          <td>${(d.tags||'').split(',').filter(Boolean).map(t => `<span class="tag">${t.trim()}</span>`).join(' ')}</td>
          <td style="color:var(--text-dim);white-space:nowrap">${d.created_at}</td>
        </tr>`).join('')}
    </table>
    ${docs.length === 0 ? '<div class="empty">No documents found</div>' : ''}
  `;
}

async function renderEntities(params = '') {
  const ents = await api('entities' + (params ? '?' + params : ''));
  return `
    <div class="page-title">Entities <span class="badge-count">${ents.length}</span></div>
    <div class="search-bar">
      <input id="ent-search" placeholder="Search entities..." onkeydown="if(event.key==='Enter')searchEntities()">
      <button class="btn" onclick="searchEntities()">Search</button>
    </div>
    <table>
      <tr><th>Name</th><th>Type</th><th>Description</th></tr>
      ${ents.map(e => `
        <tr onclick="showEntityDetail('${e.id}')" style="cursor:pointer">
          <td style="font-weight:600">${escapeHtml(e.name)}</td>
          <td><span class="tag tag-type">${e.type}</span></td>
          <td style="color:var(--text-dim)">${escapeHtml((e.description||'').substring(0, 100))}</td>
        </tr>`).join('')}
    </table>
    ${ents.length === 0 ? '<div class="empty">No entities found</div>' : ''}
  `;
}

async function renderObservations(params = '') {
  const obs = await api('observations' + (params ? '?' + params : ''));
  return `
    <div class="page-title">Observations <span class="badge-count">${obs.length}</span></div>
    <div class="search-bar">
      <input id="obs-search" placeholder="Search observations..." onkeydown="if(event.key==='Enter')searchObs()">
      <select id="obs-type">
        <option value="">All types</option>
        <option value="general">general</option><option value="bugfix">bugfix</option>
        <option value="feature">feature</option><option value="decision">decision</option>
        <option value="discovery">discovery</option>
      </select>
      <button class="btn" onclick="searchObs()">Search</button>
    </div>
    ${obs.map(o => `
      <div class="obs-card">
        <div class="obs-meta">
          <span class="tag ${tagClass(o.obs_type)}">${o.obs_type}</span>
          <a href="#" onclick="showEntityByName('${escapeHtml(o.entity)}');return false;" style="font-weight:600">${escapeHtml(o.entity)}</a>
          <span>${o.created_at}</span>
          ${o.session_id ? `<span class="tag">${o.session_id}</span>` : ''}
        </div>
        <div class="obs-content">${escapeHtml(o.content)}</div>
      </div>
    `).join('')}
    ${obs.length === 0 ? '<div class="empty">No observations found</div>' : ''}
  `;
}

async function renderRelations() {
  const rels = await api('relations');
  return `
    <div class="page-title">Relations <span class="badge-count">${rels.length}</span></div>
    <table>
      <tr><th>From</th><th>Relation</th><th>To</th><th>Confidence</th></tr>
      ${rels.map(r => `
        <tr>
          <td style="font-weight:600">${escapeHtml(r.from)}</td>
          <td><span class="tag tag-type">${r.type}</span></td>
          <td style="font-weight:600">${escapeHtml(r.to)}</td>
          <td>${r.confidence ? (r.confidence * 100).toFixed(0) + '%' : '-'}</td>
        </tr>`).join('')}
    </table>
    ${rels.length === 0 ? '<div class="empty">No relations found</div>' : ''}
  `;
}

async function renderSearch() {
  return `
    <div class="page-title">Search Knowledge Base</div>
    <div class="search-bar">
      <input id="global-search" placeholder="Search across all documents, entities, and observations..."
        onkeydown="if(event.key==='Enter')globalSearch()" style="font-size:16px;padding:12px;">
      <button class="btn" onclick="globalSearch()" style="padding:12px 24px;">Search</button>
    </div>
    <div id="search-results"></div>
  `;
}

// ── Detail Views ──

async function showDocDetail(id) {
  const d = await api('documents/' + id);
  if (d.error) { showToast(d.error); return; }
  document.getElementById('content').innerHTML = `
    <div style="margin-bottom:12px">
      <button class="btn btn-ghost" onclick="showPage('documents')">&larr; Back</button>
      <button class="btn btn-danger btn-sm" style="float:right" onclick="deleteDoc('${id}')">Delete</button>
    </div>
    <div class="detail-panel">
      <h2>${escapeHtml(d.title)}</h2>
      <div class="meta">
        <span class="tag ${tagClass(d.type)}">${d.type}</span> &middot;
        ID: <code>${d.id}</code> &middot;
        Created: ${d.created_at} &middot; Updated: ${d.updated_at} &middot;
        Importance: ${d.importance} &middot; Views: ${d.access_count}
        ${d.source ? ` &middot; Source: ${escapeHtml(d.source)}` : ''}
      </div>
      ${d.tags ? `<div style="margin-bottom:8px">${d.tags.split(',').map(t => `<span class="tag">${t.trim()}</span>`).join(' ')}</div>` : ''}
      <div class="content">${escapeHtml(d.content)}</div>
    </div>
  `;
}

async function showEntityDetail(id) {
  const e = await api('entities/' + id);
  if (e.error) { showToast(e.error); return; }
  document.getElementById('content').innerHTML = `
    <div style="margin-bottom:12px">
      <button class="btn btn-ghost" onclick="showPage('entities')">&larr; Back</button>
    </div>
    <div class="detail-panel">
      <h2>${escapeHtml(e.name)}</h2>
      <div class="meta">
        <span class="tag tag-type">${e.type}</span> &middot;
        Created: ${e.created_at}
        ${e.description ? ` &middot; ${escapeHtml(e.description)}` : ''}
      </div>
      ${e.relations_out.length || e.relations_in.length ? `
        <h3 style="margin:12px 0 6px;">Relations</h3>
        <div class="relations-list">
          ${e.relations_out.map(r => `<div class="relation-badge">${escapeHtml(e.name)} <span class="arrow">&rarr;</span> <span class="tag tag-type">${r.type}</span> <span class="arrow">&rarr;</span> ${escapeHtml(r.target)}</div>`).join('')}
          ${e.relations_in.map(r => `<div class="relation-badge">${escapeHtml(r.source)} <span class="arrow">&rarr;</span> <span class="tag tag-type">${r.type}</span> <span class="arrow">&rarr;</span> ${escapeHtml(e.name)}</div>`).join('')}
        </div>
      ` : ''}
      <h3 style="margin:16px 0 8px;">Observations (${e.observations.length})</h3>
      ${e.observations.map(o => `
        <div class="obs-card">
          <div class="obs-meta">
            <span class="tag ${tagClass(o.obs_type)}">${o.obs_type}</span>
            <span>${o.created_at}</span>
            ${o.session_id ? `<span class="tag">${o.session_id}</span>` : ''}
          </div>
          <div class="obs-content">${escapeHtml(o.content)}</div>
        </div>
      `).join('')}
      ${e.observations.length === 0 ? '<div class="empty">No observations</div>' : ''}
    </div>
  `;
}

async function showEntityByName(name) {
  const ents = await api('entities?q=' + encodeURIComponent(name));
  if (ents.length > 0) showEntityDetail(ents[0].id);
}

// ── Actions ──

function searchDocs() {
  const q = document.getElementById('doc-search').value;
  const t = document.getElementById('doc-type').value;
  const params = [];
  if (q) params.push('q=' + encodeURIComponent(q));
  if (t) params.push('type=' + encodeURIComponent(t));
  showPage('documents', params.join('&'));
}

function searchEntities() {
  const q = document.getElementById('ent-search').value;
  showPage('entities', q ? 'q=' + encodeURIComponent(q) : '');
}

function searchObs() {
  const q = document.getElementById('obs-search').value;
  const t = document.getElementById('obs-type').value;
  const params = [];
  if (q) params.push('q=' + encodeURIComponent(q));
  if (t) params.push('type=' + encodeURIComponent(t));
  showPage('observations', params.join('&'));
}

async function globalSearch() {
  const q = document.getElementById('global-search').value;
  if (!q) return;
  const [docs, ents, obs] = await Promise.all([
    api('documents?q=' + encodeURIComponent(q) + '&limit=10'),
    api('entities?q=' + encodeURIComponent(q)),
    api('observations?q=' + encodeURIComponent(q) + '&limit=10'),
  ]);
  document.getElementById('search-results').innerHTML = `
    <h3 style="margin:16px 0 8px;">Documents (${docs.length})</h3>
    ${docs.map(d => `
      <div class="obs-card" onclick="showDocDetail('${d.id}')" style="cursor:pointer">
        <div class="obs-meta"><span class="tag ${tagClass(d.type)}">${d.type}</span> <span>${d.created_at}</span></div>
        <div style="font-weight:600">${escapeHtml(d.title)}</div>
        <div class="obs-content" style="color:var(--text-dim)">${escapeHtml(d.snippet)}</div>
      </div>
    `).join('')}
    ${docs.length === 0 ? '<div class="empty">No documents found</div>' : ''}
    <h3 style="margin:16px 0 8px;">Entities (${ents.length})</h3>
    ${ents.map(e => `
      <div class="obs-card" onclick="showEntityDetail('${e.id}')" style="cursor:pointer">
        <div class="obs-meta"><span class="tag tag-type">${e.type}</span></div>
        <div style="font-weight:600">${escapeHtml(e.name)}</div>
      </div>
    `).join('')}
    ${ents.length === 0 ? '<div class="empty">No entities found</div>' : ''}
    <h3 style="margin:16px 0 8px;">Observations (${obs.length})</h3>
    ${obs.map(o => `
      <div class="obs-card">
        <div class="obs-meta"><span class="tag ${tagClass(o.obs_type)}">${o.obs_type}</span> <span>${escapeHtml(o.entity)}</span> <span>${o.created_at}</span></div>
        <div class="obs-content">${escapeHtml(o.content).substring(0, 300)}</div>
      </div>
    `).join('')}
    ${obs.length === 0 ? '<div class="empty">No observations found</div>' : ''}
  `;
}

async function deleteDoc(id) {
  if (!confirm('Delete this document?')) return;
  await fetch(API + '/api/documents/' + id, {method: 'DELETE'});
  showToast('Document deleted');
  showPage('documents');
}

// ── Router ──

async function showPage(page, params = '') {
  currentPage = page;
  document.querySelectorAll('.nav-item').forEach(n => n.classList.remove('active'));
  document.querySelectorAll('.nav-item').forEach(n => {
    if (n.textContent.trim().toLowerCase() === page) n.classList.add('active');
  });
  const el = document.getElementById('content');
  el.innerHTML = '<div class="empty">Loading...</div>';
  try {
    const renderers = {
      dashboard: renderDashboard,
      documents: () => renderDocuments(params),
      entities: () => renderEntities(params),
      observations: () => renderObservations(params),
      relations: renderRelations,
      search: renderSearch,
    };
    el.innerHTML = await (renderers[page] || renderDashboard)();
  } catch (e) {
    el.innerHTML = `<div class="empty">Error: ${e.message}</div>`;
  }
}

showPage('dashboard');
</script>
</body>
</html>"""


# ── HTTP Server ───────────────────────────────────────────────

class DashboardHandler(BaseHTTPRequestHandler):

    def do_GET(self):
        parsed = urlparse(self.path)
        path = parsed.path
        params = parse_qs(parsed.query)

        if path == "/" or path == "":
            self._html(DASHBOARD_HTML)
        elif path == "/api/stats":
            self._json(api_stats())
        elif path == "/api/documents":
            self._json(api_documents(params))
        elif path.startswith("/api/documents/"):
            doc_id = path.split("/")[-1]
            self._json(api_document_detail(doc_id))
        elif path == "/api/entities":
            self._json(api_entities(params))
        elif path.startswith("/api/entities/"):
            entity_id = path.split("/")[-1]
            self._json(api_entity_detail(entity_id))
        elif path == "/api/observations":
            self._json(api_observations(params))
        elif path == "/api/relations":
            self._json(api_relations())
        else:
            self.send_error(404)

    def do_DELETE(self):
        parsed = urlparse(self.path)
        if parsed.path.startswith("/api/documents/"):
            doc_id = parsed.path.split("/")[-1]
            self._json(api_delete_document(doc_id))
        else:
            self.send_error(404)

    def _json(self, data):
        body = json.dumps(data, default=str).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _html(self, html):
        body = html.encode()
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, fmt, *args):
        # Quiet logging — only errors
        if args and str(args[1]).startswith("4"):
            super().log_message(fmt, *args)


def main():
    if not DB_PATH.exists():
        print(f"Error: Knowledge base not found at {DB_PATH}")
        print("Run the AarnXen MCP server first to create it.")
        return

    server = HTTPServer(("127.0.0.1", PORT), DashboardHandler)
    print(f"\n  AarnXen Knowledge Base Dashboard")
    print(f"  ================================")
    print(f"  Running at: http://localhost:{PORT}")
    print(f"  Database:   {DB_PATH}")
    print(f"  Press Ctrl+C to stop\n")

    try:
        import webbrowser
        webbrowser.open(f"http://localhost:{PORT}")
    except Exception:
        pass

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down...")
        server.shutdown()


if __name__ == "__main__":
    main()
