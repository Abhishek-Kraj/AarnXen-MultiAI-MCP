"""RAG Knowledge Base — SQLite FTS5 for fast full-text search + entity storage.

Stores documents, notes, facts, and code snippets that models can search
and reference across sessions. No external vector DB needed — uses SQLite's
built-in FTS5 for ranked text search.
"""

import sqlite3
import time
import uuid
from pathlib import Path
from typing import Optional


class KnowledgeBase:
    def __init__(self, db_path: str = "~/.aarnxen/knowledge.db"):
        self._path = Path(db_path).expanduser()
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self._path))
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._create_tables()

    def _create_tables(self):
        self._conn.executescript("""
            -- Main documents table
            CREATE TABLE IF NOT EXISTS documents (
                id TEXT PRIMARY KEY,
                title TEXT NOT NULL,
                content TEXT NOT NULL,
                doc_type TEXT DEFAULT 'note',
                tags TEXT DEFAULT '',
                source TEXT DEFAULT '',
                created_at REAL NOT NULL,
                updated_at REAL NOT NULL
            );

            -- FTS5 index for fast full-text search
            CREATE VIRTUAL TABLE IF NOT EXISTS documents_fts USING fts5(
                title, content, tags,
                content=documents,
                content_rowid=rowid
            );

            -- Triggers to keep FTS in sync
            CREATE TRIGGER IF NOT EXISTS documents_ai AFTER INSERT ON documents BEGIN
                INSERT INTO documents_fts(rowid, title, content, tags)
                VALUES (new.rowid, new.title, new.content, new.tags);
            END;

            CREATE TRIGGER IF NOT EXISTS documents_ad AFTER DELETE ON documents BEGIN
                INSERT INTO documents_fts(documents_fts, rowid, title, content, tags)
                VALUES ('delete', old.rowid, old.title, old.content, old.tags);
            END;

            CREATE TRIGGER IF NOT EXISTS documents_au AFTER UPDATE ON documents BEGIN
                INSERT INTO documents_fts(documents_fts, rowid, title, content, tags)
                VALUES ('delete', old.rowid, old.title, old.content, old.tags);
                INSERT INTO documents_fts(rowid, title, content, tags)
                VALUES (new.rowid, new.title, new.content, new.tags);
            END;

            -- Entities (people, projects, concepts, etc.)
            CREATE TABLE IF NOT EXISTS entities (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL UNIQUE,
                entity_type TEXT DEFAULT 'concept',
                description TEXT DEFAULT '',
                created_at REAL NOT NULL
            );

            -- Relations between entities
            CREATE TABLE IF NOT EXISTS relations (
                id TEXT PRIMARY KEY,
                from_entity TEXT NOT NULL,
                to_entity TEXT NOT NULL,
                relation_type TEXT NOT NULL,
                created_at REAL NOT NULL,
                FOREIGN KEY (from_entity) REFERENCES entities(id),
                FOREIGN KEY (to_entity) REFERENCES entities(id)
            );

            -- Observations / facts linked to entities
            CREATE TABLE IF NOT EXISTS observations (
                id TEXT PRIMARY KEY,
                entity_id TEXT NOT NULL,
                content TEXT NOT NULL,
                created_at REAL NOT NULL,
                FOREIGN KEY (entity_id) REFERENCES entities(id)
            );
        """)

    # --- Document Operations ---

    def store_document(
        self,
        title: str,
        content: str,
        doc_type: str = "note",
        tags: str = "",
        source: str = "",
    ) -> str:
        doc_id = str(uuid.uuid4())[:8]
        now = time.time()
        self._conn.execute(
            "INSERT INTO documents (id, title, content, doc_type, tags, source, created_at, updated_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (doc_id, title, content, doc_type, tags, source, now, now),
        )
        self._conn.commit()
        return doc_id

    def search_documents(self, query: str, limit: int = 10) -> list[dict]:
        """Full-text search using FTS5 ranking."""
        rows = self._conn.execute(
            "SELECT d.id, d.title, snippet(documents_fts, 1, '>>>', '<<<', '...', 64) as snippet, "
            "d.doc_type, d.tags, d.source, rank "
            "FROM documents_fts f "
            "JOIN documents d ON d.rowid = f.rowid "
            "WHERE documents_fts MATCH ? "
            "ORDER BY rank "
            "LIMIT ?",
            (query, limit),
        ).fetchall()
        return [
            {
                "id": r[0], "title": r[1], "snippet": r[2],
                "type": r[3], "tags": r[4], "source": r[5],
                "relevance_score": round(-r[6], 3),
            }
            for r in rows
        ]

    def get_document(self, doc_id: str) -> Optional[dict]:
        row = self._conn.execute(
            "SELECT id, title, content, doc_type, tags, source FROM documents WHERE id = ?",
            (doc_id,),
        ).fetchone()
        if not row:
            return None
        return {"id": row[0], "title": row[1], "content": row[2], "type": row[3], "tags": row[4], "source": row[5]}

    def delete_document(self, doc_id: str) -> bool:
        cursor = self._conn.execute("DELETE FROM documents WHERE id = ?", (doc_id,))
        self._conn.commit()
        return cursor.rowcount > 0

    def list_documents(self, doc_type: Optional[str] = None, limit: int = 50) -> list[dict]:
        if doc_type:
            rows = self._conn.execute(
                "SELECT id, title, doc_type, tags, created_at FROM documents WHERE doc_type = ? ORDER BY updated_at DESC LIMIT ?",
                (doc_type, limit),
            ).fetchall()
        else:
            rows = self._conn.execute(
                "SELECT id, title, doc_type, tags, created_at FROM documents ORDER BY updated_at DESC LIMIT ?",
                (limit,),
            ).fetchall()
        return [{"id": r[0], "title": r[1], "type": r[2], "tags": r[3]} for r in rows]

    # --- Entity Operations ---

    def add_entity(self, name: str, entity_type: str = "concept", description: str = "") -> str:
        eid = str(uuid.uuid4())[:8]
        self._conn.execute(
            "INSERT OR IGNORE INTO entities (id, name, entity_type, description, created_at) VALUES (?, ?, ?, ?, ?)",
            (eid, name, entity_type, description, time.time()),
        )
        self._conn.commit()
        return eid

    def add_relation(self, from_name: str, to_name: str, relation_type: str) -> str:
        # Auto-create entities if they don't exist
        self.add_entity(from_name)
        self.add_entity(to_name)

        from_id = self._conn.execute("SELECT id FROM entities WHERE name = ?", (from_name,)).fetchone()[0]
        to_id = self._conn.execute("SELECT id FROM entities WHERE name = ?", (to_name,)).fetchone()[0]

        rid = str(uuid.uuid4())[:8]
        self._conn.execute(
            "INSERT INTO relations (id, from_entity, to_entity, relation_type, created_at) VALUES (?, ?, ?, ?, ?)",
            (rid, from_id, to_id, relation_type, time.time()),
        )
        self._conn.commit()
        return rid

    def add_observation(self, entity_name: str, content: str) -> str:
        self.add_entity(entity_name)
        eid = self._conn.execute("SELECT id FROM entities WHERE name = ?", (entity_name,)).fetchone()[0]
        oid = str(uuid.uuid4())[:8]
        self._conn.execute(
            "INSERT INTO observations (id, entity_id, content, created_at) VALUES (?, ?, ?, ?)",
            (oid, eid, content, time.time()),
        )
        self._conn.commit()
        return oid

    def search_entities(self, query: str) -> list[dict]:
        rows = self._conn.execute(
            "SELECT e.id, e.name, e.entity_type, e.description FROM entities e WHERE e.name LIKE ? OR e.description LIKE ?",
            (f"%{query}%", f"%{query}%"),
        ).fetchall()
        results = []
        for r in rows:
            # Get observations
            obs = self._conn.execute(
                "SELECT content FROM observations WHERE entity_id = ? ORDER BY created_at DESC LIMIT 5",
                (r[0],),
            ).fetchall()
            # Get relations
            rels = self._conn.execute(
                "SELECT e2.name, rel.relation_type FROM relations rel "
                "JOIN entities e2 ON e2.id = rel.to_entity "
                "WHERE rel.from_entity = ?",
                (r[0],),
            ).fetchall()
            results.append({
                "name": r[1], "type": r[2], "description": r[3],
                "observations": [o[0] for o in obs],
                "relations": [{"to": rel[0], "type": rel[1]} for rel in rels],
            })
        return results

    def stats(self) -> dict:
        docs = self._conn.execute("SELECT COUNT(*) FROM documents").fetchone()[0]
        entities = self._conn.execute("SELECT COUNT(*) FROM entities").fetchone()[0]
        relations = self._conn.execute("SELECT COUNT(*) FROM relations").fetchone()[0]
        observations = self._conn.execute("SELECT COUNT(*) FROM observations").fetchone()[0]
        return {"documents": docs, "entities": entities, "relations": relations, "observations": observations}

    def close(self):
        self._conn.close()
