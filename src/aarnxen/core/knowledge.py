"""RAG Knowledge Base — SQLite FTS5 for fast full-text search + entity storage.

Stores documents, notes, facts, and code snippets that models can search
and reference across sessions. No external vector DB needed — uses SQLite's
built-in FTS5 for ranked text search.
"""

import math
import sqlite3
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

_STOPWORDS = frozenset({
    "a", "an", "the", "and", "or", "but", "in", "on", "at", "to", "for",
    "of", "with", "by", "from", "is", "it", "as", "be", "was", "are",
    "been", "has", "have", "had", "do", "does", "did", "will", "would",
    "could", "should", "may", "might", "can", "this", "that", "these",
    "those", "not", "no", "if", "then", "so", "up", "out", "about",
    "into", "over", "after", "before", "between", "under", "above",
    "such", "each", "which", "their", "there", "than", "other", "its",
})


def _now_iso():
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _jaccard(a, b):
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def _tokenize(text):
    return {w.lower().strip(".,;:!?\"'()[]{}") for w in text.split() if len(w) > 2}


class KnowledgeBase:
    def __init__(self, db_path: str = "~/.aarnxen/knowledge.db"):
        self._path = Path(db_path).expanduser()
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self._path))
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._create_tables()
        self._migrate()

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

    def _has_column(self, table, column):
        cols = self._conn.execute(f"PRAGMA table_info({table})").fetchall()
        return any(c[1] == column for c in cols)

    def _migrate(self):
        # Documents: importance scoring columns
        migrations = [
            ("documents", "importance", "REAL DEFAULT 5.0"),
            ("documents", "access_count", "INTEGER DEFAULT 0"),
            ("documents", "last_accessed", "TEXT DEFAULT ''"),
            ("documents", "decay_rate", "REAL DEFAULT 1.0"),
            ("documents", "scope", "TEXT DEFAULT 'user'"),
            # Relations: confidence decay columns
            ("relations", "confidence", "REAL DEFAULT 1.0"),
            ("relations", "last_reinforced", "TEXT DEFAULT ''"),
            ("relations", "half_life_days", "REAL DEFAULT 30.0"),
        ]
        for table, column, coldef in migrations:
            if not self._has_column(table, column):
                self._conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {coldef}")
        self._conn.commit()

    # --- Document Operations ---

    def store_document(
        self,
        title: str,
        content: str,
        doc_type: str = "note",
        tags: str = "",
        source: str = "",
        importance: float = 5.0,
        scope: str = "user",
    ) -> str:
        doc_id = str(uuid.uuid4())[:8]
        now = time.time()
        now_iso = _now_iso()
        self._conn.execute(
            "INSERT INTO documents (id, title, content, doc_type, tags, source, created_at, updated_at, "
            "importance, access_count, last_accessed, decay_rate, scope) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 0, ?, 1.0, ?)",
            (doc_id, title, content, doc_type, tags, source, now, now, importance, now_iso, scope),
        )
        self._conn.commit()
        return doc_id

    def score_document(self, doc_id, query=None):
        row = self._conn.execute(
            "SELECT importance, access_count, last_accessed, decay_rate FROM documents WHERE id = ?",
            (doc_id,),
        ).fetchone()
        if not row:
            return 0.0
        importance, access_count, last_accessed, decay_rate = row

        # Relevance component
        if query:
            fts_row = self._conn.execute(
                "SELECT rank FROM documents_fts f JOIN documents d ON d.rowid = f.rowid "
                "WHERE documents_fts MATCH ? AND d.id = ?",
                (query, doc_id),
            ).fetchone()
            relevance = -fts_row[0] if fts_row else 0.0
            # Normalize: FTS5 rank is negative, closer to 0 = less relevant
            # Clamp to [0, 1] range
            relevance = min(relevance, 1.0)
        else:
            relevance = (importance or 5.0) / 10.0

        # Frequency component — saturating function
        frequency_factor = access_count / (1 + access_count)

        # Recency component — exponential decay
        if last_accessed:
            try:
                last_dt = datetime.fromisoformat(last_accessed.replace("Z", "+00:00"))
                days_since = (datetime.now(timezone.utc) - last_dt).total_seconds() / 86400
            except (ValueError, AttributeError):
                days_since = 30.0
        else:
            days_since = 30.0
        recency_factor = math.exp(-(decay_rate or 1.0) * days_since / 30)

        return round(relevance * 0.4 + frequency_factor * 0.3 + recency_factor * 0.3, 4)

    def search_documents(self, query: str, limit: int = 10, scope: Optional[str] = None) -> list[dict]:
        """Full-text search using FTS5 ranking with importance scoring."""
        if scope:
            rows = self._conn.execute(
                "SELECT d.id, d.title, snippet(documents_fts, 1, '>>>', '<<<', '...', 64) as snippet, "
                "d.doc_type, d.tags, d.source, rank "
                "FROM documents_fts f "
                "JOIN documents d ON d.rowid = f.rowid "
                "WHERE documents_fts MATCH ? AND d.scope = ? "
                "LIMIT ?",
                (query, scope, limit * 3),
            ).fetchall()
        else:
            rows = self._conn.execute(
                "SELECT d.id, d.title, snippet(documents_fts, 1, '>>>', '<<<', '...', 64) as snippet, "
                "d.doc_type, d.tags, d.source, rank "
                "FROM documents_fts f "
                "JOIN documents d ON d.rowid = f.rowid "
                "WHERE documents_fts MATCH ? "
                "LIMIT ?",
                (query, limit * 3),
            ).fetchall()

        # Score each result
        scored = []
        for r in rows:
            doc_id = r[0]
            score = self.score_document(doc_id, query=query)
            scored.append({
                "id": doc_id, "title": r[1], "snippet": r[2],
                "type": r[3], "tags": r[4], "source": r[5],
                "relevance_score": round(-r[6], 3),
                "combined_score": score,
            })

        # Sort by combined score descending
        scored.sort(key=lambda x: x["combined_score"], reverse=True)
        scored = scored[:limit]

        # Bump access stats for returned results
        now_iso = _now_iso()
        for doc in scored:
            self._conn.execute(
                "UPDATE documents SET access_count = access_count + 1, last_accessed = ? WHERE id = ?",
                (now_iso, doc["id"]),
            )
        self._conn.commit()

        return scored

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

    # --- Auto-Tagging ---

    def auto_tag(self, doc_id):
        row = self._conn.execute(
            "SELECT title, content FROM documents WHERE id = ?", (doc_id,),
        ).fetchone()
        if not row:
            return []
        text = f"{row[0]} {row[1]}"
        words = [w.lower().strip(".,;:!?\"'()[]{}") for w in text.split()]
        words = [w for w in words if len(w) > 2 and w not in _STOPWORDS]

        # Count frequencies
        freq = {}
        for w in words:
            freq[w] = freq.get(w, 0) + 1
        top = sorted(freq, key=freq.get, reverse=True)[:5]

        tags = ",".join(top)
        self._conn.execute("UPDATE documents SET tags = ? WHERE id = ?", (tags, doc_id))
        self._conn.commit()
        return top

    def search_by_tag(self, tag, limit=20):
        rows = self._conn.execute(
            "SELECT id, title, doc_type, tags, source FROM documents "
            "WHERE ',' || tags || ',' LIKE ? ORDER BY updated_at DESC LIMIT ?",
            (f"%,{tag},%", limit),
        ).fetchall()
        # Also match tags at start/end
        rows2 = self._conn.execute(
            "SELECT id, title, doc_type, tags, source FROM documents "
            "WHERE tags LIKE ? OR tags LIKE ? OR tags = ? ORDER BY updated_at DESC LIMIT ?",
            (f"{tag},%", f"%,{tag}", tag, limit),
        ).fetchall()
        seen = set()
        results = []
        for r in list(rows) + list(rows2):
            if r[0] not in seen:
                seen.add(r[0])
                results.append({"id": r[0], "title": r[1], "type": r[2], "tags": r[3], "source": r[4]})
        return results[:limit]

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
        now_iso = _now_iso()
        self._conn.execute(
            "INSERT INTO relations (id, from_entity, to_entity, relation_type, created_at, "
            "confidence, last_reinforced, half_life_days) VALUES (?, ?, ?, ?, ?, 1.0, ?, 30.0)",
            (rid, from_id, to_id, relation_type, time.time(), now_iso),
        )
        self._conn.commit()
        return rid

    def reinforce_relation(self, relation_id):
        now_iso = _now_iso()
        self._conn.execute(
            "UPDATE relations SET confidence = 1.0, last_reinforced = ? WHERE id = ?",
            (now_iso, relation_id),
        )
        self._conn.commit()

    def _effective_confidence(self, confidence, last_reinforced, half_life_days):
        if not last_reinforced:
            return confidence
        try:
            last_dt = datetime.fromisoformat(last_reinforced.replace("Z", "+00:00"))
            days_since = (datetime.now(timezone.utc) - last_dt).total_seconds() / 86400
        except (ValueError, AttributeError):
            return confidence
        half_life = half_life_days or 30.0
        return confidence * (0.5 ** (days_since / half_life))

    def get_relations_with_decay(self, entity_name):
        eid_row = self._conn.execute("SELECT id FROM entities WHERE name = ?", (entity_name,)).fetchone()
        if not eid_row:
            return []
        eid = eid_row[0]
        rows = self._conn.execute(
            "SELECT rel.id, e2.name, rel.relation_type, rel.confidence, rel.last_reinforced, rel.half_life_days "
            "FROM relations rel "
            "JOIN entities e2 ON e2.id = rel.to_entity "
            "WHERE rel.from_entity = ?",
            (eid,),
        ).fetchall()
        results = []
        for r in rows:
            eff = self._effective_confidence(r[3] or 1.0, r[4], r[5] or 30.0)
            if eff >= 0.1:
                results.append({
                    "id": r[0], "to": r[1], "type": r[2],
                    "confidence": round(r[3] or 1.0, 3),
                    "effective_confidence": round(eff, 3),
                })
        return results

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

    # --- Memory Deduplication & Consolidation ---

    def deduplicate(self, similarity_threshold=0.8):
        rows = self._conn.execute(
            "SELECT id, title, content, importance FROM documents ORDER BY importance DESC",
        ).fetchall()

        merged_count = 0
        deleted_ids = set()

        for i in range(len(rows)):
            if rows[i][0] in deleted_ids:
                continue
            tokens_i = _tokenize(f"{rows[i][1]} {rows[i][2]}")
            for j in range(i + 1, len(rows)):
                if rows[j][0] in deleted_ids:
                    continue
                tokens_j = _tokenize(f"{rows[j][1]} {rows[j][2]}")
                sim = _jaccard(tokens_i, tokens_j)
                if sim >= similarity_threshold:
                    # Keep rows[i] (higher importance due to ORDER BY), merge content from rows[j]
                    keep_id = rows[i][0]
                    drop_id = rows[j][0]
                    # Append unique content from the duplicate
                    keep_doc = self.get_document(keep_id)
                    drop_doc = self.get_document(drop_id)
                    if keep_doc and drop_doc:
                        keep_words = set(keep_doc["content"].split())
                        drop_words = drop_doc["content"].split()
                        unique = [w for w in drop_words if w not in keep_words]
                        if unique:
                            merged_content = keep_doc["content"] + " " + " ".join(unique)
                            self._conn.execute(
                                "UPDATE documents SET content = ?, updated_at = ? WHERE id = ?",
                                (merged_content, time.time(), keep_id),
                            )
                    self._conn.execute("DELETE FROM documents WHERE id = ?", (drop_id,))
                    deleted_ids.add(drop_id)
                    merged_count += 1

        self._conn.commit()
        return merged_count

    def consolidate(self):
        # Run deduplication
        merged = self.deduplicate()

        # Decay all relation confidences and remove very weak ones
        rows = self._conn.execute(
            "SELECT id, confidence, last_reinforced, half_life_days FROM relations",
        ).fetchall()
        removed_relations = 0
        for r in rows:
            eff = self._effective_confidence(r[1] or 1.0, r[2], r[3] or 30.0)
            if eff < 0.05:
                self._conn.execute("DELETE FROM relations WHERE id = ?", (r[0],))
                removed_relations += 1

        # Remove old observations with no reinforcement (>90 days)
        cutoff = time.time() - (90 * 86400)
        cursor = self._conn.execute(
            "DELETE FROM observations WHERE created_at < ?", (cutoff,),
        )
        removed_observations = cursor.rowcount

        self._conn.commit()
        return {
            "merged_documents": merged,
            "removed_relations": removed_relations,
            "removed_observations": removed_observations,
        }

    # --- Stats ---

    def stats(self) -> dict:
        docs = self._conn.execute("SELECT COUNT(*) FROM documents").fetchone()[0]
        entities = self._conn.execute("SELECT COUNT(*) FROM entities").fetchone()[0]
        relations = self._conn.execute("SELECT COUNT(*) FROM relations").fetchone()[0]
        observations = self._conn.execute("SELECT COUNT(*) FROM observations").fetchone()[0]
        return {"documents": docs, "entities": entities, "relations": relations, "observations": observations}

    def close(self):
        self._conn.close()
