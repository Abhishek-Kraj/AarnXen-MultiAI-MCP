#!/usr/bin/env python3
"""
AarnXen KB Auto-Save — Command hook for Claude Code.
Runs on Stop/SessionEnd to save session summary to knowledge.db.

Parses the most recent JSONL session file, extracts user messages,
and writes a session summary directly to the KB.
"""

import glob
import hashlib
import json
import os
import sqlite3
import time
import uuid
import sys
from pathlib import Path

DB_PATH = os.path.expanduser("~/.aarnxen/knowledge.db")
CLAUDE_PROJECTS = os.path.expanduser("~/.claude/projects")
LOCK_SUFFIX = ".autosave.lock"


def get_project_dir():
    """Derive the Claude project dir from CWD."""
    cwd = os.getcwd()
    slug = cwd.replace("/", "-")
    project_path = os.path.join(CLAUDE_PROJECTS, slug)
    if os.path.isdir(project_path):
        return project_path
    # Try all project dirs and find best match
    for d in os.listdir(CLAUDE_PROJECTS):
        full = os.path.join(CLAUDE_PROJECTS, d)
        if os.path.isdir(full) and slug.endswith(d.lstrip("-")):
            return full
    return None


def get_latest_session(project_dir):
    """Find the most recently modified JSONL file."""
    files = glob.glob(os.path.join(project_dir, "*.jsonl"))
    if not files:
        return None
    return max(files, key=os.path.getmtime)


def _file_hash(filepath, tail_bytes=65536):
    """Fast hash of the tail of a file (captures recent changes)."""
    h = hashlib.sha256()
    size = os.path.getsize(filepath)
    with open(filepath, "rb") as f:
        if size > tail_bytes:
            f.seek(size - tail_bytes)
        h.update(f.read())
    h.update(str(size).encode())
    return h.hexdigest()[:16]


def is_already_saved(session_file):
    """Check lock file to avoid duplicate saves (hash-based)."""
    lock = session_file + LOCK_SUFFIX
    if os.path.exists(lock):
        try:
            with open(lock) as f:
                saved_hash = f.read().strip()
            current_hash = _file_hash(session_file)
            return saved_hash == current_hash
        except (ValueError, OSError):
            pass
    return False


def mark_saved(session_file):
    """Write lock file with content hash."""
    lock = session_file + LOCK_SUFFIX
    with open(lock, "w") as f:
        f.write(_file_hash(session_file))


def extract_session_info(session_file, max_lines=5000):
    """Extract user messages and basic stats from JSONL."""
    user_messages = []
    assistant_text_samples = []
    tool_calls = set()
    total_user = 0
    total_assistant = 0

    with open(session_file) as f:
        # Read from end for efficiency on large files
        lines = f.readlines()

    # Process last max_lines for large files
    process_lines = lines[-max_lines:] if len(lines) > max_lines else lines

    for line in process_lines:
        try:
            d = json.loads(line)
        except json.JSONDecodeError:
            continue

        entry_type = d.get("type", "")

        if entry_type == "user":
            total_user += 1
            msg = d.get("message", {})
            if isinstance(msg, dict):
                content = msg.get("content", "")
                if isinstance(content, str) and len(content.strip()) > 3:
                    user_messages.append(content.strip()[:300])
                elif isinstance(content, list):
                    for item in content:
                        if isinstance(item, dict) and item.get("type") == "text":
                            text = item.get("text", "").strip()
                            if len(text) > 3:
                                user_messages.append(text[:300])

        elif entry_type == "assistant":
            total_assistant += 1
            msg = d.get("message", {})
            if isinstance(msg, dict):
                content = msg.get("content", "")
                if isinstance(content, str) and len(content.strip()) > 10:
                    assistant_text_samples.append(content.strip()[:200])
                elif isinstance(content, list):
                    for item in content:
                        if isinstance(item, dict):
                            if item.get("type") == "text":
                                text = item.get("text", "").strip()
                                if len(text) > 10:
                                    assistant_text_samples.append(text[:200])
                            elif item.get("type") == "tool_use":
                                tool_calls.add(item.get("name", "unknown"))

    return {
        "user_messages": user_messages,
        "assistant_samples": assistant_text_samples[-10:],
        "tool_calls": sorted(tool_calls),
        "total_user": total_user,
        "total_assistant": total_assistant,
        "session_id": os.path.basename(session_file).replace(".jsonl", ""),
    }


def build_summary(info, project_name):
    """Build a concise session summary from extracted info."""
    user_msgs = info["user_messages"]
    if not user_msgs:
        return None, None

    # Get unique user requests (deduplicated, last 20)
    seen = set()
    unique_msgs = []
    for msg in user_msgs:
        key = msg[:50].lower()
        if key not in seen:
            seen.add(key)
            unique_msgs.append(msg)
    unique_msgs = unique_msgs[-20:]

    # Build title from first substantial message
    first_msg = next((m for m in unique_msgs if len(m) > 10), unique_msgs[0] if unique_msgs else "Session")
    title = f"[{project_name}] Session: {first_msg[:80]}"

    # Build content
    content_parts = [
        f"Session in {project_name} ({info['total_user']} user msgs, {info['total_assistant']} assistant msgs).",
        "",
        "User requests:",
    ]
    for msg in unique_msgs[-15:]:
        content_parts.append(f"- {msg[:200]}")

    if info["tool_calls"]:
        content_parts.append("")
        content_parts.append(f"Tools used: {', '.join(info['tool_calls'][:20])}")

    if info["assistant_samples"]:
        content_parts.append("")
        content_parts.append("Key responses:")
        for sample in info["assistant_samples"][-5:]:
            content_parts.append(f"- {sample[:200]}")

    content = "\n".join(content_parts)
    return title, content


def save_to_kb(title, content, project_name, session_id):
    """Write directly to knowledge.db."""
    if not os.path.exists(DB_PATH):
        print(f"DB not found: {DB_PATH}", file=sys.stderr)
        return False

    doc_id = str(uuid.uuid4())[:8]
    now = time.time()
    today = time.strftime("%Y-%m-%d")

    conn = sqlite3.connect(DB_PATH)
    try:
        conn.execute(
            """INSERT INTO documents (id, title, content, doc_type, tags, source, created_at, updated_at, importance)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                doc_id,
                title,
                content,
                "session",
                f"{project_name}, auto-save, {today}",
                "auto-save-hook",
                now,
                now,
                4.0,
            ),
        )
        conn.commit()
        print(f"Saved session {session_id[:8]}... as doc {doc_id} to KB", file=sys.stderr)
        return True
    except Exception as e:
        print(f"DB error: {e}", file=sys.stderr)
        return False
    finally:
        conn.close()


def main():
    cwd = os.getcwd()
    project_name = os.path.basename(cwd) or "unknown"

    project_dir = get_project_dir()
    if not project_dir:
        # Fallback: try all project dirs, find most recent JSONL
        all_jsonl = glob.glob(os.path.join(CLAUDE_PROJECTS, "*", "*.jsonl"))
        if not all_jsonl:
            return
        session_file = max(all_jsonl, key=os.path.getmtime)
        # Only process if modified in last 5 minutes
        if time.time() - os.path.getmtime(session_file) > 300:
            return
    else:
        session_file = get_latest_session(project_dir)

    if not session_file:
        return

    # Skip if already saved at this size
    if is_already_saved(session_file):
        return

    # Skip tiny sessions (< 10KB = probably just opened and closed)
    if os.path.getsize(session_file) < 10000:
        return

    info = extract_session_info(session_file)

    # Skip if fewer than 3 user messages (not a real session)
    if info["total_user"] < 3:
        return

    title, content = build_summary(info, project_name)
    if not title:
        return

    if save_to_kb(title, content, project_name, info["session_id"]):
        mark_saved(session_file)


if __name__ == "__main__":
    main()
