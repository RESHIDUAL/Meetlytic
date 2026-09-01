"""
Meeting Memory Database.
SQLite relational storage retaining exclusively validated professional meeting items,
topics, decisions, deadlines, and assigned responsibilities.
Personal/casual content is strictly excluded from all database schemas.
"""

import sqlite3
import json
import os
import time
from typing import List, Dict, Any, Optional

class MeetingDatabase:
    """
    SQLite Meeting Memory Storage.
    """

    def __init__(self, db_path: str = "meeting_records.db"):
        self.db_path = db_path
        self._init_db()

    def _get_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self):
        """Create tables if they do not exist."""
        with self._get_connection() as conn:
            cursor = conn.cursor()

            cursor.execute("""
            CREATE TABLE IF NOT EXISTS meetings (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_name TEXT,
                start_time TEXT NOT NULL,
                end_time TEXT,
                privacy_mode TEXT DEFAULT 'BALANCED',
                total_segments_analyzed INTEGER DEFAULT 0,
                professional_segments_retained INTEGER DEFAULT 0,
                discarded_segments_count INTEGER DEFAULT 0
            )
            """)

            cursor.execute("""
            CREATE TABLE IF NOT EXISTS meeting_items (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                meeting_id INTEGER NOT NULL,
                timestamp TEXT NOT NULL,
                category TEXT NOT NULL,  -- 'task', 'decision', 'deadline', 'requirement', 'project'
                content TEXT NOT NULL,
                speaker TEXT DEFAULT 'Speaker A',
                confidence REAL DEFAULT 1.0,
                keywords_json TEXT DEFAULT '[]',
                FOREIGN KEY (meeting_id) REFERENCES meetings(id) ON DELETE CASCADE
            )
            """)

            cursor.execute("""
            CREATE TABLE IF NOT EXISTS meeting_summaries (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                meeting_id INTEGER NOT NULL,
                generated_at TEXT NOT NULL,
                summary_text TEXT NOT NULL,
                topics_json TEXT DEFAULT '[]',
                decisions_json TEXT DEFAULT '[]',
                action_items_json TEXT DEFAULT '[]',
                deadlines_json TEXT DEFAULT '[]',
                people_json TEXT DEFAULT '[]',
                FOREIGN KEY (meeting_id) REFERENCES meetings(id) ON DELETE CASCADE
            )
            """)
            conn.commit()

    def start_meeting_session(self, session_name: str = "Meeting Session",
                              privacy_mode: str = "BALANCED") -> int:
        """Create a new meeting session record and return its meeting_id."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            start_time = time.strftime("%Y-%m-%d %H:%M:%S")
            cursor.execute("""
                INSERT INTO meetings (session_name, start_time, privacy_mode)
                VALUES (?, ?, ?)
            """, (session_name, start_time, privacy_mode))
            conn.commit()
            return int(cursor.lastrowid)

    def add_professional_item(self, meeting_id: int, category: str, content: str,
                              speaker: str = "Speaker A", confidence: float = 1.0,
                              keywords: Optional[List[str]] = None) -> int:
        """Insert a single extracted professional intelligence record."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            timestamp = time.strftime("%H:%M:%S")
            kw_json = json.dumps(keywords or [])
            cursor.execute("""
                INSERT INTO meeting_items (meeting_id, timestamp, category, content, speaker, confidence, keywords_json)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (meeting_id, timestamp, category, content, speaker, confidence, kw_json))
            conn.commit()
            return int(cursor.lastrowid)

    def update_meeting_metrics(self, meeting_id: int, total_segments: int,
                               retained_segments: int, discarded_segments: int):
        """Update meeting audit counters."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                UPDATE meetings
                SET total_segments_analyzed = ?,
                    professional_segments_retained = ?,
                    discarded_segments_count = ?
                WHERE id = ?
            """, (total_segments, retained_segments, discarded_segments, meeting_id))
            conn.commit()

    def end_meeting_session(self, meeting_id: int):
        """Mark meeting session as concluded."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            end_time = time.strftime("%Y-%m-%d %H:%M:%S")
            cursor.execute("UPDATE meetings SET end_time = ? WHERE id = ?", (end_time, meeting_id))
            conn.commit()

    def get_meeting_items(self, meeting_id: int, category: Optional[str] = None) -> List[Dict[str, Any]]:
        """Retrieve all stored professional items for a meeting."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            if category:
                cursor.execute("""
                    SELECT * FROM meeting_items
                    WHERE meeting_id = ? AND category = ?
                    ORDER BY id ASC
                """, (meeting_id, category))
            else:
                cursor.execute("""
                    SELECT * FROM meeting_items
                    WHERE meeting_id = ?
                    ORDER BY id ASC
                """, (meeting_id,))
            rows = cursor.fetchall()
            return [dict(r) for r in rows]

    def get_meeting(self, meeting_id: int) -> Optional[Dict[str, Any]]:
        """Fetch meeting row by ID."""
        return self.get_meeting_metadata(meeting_id)

    def get_meeting_metadata(self, meeting_id: int) -> Optional[Dict[str, Any]]:
        """Get top-level meeting info."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM meetings WHERE id = ?", (meeting_id,))
            row = cursor.fetchone()
            return dict(row) if row else None

    def save_summary(self, meeting_id: int, summary_text: str,
                     topics: List[str], decisions: List[str],
                     action_items: List[str], deadlines: List[str],
                     people: List[str]):
        """Persist structured final meeting summary."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            now = time.strftime("%Y-%m-%d %H:%M:%S")
            cursor.execute("""
                INSERT INTO meeting_summaries
                (meeting_id, generated_at, summary_text, topics_json, decisions_json, action_items_json, deadlines_json, people_json)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                meeting_id, now, summary_text,
                json.dumps(topics), json.dumps(decisions),
                json.dumps(action_items), json.dumps(deadlines),
                json.dumps(people)
            ))
            conn.commit()

    def get_latest_summary(self, meeting_id: int) -> Optional[Dict[str, Any]]:
        """Fetch generated summary for a meeting."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT * FROM meeting_summaries
                WHERE meeting_id = ?
                ORDER BY id DESC LIMIT 1
            """, (meeting_id,))
            row = cursor.fetchone()
            return dict(row) if row else None
