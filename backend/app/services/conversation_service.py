import json
from datetime import datetime, timezone
from uuid import uuid4

from app.db.database import get_connection

# How many prior turns (user+assistant pairs) to feed back to the LLM
# as conversation context. Kept small deliberately: each included
# turn eats into the same context-window budget PromptBuilder already
# has to protect (see RESERVED_TOKENS_FOR_SYSTEM_AND_ANSWER), and more
# history isn't free just because it's cheap to store.
MAX_HISTORY_MESSAGES = 6


class ConversationService:
    """
    Persists chat turns to SQLite (conversations/messages tables) and
    hands back recent history for ChatService to fold into the next
    prompt. Retrieval itself still runs on the raw current question
    only -- condensing a follow-up like "what about her second job?"
    into a standalone retrieval query is a separate, not-yet-built
    piece (see README's conversation-memory roadmap note). This layer
    only makes the LLM's *generation* turn conversation-aware.
    """

    @staticmethod
    def create_conversation() -> str:
        conversation_id = str(uuid4())
        now = datetime.now(timezone.utc).isoformat()

        with get_connection() as conn:
            conn.execute(
                """
                INSERT INTO conversations (conversation_id, title, created_at, updated_at)
                VALUES (?, NULL, ?, ?)
                """,
                (conversation_id, now, now),
            )

        return conversation_id

    @staticmethod
    def conversation_exists(conversation_id: str) -> bool:
        with get_connection() as conn:
            row = conn.execute(
                "SELECT 1 FROM conversations WHERE conversation_id = ?",
                (conversation_id,),
            ).fetchone()
        return row is not None

    @classmethod
    def get_or_create(cls, conversation_id: str | None) -> str:
        if conversation_id and cls.conversation_exists(conversation_id):
            return conversation_id
        return cls.create_conversation()

    @staticmethod
    def add_message(
        conversation_id: str,
        role: str,
        content: str,
        sources: list[dict] | None = None,
    ) -> None:
        now = datetime.now(timezone.utc).isoformat()

        with get_connection() as conn:
            conn.execute(
                """
                INSERT INTO messages (message_id, conversation_id, role, content, sources_json, created_at)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    str(uuid4()),
                    conversation_id,
                    role,
                    content,
                    json.dumps(sources) if sources is not None else None,
                    now,
                ),
            )
            conn.execute(
                "UPDATE conversations SET updated_at = ? WHERE conversation_id = ?",
                (now, conversation_id),
            )

    @staticmethod
    def get_recent_messages(
        conversation_id: str, limit: int = MAX_HISTORY_MESSAGES
    ) -> list[dict]:
        """
        Returns up to `limit` most recent messages, oldest first (the
        order an LLM chat `messages` array needs) -- NOT including
        whatever question the caller is about to ask, since that
        hasn't been saved yet at the point ChatService needs history.
        """
        with get_connection() as conn:
            rows = conn.execute(
                """
                SELECT role, content, created_at FROM messages
                WHERE conversation_id = ?
                ORDER BY created_at DESC
                LIMIT ?
                """,
                (conversation_id, limit),
            ).fetchall()

        return [
            {"role": row["role"], "content": row["content"]}
            for row in reversed(rows)
        ]

    @staticmethod
    def get_full_history(conversation_id: str) -> list[dict]:
        with get_connection() as conn:
            rows = conn.execute(
                """
                SELECT role, content, sources_json, created_at FROM messages
                WHERE conversation_id = ?
                ORDER BY created_at ASC
                """,
                (conversation_id,),
            ).fetchall()

        return [
            {
                "role": row["role"],
                "content": row["content"],
                "sources": (
                    json.loads(row["sources_json"])
                    if row["sources_json"]
                    else None
                ),
                "created_at": row["created_at"],
            }
            for row in rows
        ]
