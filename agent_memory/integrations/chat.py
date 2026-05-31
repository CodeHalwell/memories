"""Framework-agnostic chat connector for wiring chatbots to the memory system.

This is the small reusable core that every chatbot platform binding (Discord,
Slack, Telegram, a web widget, …) needs, with **no platform dependency**:

  - map a platform user to an isolated memory ``namespace``,
  - recall relevant memories to inject as context before replying,
  - record user/assistant turns after they happen.

A platform binding is then a thin shim that receives a message, calls
:meth:`ChatConnector.handle_turn` (and :meth:`record_assistant_message` after
generating a reply), and does its own I/O. Because this layer is pure and
async, it is fully unit-testable and runs on any profile (incl. lite/edge).

Example (pseudo-binding)::

    connector = ChatConnector(service)

    async def on_message(msg):
        ctx = await connector.context_block(msg.author_id, msg.text)
        reply = await my_llm(system=ctx, user=msg.text)
        await connector.record_assistant_message(msg.author_id, reply)
        await channel.send(reply)
"""

from __future__ import annotations

from typing import Any

from agent_memory.service import MemoryService


class ChatConnector:
    """Maps platform users to namespaces and exposes recall/record helpers.

    Args:
        service: An initialized :class:`MemoryService`.
        namespace_prefix: Prefix for per-user namespaces (``"<prefix>:<user_id>"``),
            so different bots/platforms sharing a store don't collide.
        top_k: Default number of memories to recall.
    """

    def __init__(
        self,
        service: MemoryService,
        *,
        namespace_prefix: str = "chat",
        top_k: int | None = None,
    ) -> None:
        self._service = service
        self._namespace_prefix = namespace_prefix
        self._top_k = top_k

    # ── Identity mapping ──

    def namespace_for(self, user_id: str) -> str:
        """The isolated namespace for a platform user."""
        return f"{self._namespace_prefix}:{user_id}"

    def _session(self, user_id: str, session_id: str | None) -> str:
        # Default the session to the user's namespace so a user's history is
        # coherent across turns when the caller doesn't track sessions.
        return session_id or self.namespace_for(user_id)

    # ── Recall ──

    async def recall(
        self,
        user_id: str,
        query: str,
        *,
        session_id: str | None = None,
        top_k: int | None = None,
    ) -> list[dict[str, Any]]:
        """Return memories relevant to ``query`` for ``user_id`` (namespace-scoped)."""
        result = await self._service.retrieve(
            query=query,
            session_id=self._session(user_id, session_id),
            top_k=top_k if top_k is not None else self._top_k,
            namespace=self.namespace_for(user_id),
        )
        return result["memories"]

    async def context_block(
        self,
        user_id: str,
        query: str,
        *,
        session_id: str | None = None,
        top_k: int | None = None,
        header: str = "Relevant memory about this user:",
    ) -> str:
        """Recall memories and format them as a prompt-injectable text block.

        Returns an empty string when nothing relevant is found, so callers can
        unconditionally prepend the result to a system prompt.
        """
        memories = await self.recall(user_id, query, session_id=session_id, top_k=top_k)
        if not memories:
            return ""
        lines = [header]
        lines.extend(f"- {m['content']}" for m in memories)
        return "\n".join(lines)

    # ── Record ──

    async def record_user_message(
        self, user_id: str, text: str, *, session_id: str | None = None, turn: int = 0,
    ) -> dict[str, Any]:
        """Record an incoming user message as a candidate memory."""
        return await self._service.save_turn(
            content=text, session_id=self._session(user_id, session_id),
            turn=turn, role="user", namespace=self.namespace_for(user_id),
        )

    async def record_assistant_message(
        self, user_id: str, text: str, *, session_id: str | None = None, turn: int = 0,
    ) -> dict[str, Any]:
        """Record an assistant reply as a candidate memory."""
        return await self._service.save_turn(
            content=text, session_id=self._session(user_id, session_id),
            turn=turn, role="assistant", namespace=self.namespace_for(user_id),
        )

    # ── One-call convenience ──

    async def handle_turn(
        self,
        user_id: str,
        text: str,
        *,
        session_id: str | None = None,
        turn: int = 0,
        recall_query: str | None = None,
        top_k: int | None = None,
    ) -> dict[str, Any]:
        """Recall context for an incoming message and record it, in one call.

        Recall happens *before* recording so the returned context reflects prior
        turns only. Returns ``{namespace, session_id, context, recorded}``.
        """
        namespace = self.namespace_for(user_id)
        session = self._session(user_id, session_id)
        context = await self.recall(
            user_id, recall_query or text, session_id=session_id, top_k=top_k,
        )
        recorded = await self.record_user_message(
            user_id, text, session_id=session_id, turn=turn,
        )
        return {
            "namespace": namespace,
            "session_id": session,
            "context": context,
            "recorded": recorded,
        }
