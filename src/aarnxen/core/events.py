"""Lightweight event/webhook system for internal notifications."""
from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Callable
from collections import deque


@dataclass
class Event:
    type: str
    data: dict
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


class EventBus:
    def __init__(self, max_history: int = 1000):
        self._handlers: dict[str, list[Callable]] = {}
        self._history: deque = deque(maxlen=max_history)
        self._logger = logging.getLogger("aarnxen.events")

    def on(self, event_type: str, handler: Callable):
        """Register an event handler."""
        self._handlers.setdefault(event_type, []).append(handler)

    async def emit(self, event_type: str, data: dict):
        """Emit an event to all registered handlers."""
        event = Event(type=event_type, data=data)
        self._history.append(event)
        self._logger.debug("Event: %s %s", event_type, data)
        for handler in self._handlers.get(event_type, []):
            try:
                result = handler(event)
                if asyncio.iscoroutine(result):
                    await result
            except Exception:
                self._logger.warning("Event handler failed for %s", event_type, exc_info=True)
        # Also fire wildcard handlers
        for handler in self._handlers.get("*", []):
            try:
                result = handler(event)
                if asyncio.iscoroutine(result):
                    await result
            except Exception:
                pass

    def get_history(self, event_type: str = "", limit: int = 50) -> list[dict]:
        """Get recent events, optionally filtered by type."""
        events = list(self._history)
        if event_type:
            events = [e for e in events if e.type == event_type]
        return [{"type": e.type, "data": e.data, "timestamp": e.timestamp.isoformat()} for e in events[-limit:]]
