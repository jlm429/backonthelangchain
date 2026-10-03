"""Bounded, in-process request limiting for provider-backed runs."""

from __future__ import annotations

import asyncio
import time
from collections import OrderedDict, deque


class InMemoryRateLimiter:
    """Limit calls per client while bounding tracked client memory."""

    def __init__(
        self,
        *,
        requests: int = 10,
        window_seconds: float = 60.0,
        max_clients: int = 10_000,
    ) -> None:
        self.requests = requests
        self.window_seconds = window_seconds
        self.max_clients = max_clients
        self._clients: OrderedDict[str, deque[float]] = OrderedDict()
        self._lock = asyncio.Lock()

    async def allow(self, client_id: str) -> bool:
        """Return whether this client may start another paid request."""
        now = time.monotonic()
        cutoff = now - self.window_seconds
        async with self._lock:
            attempts = self._clients.pop(client_id, deque())
            while attempts and attempts[0] <= cutoff:
                attempts.popleft()

            allowed = len(attempts) < self.requests
            if allowed:
                attempts.append(now)
            self._clients[client_id] = attempts

            while len(self._clients) > self.max_clients:
                self._clients.popitem(last=False)
            return allowed
