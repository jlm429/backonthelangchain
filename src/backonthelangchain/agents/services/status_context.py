"""Build explicit demo evidence from user reports and simulated system state."""

from __future__ import annotations

import re
from typing import Any

from backonthelangchain.agents.schemas import SystemName, SystemStatusLevel

SYSTEM_ALIASES: dict[SystemName, tuple[str, ...]] = {
    "authentication": ("authentication", "auth", "login", "log in", "sign in", "mfa"),
    "billing": ("billing", "invoice", "charge", "subscription", "payment"),
    "checkout": ("checkout", "purchase", "cart", "order"),
    "api": ("api", "endpoint", "webhook", "request"),
}

OUTAGE_MARKERS = (
    "down",
    "outage",
    "unavailable",
    "not working",
    "failing",
    "failure",
    "broken",
    "error",
    "cannot",
    "can't",
    "unable",
)


def _contains_phrase(text: str, phrase: str) -> bool:
    return re.search(rf"\b{re.escape(phrase)}\b", text) is not None


def build_simulated_status_evidence(
    user_query: str,
    statuses: dict[SystemName, SystemStatusLevel],
) -> dict[str, Any]:
    """Compare an outage report with explicitly simulated demo state."""
    normalized_query = user_query.casefold()
    reports_problem = any(
        _contains_phrase(normalized_query, marker) for marker in OUTAGE_MARKERS
    )
    mentioned = [
        system
        for system, aliases in SYSTEM_ALIASES.items()
        if any(_contains_phrase(normalized_query, alias) for alias in aliases)
    ]

    reports: list[dict[str, Any]] = []
    if reports_problem:
        if not mentioned:
            reports.append(
                {
                    "system": "unspecified",
                    "user_reported_problem": True,
                    "simulated_status": None,
                    "corroboration": "reported_only",
                }
            )
        for system in mentioned:
            simulated_status = statuses[system]
            if simulated_status == "outage":
                corroboration = "corroborated_outage"
            elif simulated_status == "degraded":
                corroboration = "partially_corroborated"
            else:
                corroboration = "not_corroborated"
            reports.append(
                {
                    "system": system,
                    "user_reported_problem": True,
                    "simulated_status": simulated_status,
                    "corroboration": corroboration,
                }
            )

    return {
        "source": "simulated_demo_state",
        "is_real_monitoring": False,
        "statuses": dict(statuses),
        "user_reported_problem": reports_problem,
        "reports": reports,
        "notice": (
            "Demo evidence only. These values do not come from a monitoring service."
        ),
    }
