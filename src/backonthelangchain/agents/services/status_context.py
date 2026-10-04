"""Build explicit demo evidence from user reports and simulated system state."""

from __future__ import annotations

import re
from typing import Any

from backonthelangchain.agents.schemas import SystemName, SystemStatusLevel

SYSTEM_ALIASES: dict[SystemName, tuple[str, ...]] = {
    "authentication": ("authentication", "auth", "login", "log in", "sign in", "mfa"),
    "billing": ("billing", "invoice", "charge", "subscription", "payment"),
    "checkout": ("checkout", "purchase", "cart", "order"),
    "api": ("api", "endpoint", "webhook"),
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
CLAUSE_BOUNDARY = re.compile(r"[.!?;]+|\b(?:but|while|whereas)\b")
PHRASE_BOUNDARY = re.compile(r",+|\band\b")
HEALTHY_MARKERS = (
    "available",
    "fine",
    "healthy",
    "normal",
    "ok",
    "okay",
    "operational",
    "up",
)


def _contains_phrase(text: str, phrase: str) -> bool:
    return re.search(rf"\b{re.escape(phrase)}\b", text) is not None


def _contains_affirmative_outage_marker(text: str, marker: str) -> bool:
    for match in re.finditer(rf"\b{re.escape(marker)}\b", text):
        prefix = text[: match.start()]
        if not re.search(
            (
                r"(?:\bno|\bnot|\bnever|\bwithout|\bisn't|\bisnt)\s+"
                r"(?:(?:an?|the)\s+)?(?:[\w'-]+\s+){0,2}$"
            ),
            prefix,
        ):
            return True
    return False


def _mentioned_systems(text: str) -> set[SystemName]:
    return {
        system
        for system, aliases in SYSTEM_ALIASES.items()
        if any(_contains_phrase(text, alias) for alias in aliases)
    }


def build_simulated_status_evidence(
    user_query: str,
    statuses: dict[SystemName, SystemStatusLevel],
) -> dict[str, Any]:
    """Compare an outage report with explicitly simulated demo state."""
    normalized_query = user_query.casefold()
    reported_systems: set[SystemName] = set()
    unspecified_report = False
    for clause in CLAUSE_BOUNDARY.split(normalized_query):
        pending_systems: set[SystemName] = set()
        for phrase in PHRASE_BOUNDARY.split(clause):
            phrase_systems = _mentioned_systems(phrase)
            has_affirmative_marker = any(
                _contains_affirmative_outage_marker(phrase, marker)
                for marker in OUTAGE_MARKERS
            )
            if has_affirmative_marker:
                associated_systems = pending_systems | phrase_systems
                if associated_systems:
                    reported_systems.update(associated_systems)
                else:
                    unspecified_report = True
                pending_systems.clear()
                continue

            has_status_language = any(
                _contains_phrase(phrase, marker)
                for marker in (*OUTAGE_MARKERS, *HEALTHY_MARKERS)
            )
            if phrase_systems and not has_status_language:
                pending_systems.update(phrase_systems)
            else:
                pending_systems.clear()

    reports: list[dict[str, Any]] = []
    if unspecified_report and not reported_systems:
        reports.append(
            {
                "system": "unspecified",
                "user_reported_problem": True,
                "simulated_status": None,
                "corroboration": "reported_only",
            }
        )
    for system in SYSTEM_ALIASES:
        if system not in reported_systems:
            continue
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
        "user_reported_problem": bool(reports),
        "reports": reports,
        "notice": (
            "Demo evidence only. These values do not come from a monitoring service."
        ),
    }
