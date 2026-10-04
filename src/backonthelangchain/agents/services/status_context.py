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
COMPONENT_CONNECTOR = re.compile(
    r"\s*(?:(?:,|\band\b|\bor\b|\bnor\b|\bas well as\b)\s*)+"
)
PREDICATE_CONJUNCTION = re.compile(r"(?:,\s*)?\b(?:and|or)\b")
COORDINATED_NEGATION = re.compile(r"\bneither\b[^.!?;]*\bnor\b[^.!?;]*$")
NEGATION_PREFIX = re.compile(
    r"(?:\bno|\bnot|\bnever|\bwithout|\bisn't|\bisnt|"
    r"\baren't|\barent|\bwasn't|\bwasnt|\bweren't|\bwerent|"
    r"\bdoesn't|\bdoesnt|\bdon't|\bdont|\bdidn't|\bdidnt)\s+"
    r"(?:(?:an?|the)\s+)?(?:[\w'-]+\s+){0,4}$"
)


def _is_negated(text: str, marker_start: int) -> bool:
    prefix = text[:marker_start]
    negation_scope = prefix
    for conjunction in PREDICATE_CONJUNCTION.finditer(prefix):
        preceding_text = prefix[: conjunction.start()]
        if (
            any(
                re.search(rf"\b{re.escape(marker)}\b", preceding_text)
                for marker in OUTAGE_MARKERS
            )
            or COORDINATED_NEGATION.search(preceding_text)
            or NEGATION_PREFIX.search(preceding_text)
        ):
            negation_scope = prefix[conjunction.end() :]
    return bool(
        COORDINATED_NEGATION.search(negation_scope)
        or NEGATION_PREFIX.search(negation_scope)
    )


def _affirmative_marker_spans(text: str) -> list[tuple[int, int]]:
    spans = []
    for marker in OUTAGE_MARKERS:
        for match in re.finditer(rf"\b{re.escape(marker)}\b", text):
            if not _is_negated(text, match.start()):
                spans.append(match.span())
    return sorted(set(spans))


def _component_groups(text: str) -> list[tuple[int, int, set[SystemName]]]:
    mentions = sorted(
        (
            match.start(),
            match.end(),
            system,
        )
        for system, aliases in SYSTEM_ALIASES.items()
        for alias in aliases
        for match in re.finditer(rf"\b{re.escape(alias)}\b", text)
    )
    groups: list[tuple[int, int, set[SystemName]]] = []
    for start, end, system in mentions:
        if groups and COMPONENT_CONNECTOR.fullmatch(text[groups[-1][1] : start]):
            group_start, _, systems = groups[-1]
            groups[-1] = (group_start, end, systems | {system})
        else:
            groups.append((start, end, {system}))
    return groups


def _distance(
    marker: tuple[int, int],
    group: tuple[int, int, set[SystemName]],
) -> int:
    marker_start, marker_end = marker
    group_start, group_end, _ = group
    if marker_end <= group_start:
        return group_start - marker_end
    if group_end <= marker_start:
        return marker_start - group_end
    return 0


def build_simulated_status_evidence(
    user_query: str,
    statuses: dict[SystemName, SystemStatusLevel],
) -> dict[str, Any]:
    """Compare an outage report with explicitly simulated demo state."""
    normalized_query = user_query.casefold()
    reported_systems: set[SystemName] = set()
    unspecified_report = False
    for clause in CLAUSE_BOUNDARY.split(normalized_query):
        groups = _component_groups(clause)
        for marker in _affirmative_marker_spans(clause):
            if not groups:
                unspecified_report = True
                continue
            nearest_group = min(groups, key=lambda group: _distance(marker, group))
            reported_systems.update(nearest_group[2])

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
