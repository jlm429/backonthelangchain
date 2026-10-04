import asyncio
from types import SimpleNamespace

import pytest

from backonthelangchain.agents.graphs import (
    build_unified_support_graph,
)
from backonthelangchain.agents.schemas import (
    BillingResponse,
    JevSupportDecision,
    RouteDecision,
    SafetyResult,
)
from backonthelangchain.agents.services.status_context import (
    build_simulated_status_evidence,
)
from backonthelangchain.api.schemas import (
    SimulatedSystemState,
    SupportGraphOptions,
    SupportRunRequest,
)
from backonthelangchain.examples.unified_support import (
    UnifiedSupportRunner,
    describe_unified_support_graph,
)


class FakeSafetyService:
    def __init__(self, *, allowed=True):
        self.allowed = allowed
        self.queries = []

    def check(self, query):
        self.queries.append(query)
        return SafetyResult(
            is_safe=self.allowed,
            flagged=not self.allowed,
            model="fake-moderation",
            reason="Allowed." if self.allowed else "Flagged.",
        )


class FakeJevService:
    def __init__(self, *, route="tech_support", escalation=0.1):
        self.route = route
        self.escalation = escalation
        self.calls = []

    def evaluate(self, query, *, context=None):
        self.calls.append((query, context))
        return JevSupportDecision(
            support_route=self.route,
            support_route_confidence=0.9,
            support_route_probabilities={
                "tech_support": 0.9 if self.route == "tech_support" else 0.1,
                "billing": 0.9 if self.route == "billing" else 0.1,
            },
            needs_human_escalation=self.escalation,
            model="fake-jev",
        )


class FakeRouterService:
    def route(self, _query, *, context=None):
        return RouteDecision(domain="tech_support", reason="Fake fallback.")


class FakeTechSupportService:
    def __init__(self):
        self.calls = []

    def answer(self, query, *, system_context=None, rag_context=None):
        self.calls.append((query, system_context, rag_context))
        return "Technical answer.", system_context or "No status context."


class FakeBillingService:
    def answer(self, _query):
        return (
            BillingResponse(
                summary="Billing summary.",
                next_step="Review invoice.",
                urgency="low",
            ),
            "Fake subscription.",
        )


class FakeRAGPipeline:
    def __init__(self):
        self.queries = []

    def run(self, query):
        self.queries.append(query)
        source = SimpleNamespace(
            chunk_id="faq-1",
            metadata={"title": "Login help"},
            source="/srv/backonthelangchain/data/fake-faq.md",
            retrieval_score=0.9,
            rerank_score=None,
        )
        return SimpleNamespace(
            context="Use a recovery code.",
            reranked_chunks=[source],
        )


def make_graph_factory(*, safety=None, jev=None, tech=None, rag=None):
    safety = safety or FakeSafetyService()
    jev = jev or FakeJevService()
    tech = tech or FakeTechSupportService()
    rag = rag or FakeRAGPipeline()

    def factory(options):
        return build_unified_support_graph(
            options=options,
            safety_service=safety,
            jev_router_service=jev,
            fallback_router_service=FakeRouterService(),
            tech_support_service=tech,
            billing_service=FakeBillingService(),
            rag_pipeline=rag,
        )

    return factory, safety, jev, tech, rag


def request(
    *,
    retrieval=False,
    status="operational",
    status_system="authentication",
    query="Login is down",
):
    statuses = {
        "authentication": "operational",
        "billing": "operational",
        "checkout": "operational",
        "api": "operational",
    }
    statuses[status_system] = status
    return SupportRunRequest(
        query=query,
        options=SupportGraphOptions(faq_retrieval=retrieval),
        simulated_status=SimulatedSystemState(**statuses),
    )


def collect_events(runner, payload):
    async def collect():
        return [event async for event in runner.events(payload)]

    return asyncio.run(collect())


def completed_result(events):
    return next(event["result"] for event in events if event["type"] == "run_completed")


def test_graph_description_comes_from_enabled_executable_shape():
    disabled = describe_unified_support_graph(SupportGraphOptions())
    enabled = describe_unified_support_graph(
        SupportGraphOptions(faq_retrieval=True)
    )

    disabled_nodes = {node["id"]: node for node in disabled["nodes"]}
    enabled_edges = {(edge["source"], edge["target"]) for edge in enabled["edges"]}
    disabled_edges = {
        (edge["source"], edge["target"]) for edge in disabled["edges"]
    }

    assert disabled_nodes["safety_check"]["required"] is True
    assert disabled_nodes["faq_retrieval"]["enabled"] is False
    assert ("jev_router", "tech_support_answer") in disabled_edges
    assert ("jev_router", "faq_retrieval") in enabled_edges
    assert ("faq_retrieval", "tech_support_answer") in enabled_edges
    assert next(
        edge
        for edge in disabled["edges"]
        if edge["source"] == "jev_router"
        and edge["target"] == "tech_support_answer"
    )["branch"] == "technical"
    assert {
        (edge["source"], edge["target"], edge["branch"])
        for edge in enabled["edges"]
        if edge["conditional"]
    } == {
        ("safety_check", "blocked_response", "flagged"),
        ("safety_check", "simulated_status_context", "allowed"),
        ("jev_router", "human_escalation", "escalate"),
        ("jev_router", "billing_answer", "billing"),
        ("jev_router", "faq_retrieval", "technical"),
    }


def test_optional_retrieval_is_executed_only_when_enabled():
    factory, _, _, tech, rag = make_graph_factory()
    runner = UnifiedSupportRunner(factory)

    disabled_events = collect_events(runner, request(retrieval=False))
    enabled_events = collect_events(runner, request(retrieval=True))

    assert "faq_retrieval" not in completed_result(disabled_events)["execution_path"]
    assert "faq_retrieval" in completed_result(enabled_events)["execution_path"]
    assert rag.queries == ["Login is down"]
    assert tech.calls[0][2] is None
    assert tech.calls[1][2] == "Use a recovery code."
    enabled_result = completed_result(enabled_events)
    assert enabled_result["retrieval"]["sources"][0]["source"] == "fake-faq.md"
    assert "/srv/backonthelangchain" not in str(enabled_events)


def test_mandatory_moderation_blocks_all_later_stages():
    factory, _, jev, _, rag = make_graph_factory(
        safety=FakeSafetyService(allowed=False)
    )

    events = collect_events(UnifiedSupportRunner(factory), request(retrieval=True))
    result = completed_result(events)

    assert result["outcome"] == "blocked"
    assert result["execution_path"] == ["safety_check", "blocked_response"]
    assert jev.calls == []
    assert rag.queries == []


@pytest.mark.parametrize(
    ("status", "expected"),
    [
        ("operational", "not_corroborated"),
        ("degraded", "partially_corroborated"),
        ("outage", "corroborated_outage"),
    ],
)
def test_each_simulated_status_level_reaches_jev_context(status, expected):
    factory, _, jev, _, _ = make_graph_factory()

    events = collect_events(
        UnifiedSupportRunner(factory),
        request(status=status),
    )

    result = completed_result(events)
    report = result["system_status"]["evidence"]["reports"][0]
    assert report["corroboration"] == expected
    assert jev.calls[0][1]["statuses"]["authentication"] == status
    assert jev.calls[0][1]["is_real_monitoring"] is False


def test_reported_only_outage_is_distinct_from_corroborating_state():
    statuses = SimulatedSystemState(
        authentication="operational",
        billing="operational",
        checkout="operational",
        api="operational",
    ).model_dump()

    reported_only = build_simulated_status_evidence(
        "Production is down",
        statuses,
    )
    corroborated = build_simulated_status_evidence(
        "Checkout is down",
        {**statuses, "checkout": "outage"},
    )

    assert reported_only["reports"][0]["corroboration"] == "reported_only"
    assert corroborated["reports"][0]["corroboration"] == "corroborated_outage"


@pytest.mark.parametrize(
    "query",
    [
        "Checkout is not down",
        "There is no checkout outage",
        "Checkout isn't unavailable",
    ],
)
def test_negated_outage_language_does_not_create_evidence(query):
    factory, _, jev, _, _ = make_graph_factory()

    events = collect_events(
        UnifiedSupportRunner(factory),
        request(status="outage", status_system="checkout", query=query),
    )

    evidence = completed_result(events)["system_status"]["evidence"]
    assert evidence["user_reported_problem"] is False
    assert evidence["reports"] == []
    assert jev.calls[0][1]["user_reported_problem"] is False


def test_live_events_use_graph_node_ids_and_preserve_actual_path():
    factory, _, _, _, _ = make_graph_factory()

    events = collect_events(UnifiedSupportRunner(factory), request())
    starts = [event["node_id"] for event in events if event["type"] == "node_started"]
    result = completed_result(events)

    assert starts == result["execution_path"]
    assert starts == [
        "safety_check",
        "simulated_status_context",
        "jev_router",
        "tech_support_answer",
    ]
    assert all("reasoning" not in event for event in events)
    status_event = next(
        event
        for event in events
        if event.get("type") == "node_completed"
        and event.get("node_id") == "simulated_status_context"
    )
    assert status_event["output"]["status_evidence"]["user_reported_problem"] is True
    assert set(status_event["output"]) == {"status_evidence"}


@pytest.mark.parametrize(
    ("jev", "expected_path", "outcome"),
    [
        (
            FakeJevService(route="billing"),
            [
                "safety_check",
                "simulated_status_context",
                "jev_router",
                "billing_answer",
            ],
            "completed",
        ),
        (
            FakeJevService(escalation=0.8),
            [
                "safety_check",
                "simulated_status_context",
                "jev_router",
                "human_escalation",
            ],
            "escalated",
        ),
    ],
)
def test_unified_graph_preserves_billing_and_escalation_paths(
    jev,
    expected_path,
    outcome,
):
    factory, _, _, _, rag = make_graph_factory(jev=jev)

    events = collect_events(
        UnifiedSupportRunner(factory),
        request(retrieval=True, query="Please help with this request"),
    )
    result = completed_result(events)

    assert result["execution_path"] == expected_path
    assert result["outcome"] == outcome
    assert rag.queries == []


def test_node_failure_is_streamed_without_provider_details():
    class FailingSafetyService:
        def check(self, _query):
            raise RuntimeError("provider token must stay private")

    factory, _, _, _, _ = make_graph_factory(safety=FailingSafetyService())

    events = collect_events(UnifiedSupportRunner(factory), request())

    assert events[-2] == {
        "type": "node_failed",
        "node_id": "safety_check",
        "message": "This stage could not be completed.",
    }
    assert events[-1]["type"] == "run_failed"
    assert "provider token" not in str(events)
