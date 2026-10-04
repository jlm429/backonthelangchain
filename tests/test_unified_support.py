import asyncio
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import re
from threading import Barrier
from time import sleep
from types import SimpleNamespace

import pytest

from backonthelangchain.agents import graphs as graph_module
from backonthelangchain.agents.graphs import (
    UnifiedSupportOptions,
    _LazyDefaultRAGPipeline,
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
from backonthelangchain.rag.chunking import split_markdown_faqs
from backonthelangchain.rag.loaders import load_text_file
from backonthelangchain.rag.pipelines import (
    TechSupportRAGPipeline,
    load_support_knowledge_chunks,
)
from backonthelangchain.rag.rerankers import NoOpReranker
from backonthelangchain.rag.retrieval import RetrievedChunk


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
    def __init__(self, *, route="tech_support", escalation=0.1, confidence=0.9):
        self.route = route
        self.escalation = escalation
        self.confidence = confidence
        self.calls = []

    def evaluate(self, query, *, context=None):
        self.calls.append((query, context))
        return JevSupportDecision(
            support_route=self.route,
            support_route_confidence=self.confidence,
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
            text="Use a recovery code.",
            retrieval_score=0.9,
            rerank_score=None,
        )
        return SimpleNamespace(
            query=query,
            context="Use a recovery code.",
            retrieved_chunks=[source],
            reranked_chunks=[source],
        )


class DeterministicScenarioJevService:
    """Fake Jev outputs used to expose scores without forcing graph outcomes."""

    def __init__(self):
        self.calls = []

    def evaluate(self, query, *, context=None):
        self.calls.append((query, context))
        normalized = query.casefold()
        route = "billing" if "charged twice" in normalized else "tech_support"
        confidence = 0.96 if route == "billing" else 0.93
        escalation = 0.08
        if "reset my password five times" in normalized:
            escalation = 0.46
        elif "actively losing sales" in normalized:
            escalation = 0.68
        elif "connect me to a human" in normalized:
            escalation = 0.91
        elif "completely down" in normalized:
            assessment = context["assessment"] if context else "not_applicable"
            escalation = 0.71 if assessment == "corroborated" else 0.32
        return JevSupportDecision(
            support_route=route,
            support_route_confidence=confidence,
            support_route_probabilities={
                "tech_support": 1.0 - confidence if route == "billing" else confidence,
                "billing": confidence if route == "billing" else 1.0 - confidence,
            },
            needs_human_escalation=escalation,
            model="fake-jev-scenarios",
        )


class AccountingRAGPipeline:
    def __init__(self):
        self.queries = []

    def run(self, query):
        self.queries.append(query)
        text = (
            "## Accounting Workstation Recovery\n\n"
            "Open Service Manager and restart the \"Acme Report Writer\" daemon. "
            "Wait until its status reads READY, reopen Monthly Reporting, and "
            "escalate to Accounting Platform Support if READY fails after two attempts."
        )
        source = SimpleNamespace(
            chunk_id="accounting-demo-1",
            metadata={"title": "Accounting Workstation Recovery"},
            source="/private/repository/accounting_workstation_recovery.md",
            text=text,
            retrieval_score=0.97,
            rerank_score=None,
        )
        context = (
            "[FAQ 1] Accounting Workstation Recovery\n"
            "Source: accounting_workstation_recovery.md\n"
            "Chunk ID: accounting-demo-1\n"
            f"{text}"
        )
        return SimpleNamespace(
            query=query,
            context=context,
            retrieved_chunks=[source],
            reranked_chunks=[source],
        )


class ContextAwareTechSupportService(FakeTechSupportService):
    def answer(self, query, *, system_context=None, rag_context=None):
        self.calls.append((query, system_context, rag_context))
        if rag_context and "Acme Report Writer" in rag_context:
            return (
                "Restart the Acme Report Writer daemon in Service Manager, wait "
                "for READY, and reopen Monthly Reporting. Escalate to Accounting "
                "Platform Support if READY fails after two attempts.",
                system_context,
            )
        return "Try generic workstation troubleshooting and contact support.", system_context


class LexicalDemoRetriever:
    """Dependency-free fake retriever over the real demo document chunks."""

    def __init__(self, chunks):
        self.chunks = chunks

    def retrieve(self, query, *, top_k):
        query_terms = set(re.findall(r"[a-z]+", query.casefold()))
        scored = []
        for chunk in self.chunks:
            chunk_terms = set(re.findall(r"[a-z]+", chunk.text.casefold()))
            overlap = len(query_terms & chunk_terms)
            scored.append(
                RetrievedChunk(
                    chunk=chunk,
                    score=overlap / max(len(query_terms), 1),
                )
            )
        return sorted(scored, key=lambda item: item.score, reverse=True)[:top_k]


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
    assert enabled_result["retrieval"]["documents"][0]["source"] == "fake-faq.md"
    assert "/srv/backonthelangchain" not in str(enabled_events)


def test_runner_builds_request_scoped_graphs_without_checkpoints():
    factory, _, _, _, _ = make_graph_factory()
    calls = []
    checkpointers = []

    def counting_factory(options):
        calls.append(options)
        graph = factory(options)
        checkpointers.append(graph.checkpointer)
        return graph

    runner = UnifiedSupportRunner(counting_factory)
    collect_events(runner, request(retrieval=True))
    collect_events(runner, request(retrieval=True, query="Checkout is down"))

    assert calls == [
        UnifiedSupportOptions(faq_retrieval=True),
        UnifiedSupportOptions(faq_retrieval=True),
    ]
    assert checkpointers == [None, None]


def test_default_rag_pipeline_initializes_once_under_concurrency(monkeypatch):
    builds = []

    class FakePipeline:
        def run(self, query):
            return query

    def build_pipeline():
        builds.append(True)
        sleep(0.05)
        return FakePipeline()

    monkeypatch.setattr(graph_module, "_build_default_rag_pipeline", build_pipeline)
    pipeline = _LazyDefaultRAGPipeline()
    start = Barrier(2)

    def run(query):
        start.wait()
        return pipeline.run(query)

    with ThreadPoolExecutor(max_workers=2) as executor:
        results = list(executor.map(run, ["login", "checkout"]))

    assert sorted(results) == ["checkout", "login"]
    assert builds == [True]


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
        "Checkout is no longer down",
        "Checkout did not appear to be down",
        "There is no checkout outage",
        "Checkout isn't unavailable",
        "Authentication and checkout aren't down",
        "Neither billing nor checkout is down",
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


@pytest.mark.parametrize(
    "query",
    [
        "Checkout is not operational and authentication is down",
        "Billing doesn't work and authentication is down",
        "Checkout is not down and authentication is down",
        "Neither billing nor checkout is down and authentication is down",
        "Neither billing nor checkout is down, and authentication is down",
    ],
)
def test_coordinated_negation_does_not_hide_a_later_outage_report(query):
    factory, _, jev, _, _ = make_graph_factory()

    events = collect_events(
        UnifiedSupportRunner(factory),
        request(
            status="outage",
            query=query,
        ),
    )

    evidence = completed_result(events)["system_status"]["evidence"]
    assert [report["system"] for report in evidence["reports"]] == [
        "authentication"
    ]
    assert evidence["reports"][0]["corroboration"] == "corroborated_outage"
    assert jev.calls[0][1]["reports"] == evidence["reports"]


def test_outage_evidence_is_associated_with_its_component_clause():
    factory, _, jev, _, _ = make_graph_factory()

    events = collect_events(
        UnifiedSupportRunner(factory),
        request(
            status="outage",
            status_system="checkout",
            query="Billing looks normal, but checkout is down",
        ),
    )

    evidence = completed_result(events)["system_status"]["evidence"]
    assert [report["system"] for report in evidence["reports"]] == ["checkout"]
    assert evidence["reports"][0]["corroboration"] == "corroborated_outage"
    assert jev.calls[0][1]["reports"] == evidence["reports"]


@pytest.mark.parametrize(
    ("query", "expected_systems"),
    [
        ("Billing is operational and checkout is down", ["checkout"]),
        ("Authentication and checkout are down", ["authentication", "checkout"]),
        (
            "An outage affects authentication as well as checkout",
            ["authentication", "checkout"],
        ),
        ("An outage affects checkout and billing", ["billing", "checkout"]),
        ("I can't log in to view my invoice", ["authentication"]),
        ("I can't request a payment refund", ["billing"]),
    ],
)
def test_outage_evidence_is_associated_with_component_phrases(
    query,
    expected_systems,
):
    statuses = SimulatedSystemState(
        authentication="outage",
        billing="outage",
        checkout="outage",
        api="outage",
    ).model_dump()

    evidence = build_simulated_status_evidence(query, statuses)

    assert [report["system"] for report in evidence["reports"]] == expected_systems


def test_mixed_service_evidence_is_preserved_in_stage_and_execution_summaries():
    statuses = SimulatedSystemState(
        authentication="outage",
        billing="operational",
        checkout="operational",
        api="operational",
    )
    payload = SupportRunRequest(
        query="Authentication and checkout are down",
        options=SupportGraphOptions(faq_retrieval=False),
        simulated_status=statuses,
    )
    factory, _, _, _, _ = make_graph_factory()

    result = completed_result(
        collect_events(UnifiedSupportRunner(factory), payload)
    )

    evidence = result["system_status"]["evidence"]
    assert evidence["assessment"] == "mixed"
    assert {
        report["system"]: report["relation"] for report in evidence["reports"]
    } == {
        "authentication": "corroborated",
        "checkout": "contradicted",
    }
    status_stage = next(
        item
        for item in result["stage_evidence"]
        if item["stage_id"] == "simulated_status_context"
    )
    assert status_stage["summary"] == (
        "System evidence was mixed. Service relations: "
        "authentication=corroborated, checkout=contradicted."
    )
    system_fact = next(
        fact["text"]
        for fact in result["execution_summary"]["facts"]
        if fact["category"] == "system_evidence"
    )
    assert "assessment was mixed" in system_fact
    assert "authentication=corroborated, checkout=contradicted" in system_fact
    assert result["provenance"]["system_evidence_assessment"] == "mixed"


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
    assert status_event["evidence"]["outputs"]["assessment"] == "contradicted"
    assert status_event["evidence"]["outputs"]["reports"][0][
        "user_reported_problem"
    ] is True
    assert set(status_event["evidence"]) == {
        "stage_id",
        "label",
        "summary",
        "inputs",
        "outputs",
    }


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


@pytest.mark.parametrize(
    ("query", "expected_route", "expected_outcome", "expected_score"),
    [
        (
            "I cannot log in after enabling MFA.",
            "tech_support",
            "completed",
            0.08,
        ),
        (
            "I have reset my password five times, re-enrolled MFA twice, and I "
            "still cannot access my account. I have been locked out since yesterday.",
            "tech_support",
            "completed",
            0.46,
        ),
        (
            "Our checkout system has been unavailable for 45 minutes and customers "
            "cannot place orders. We are actively losing sales.",
            "tech_support",
            "completed",
            0.68,
        ),
        (
            "I have tried the troubleshooting steps several times and this still "
            "isn't working. Please connect me to a human.",
            "human_escalation",
            "escalated",
            0.91,
        ),
        (
            "I was charged twice for my subscription this month.",
            "billing",
            "completed",
            0.08,
        ),
    ],
)
def test_deterministic_jev_scenarios_expose_actual_scores_without_overrides(
    query,
    expected_route,
    expected_outcome,
    expected_score,
):
    factory, _, _, _, _ = make_graph_factory(jev=DeterministicScenarioJevService())

    result = completed_result(
        collect_events(UnifiedSupportRunner(factory), request(query=query))
    )

    assert result["routing"]["destination"] == expected_route
    assert result["outcome"] == expected_outcome
    assert result["jev"]["human_escalation_probability"] == expected_score
    assert result["jev"]["human_escalation_threshold"] == 0.8
    assert result["jev"]["human_escalation_threshold_met"] is (
        expected_score >= 0.8
    )


def test_provenance_keeps_jev_classification_distinct_from_fallback_route():
    factory, _, _, _, _ = make_graph_factory(
        jev=FakeJevService(route="billing", confidence=0.6)
    )

    result = completed_result(
        collect_events(
            UnifiedSupportRunner(factory),
            request(query="I was charged twice for my subscription."),
        )
    )

    assert result["jev"]["classified_route"] == "billing"
    assert result["routing"]["destination"] == "tech_support"
    assert result["provenance"]["jev_route"] == "billing"


def test_reported_checkout_outage_exposes_contradictory_and_corroborating_evidence():
    query = "Our checkout system is completely down and nobody can place an order."
    jev = DeterministicScenarioJevService()
    factory, _, _, _, _ = make_graph_factory(jev=jev)
    runner = UnifiedSupportRunner(factory)

    operational = completed_result(
        collect_events(
            runner,
            request(query=query, status_system="checkout", status="operational"),
        )
    )
    outage = completed_result(
        collect_events(
            runner,
            request(query=query, status_system="checkout", status="outage"),
        )
    )

    assert operational["system_status"]["evidence"]["assessment"] == "contradicted"
    assert outage["system_status"]["evidence"]["assessment"] == "corroborated"
    assert operational["jev"]["human_escalation_probability"] == 0.32
    assert outage["jev"]["human_escalation_probability"] == 0.71
    assert operational["outcome"] == outage["outcome"] == "completed"
    assert operational["jev"]["human_escalation_threshold_met"] is False
    assert outage["jev"]["human_escalation_threshold_met"] is False


def test_accounting_rag_ab_changes_supplied_context_and_generated_response():
    query = (
        "I restarted my accounting workstation and now the monthly report writer "
        "won't generate reports. What should I do?"
    )
    tech = ContextAwareTechSupportService()
    rag = AccountingRAGPipeline()
    factory, _, _, _, _ = make_graph_factory(tech=tech, rag=rag)
    runner = UnifiedSupportRunner(factory)

    without_rag = completed_result(
        collect_events(runner, request(query=query, retrieval=False))
    )
    with_rag = completed_result(
        collect_events(runner, request(query=query, retrieval=True))
    )

    assert "Acme Report Writer" not in without_rag["answer"]
    assert without_rag["retrieval"]["context_supplied_to_response"] is None
    assert with_rag["retrieval"]["documents"][0]["title"] == (
        "Accounting Workstation Recovery"
    )
    assert with_rag["retrieval"]["documents"][0]["retrieval_score"] == 0.97
    assert "Acme Report Writer" in with_rag["retrieval"][
        "context_supplied_to_response"
    ]
    assert with_rag["response_generation"]["inputs"][
        "retrieved_knowledge_supplied"
    ] == with_rag["retrieval"]["context_supplied_to_response"]
    assert "Acme Report Writer" in with_rag["answer"]
    assert without_rag["answer"] != with_rag["answer"]


def test_demo_knowledge_base_contains_distinct_fictional_procedures():
    knowledge_directory = (
        Path(__file__).parents[1]
        / "src/backonthelangchain/rag/data/demo_support_knowledge"
    )
    chunks = [
        chunk
        for document_path in sorted(knowledge_directory.glob("*.md"))
        for chunk in split_markdown_faqs(load_text_file(document_path))
    ]

    assert {chunk.metadata["title"] for chunk in chunks} == {
        "Accounting Workstation Recovery",
        "Field VPN Certificate Recovery",
        "Warehouse Scanner Synchronization",
        "Meeting Room Display Recovery",
    }
    accounting = next(
        chunk for chunk in chunks if chunk.metadata["title"] == "Accounting Workstation Recovery"
    )
    assert "Acme Report Writer" in accounting.text
    assert "READY" in accounting.text
    assert "Accounting Platform Support" in accounting.text


def test_deterministic_retrieval_selects_accounting_recovery_document():
    knowledge_directory = (
        Path(__file__).parents[1]
        / "src/backonthelangchain/rag/data/demo_support_knowledge"
    )
    chunks = load_support_knowledge_chunks(knowledge_directory)
    pipeline = TechSupportRAGPipeline(
        faq_path=knowledge_directory,
        embedding_model=object(),
        retriever=LexicalDemoRetriever(chunks),
        reranker=NoOpReranker(),
        retrieve_top_k=4,
        rerank_top_k=1,
    )

    result = pipeline.run(
        "I restarted my accounting workstation and the monthly report writer "
        "will not generate reports."
    )

    assert result.reranked_chunks[0].metadata["title"] == (
        "Accounting Workstation Recovery"
    )
    assert "Acme Report Writer" in result.context
    assert "accounting_workstation_recovery.md" in result.context
    assert str(knowledge_directory) not in result.context


def test_stage_evidence_and_summary_match_the_executed_graph_state():
    factory, _, _, _, _ = make_graph_factory(jev=DeterministicScenarioJevService())
    result = completed_result(
        collect_events(
            UnifiedSupportRunner(factory),
            request(
                query="Our checkout system is completely down and nobody can place an order.",
                status_system="checkout",
                status="outage",
            ),
        )
    )

    assert [item["stage_id"] for item in result["stage_evidence"]] == [
        "__start__",
        *result["execution_path"],
        "__end__",
    ]
    summary_text = " ".join(
        fact["text"] for fact in result["execution_summary"]["facts"]
    )
    assert "OpenAI Moderation allowed" in summary_text
    assert "assessment was corroborated" in summary_text
    assert "Escalation probability was 0.71" in summary_text
    assert "0.80 escalation threshold was not met" in summary_text
    assert "workflow outcome was completed at tech_support" in summary_text
    assert result["provenance"]["human_escalation_triggered"] is False


def test_public_evidence_omits_private_provider_fields_and_sensitive_errors():
    sentinel = "private-provider-reasoning-and-secret-token"
    safety = FakeSafetyService()
    safety.private_reasoning = sentinel
    jev = FakeJevService()
    jev.raw_provider_response = {"authorization": sentinel}
    tech = FakeTechSupportService()
    tech.hidden_chain_of_thought = sentinel
    factory, _, _, _, _ = make_graph_factory(safety=safety, jev=jev, tech=tech)

    events = collect_events(UnifiedSupportRunner(factory), request())
    serialized = json.dumps(events).casefold()

    assert sentinel not in serialized
    for forbidden_key in (
        '"chain_of_thought"',
        '"hidden_reasoning"',
        '"raw_provider_response"',
        '"authorization"',
        '"api_key"',
        '"request_headers"',
        '"reasoning_tokens"',
    ):
        assert forbidden_key not in serialized
