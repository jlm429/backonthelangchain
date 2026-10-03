from types import SimpleNamespace

import pytest

from backonthelangchain.agents.graphs import build_jev_support_router_graph
from backonthelangchain.agents.nodes import HUMAN_ESCALATION_ANSWER
from backonthelangchain.agents.schemas import (
    BillingResponse,
    JevSupportDecision,
    RouteDecision,
    SafetyResult,
)
from backonthelangchain.agents.services.jev_router import (
    JEV_MODEL,
    JEV_SUPPORT_QUESTIONS,
    JevSupportRouterError,
    JevSupportRouterService,
)


class FakeTypeSafeClient:
    def __init__(self, response):
        self.response = response
        self.requests = []

    def system_one(self, **request):
        self.requests.append(request)
        return self.response


class FakeSafetyService:
    def __init__(self, *, is_safe: bool = True):
        self.is_safe = is_safe
        self.queries = []

    def check(self, query):
        self.queries.append(query)
        return SafetyResult(
            is_safe=self.is_safe,
            flagged=not self.is_safe,
            model="fake-moderation",
            reason="fake moderation result",
        )


class FakeJevService:
    def __init__(self, result=None, error=None):
        self.result = result
        self.error = error
        self.queries = []

    def evaluate(self, query):
        self.queries.append(query)
        if self.error is not None:
            raise self.error
        return self.result


class FakeRouterService:
    def __init__(self, domain="tech_support"):
        self.domain = domain
        self.queries = []

    def route(self, query):
        self.queries.append(query)
        return RouteDecision(domain=self.domain, reason="fake fallback route")


class FakeTechSupportService:
    def __init__(self):
        self.queries = []

    def answer(self, query):
        self.queries.append(query)
        return "fake technical answer", "fake system status"


class FakeBillingService:
    def __init__(self):
        self.queries = []

    def answer(self, query):
        self.queries.append(query)
        return (
            BillingResponse(
                summary="fake billing summary",
                next_step="fake billing next step",
                urgency="low",
            ),
            "fake subscription status",
        )


def jev_result(
    *,
    route="tech_support",
    confidence=0.90,
    escalation=0.10,
):
    return JevSupportDecision(
        support_route=route,
        support_route_confidence=confidence,
        support_route_probabilities={
            "tech_support": 0.90 if route == "tech_support" else 0.10,
            "billing": 0.90 if route == "billing" else 0.10,
        },
        needs_human_escalation=escalation,
        model=JEV_MODEL,
    )


def build_graph(*, safety, jev, fallback, tech, billing):
    return build_jev_support_router_graph(
        safety_service=safety,
        jev_router_service=jev,
        fallback_router_service=fallback,
        tech_support_service=tech,
        billing_service=billing,
    )


def invoke(graph, query, thread_id):
    return graph.invoke(
        {"user_query": query},
        config={"configurable": {"thread_id": thread_id}},
    )


def test_jev_service_makes_one_request_with_shared_named_state():
    response = SimpleNamespace(
        model=JEV_MODEL,
        answers={
            "support_route": SimpleNamespace(
                choice="billing",
                confidence=0.82,
                probabilities={"tech_support": 0.09, "billing": 0.91},
            ),
            "needs_human_escalation": SimpleNamespace(noul=0.23),
        },
    )
    client = FakeTypeSafeClient(response)

    result = JevSupportRouterService(client=client).evaluate("I was charged twice")

    assert len(client.requests) == 1
    assert client.requests[0] == {
        "model": "jev-1.13.0",
        "state": {"support_query": "I was charged twice"},
        "questions": JEV_SUPPORT_QUESTIONS,
    }
    assert result.support_route == "billing"
    assert result.support_route_confidence == 0.82
    assert result.needs_human_escalation == 0.23


def test_jev_model_cannot_be_overridden():
    with pytest.raises(TypeError):
        JevSupportRouterService(client=FakeTypeSafeClient(None), model="jev-latest")

    with pytest.raises(TypeError):
        build_jev_support_router_graph(jev_model="jev-latest")


def test_jev_service_rejects_an_unbounded_route():
    response = SimpleNamespace(
        model=JEV_MODEL,
        answers={
            "support_route": SimpleNamespace(
                choice="general",
                confidence=0.99,
                probabilities={"general": 0.99},
            ),
            "needs_human_escalation": SimpleNamespace(noul=0.0),
        },
    )

    with pytest.raises(JevSupportRouterError):
        JevSupportRouterService(client=FakeTypeSafeClient(response)).evaluate("hello")


def test_moderation_blocks_before_jev_runs():
    safety = FakeSafetyService(is_safe=False)
    jev = FakeJevService(result=jev_result())
    fallback = FakeRouterService()
    graph = build_graph(
        safety=safety,
        jev=jev,
        fallback=fallback,
        tech=FakeTechSupportService(),
        billing=FakeBillingService(),
    )

    response = invoke(graph, "blocked query", "moderation-block")

    assert response["answer"].startswith("I cannot assist")
    assert jev.queries == []
    assert fallback.queries == []


def test_moderation_block_clears_a_persisted_jev_decision():
    safety = FakeSafetyService()
    jev = FakeJevService(result=jev_result())
    graph = build_graph(
        safety=safety,
        jev=jev,
        fallback=FakeRouterService(),
        tech=FakeTechSupportService(),
        billing=FakeBillingService(),
    )
    thread_id = "moderation-clears-jev"

    first_response = invoke(graph, "login error", thread_id)
    assert first_response["jev_decision_available"] is True

    safety.is_safe = False
    response = invoke(graph, "blocked query", thread_id)

    assert response["answer"].startswith("I cannot assist")
    assert response["jev_decision_available"] is False


def test_high_confidence_jev_choice_routes_without_fallback():
    jev = FakeJevService(result=jev_result(route="tech_support", confidence=0.70))
    fallback = FakeRouterService(domain="billing")
    tech = FakeTechSupportService()
    graph = build_graph(
        safety=FakeSafetyService(),
        jev=jev,
        fallback=fallback,
        tech=tech,
        billing=FakeBillingService(),
    )

    response = invoke(graph, "login error", "direct-jev-route")

    assert response["answer"] == "fake technical answer"
    assert jev.queries == ["login error"]
    assert fallback.queries == []
    assert tech.queries == ["login error"]


def test_low_confidence_jev_choice_uses_fallback_router():
    jev = FakeJevService(result=jev_result(confidence=0.69, escalation=0.40))
    fallback = FakeRouterService(domain="billing")
    billing = FakeBillingService()
    graph = build_graph(
        safety=FakeSafetyService(),
        jev=jev,
        fallback=fallback,
        tech=FakeTechSupportService(),
        billing=billing,
    )

    response = invoke(graph, "ambiguous request", "low-confidence-fallback")

    assert response["answer"]["summary"] == "fake billing summary"
    assert response["needs_human_escalation"] == 0.0
    assert response["jev_human_escalation_probability"] == 0.40
    assert response["jev_decision_available"] is True
    assert fallback.queries == ["ambiguous request"]
    assert billing.queries == ["ambiguous request"]


def test_jev_failure_uses_fallback_router():
    jev = FakeJevService(error=JevSupportRouterError("provider unavailable"))
    fallback = FakeRouterService(domain="tech_support")
    tech = FakeTechSupportService()
    graph = build_graph(
        safety=FakeSafetyService(),
        jev=jev,
        fallback=fallback,
        tech=tech,
        billing=FakeBillingService(),
    )

    response = invoke(graph, "cannot sign in", "jev-failure-fallback")

    assert response["answer"] == "fake technical answer"
    assert response["jev_decision_available"] is False
    assert fallback.queries == ["cannot sign in"]


@pytest.mark.parametrize("fallback_cause", ["failure", "low_confidence"])
def test_fallback_clears_persisted_human_escalation(fallback_cause):
    jev = FakeJevService(result=jev_result(escalation=0.80))
    fallback = FakeRouterService(domain="tech_support")
    tech = FakeTechSupportService()
    graph = build_graph(
        safety=FakeSafetyService(),
        jev=jev,
        fallback=fallback,
        tech=tech,
        billing=FakeBillingService(),
    )
    thread_id = f"persisted-escalation-{fallback_cause}"

    first_response = invoke(graph, "I need a person", thread_id)
    assert first_response["answer"] == HUMAN_ESCALATION_ANSWER

    if fallback_cause == "failure":
        jev.error = JevSupportRouterError("provider unavailable")
    else:
        jev.result = jev_result(confidence=0.69, escalation=0.10)

    response = invoke(graph, "cannot sign in", thread_id)

    assert response["answer"] == "fake technical answer"
    assert fallback.queries == ["cannot sign in"]
    assert tech.queries == ["cannot sign in"]


def test_escalation_threshold_routes_to_deterministic_human_node():
    jev = FakeJevService(result=jev_result(confidence=0.10, escalation=0.80))
    fallback = FakeRouterService()
    tech = FakeTechSupportService()
    billing = FakeBillingService()
    graph = build_graph(
        safety=FakeSafetyService(),
        jev=jev,
        fallback=fallback,
        tech=tech,
        billing=billing,
    )

    response = invoke(graph, "I need a person", "human-escalation")

    assert response["answer"] == HUMAN_ESCALATION_ANSWER
    assert fallback.queries == []
    assert tech.queries == []
    assert billing.queries == []
