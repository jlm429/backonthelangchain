from backonthelangchain.examples.jev_support import run_jev_support_router


class FakeGraph:
    def __init__(self, state: dict):
        self.state = state
        self.calls = []

    def invoke(self, input: dict, config: dict) -> dict:
        self.calls.append((input, config))
        return self.state


def test_runner_returns_structured_result_from_existing_graph():
    graph = FakeGraph(
        {
            "answer": "Use your recovery code.",
            "is_safe": True,
            "moderation_model": "fake-moderation",
            "safety_reason": "Allowed.",
            "domain": "tech_support",
            "route_reason": "Jev selected the route.",
            "jev_model": "jev-1.13.0",
            "jev_route_confidence": 0.92,
            "jev_route_probabilities": {
                "tech_support": 0.92,
                "billing": 0.08,
            },
            "jev_human_escalation_probability": 0.1,
            "jev_decision_available": True,
            "needs_human_escalation": 0.1,
            "jev_used_fallback": False,
        }
    )

    result = run_jev_support_router("MFA trouble", graph=graph)

    assert result.outcome == "completed"
    assert result.answer == "Use your recovery code."
    assert result.safety.allowed is True
    assert result.routing.destination == "tech_support"
    assert result.jev.route_confidence == 0.92
    assert result.jev.human_escalation_probability == 0.1
    assert graph.calls[0][0] == {"user_query": "MFA trouble"}
    assert graph.calls[0][1]["configurable"]["thread_id"].startswith("jev-support-")


def test_runner_exposes_moderation_block_without_a_jev_result():
    graph = FakeGraph(
        {
            "answer": "Please rephrase your question.",
            "is_safe": False,
            "moderation_model": "fake-moderation",
            "safety_reason": "Blocked.",
        }
    )

    result = run_jev_support_router("blocked", graph=graph)

    assert result.outcome == "blocked"
    assert result.routing.destination == "blocked"
    assert result.jev is None


def test_runner_preserves_jev_observation_during_low_confidence_fallback():
    graph = FakeGraph(
        {
            "answer": "A billing agent can help.",
            "is_safe": True,
            "domain": "billing",
            "route_reason": "Jev confidence was low. Fallback selected billing.",
            "jev_model": "jev-1.13.0",
            "jev_route_confidence": 0.4,
            "jev_route_probabilities": {"tech_support": 0.4, "billing": 0.6},
            "jev_human_escalation_probability": 0.4,
            "jev_decision_available": True,
            "needs_human_escalation": 0.0,
            "jev_used_fallback": True,
        }
    )

    result = run_jev_support_router("ambiguous", graph=graph)

    assert result.outcome == "completed"
    assert result.routing.used_fallback is True
    assert result.jev is not None
    assert result.jev.human_escalation_probability == 0.4


def test_runner_omits_jev_summary_when_provider_failed():
    graph = FakeGraph(
        {
            "answer": "Try your recovery code.",
            "is_safe": True,
            "domain": "tech_support",
            "route_reason": "Jev failed. Fallback selected technical support.",
            "jev_decision_available": False,
            "needs_human_escalation": 0.0,
            "jev_used_fallback": True,
        }
    )

    result = run_jev_support_router("cannot log in", graph=graph)

    assert result.outcome == "completed"
    assert result.routing.used_fallback is True
    assert result.jev is None
