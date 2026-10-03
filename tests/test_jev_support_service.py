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
