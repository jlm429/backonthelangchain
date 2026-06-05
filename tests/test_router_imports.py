from backonthelangchain.agents.graphs import build_support_router_graph


def test_build_support_router_graph_is_importable():
    assert callable(build_support_router_graph)
