import json
import pytest


class TestGraphConstruction:
    """
    Validates nodes addition, dependency weights, and tracking properties.
    """

    def test_add_service(self, config):
        """
        Verify that adding a service correctly registers it as a node in the network.

        Args:
            config: Test configuration fixture
        """
        from tracing.causal_graph import CausalGraph
        cg = CausalGraph(config)
        cg.add_service("svc-a")
        assert "svc-a" in cg.graph.nodes

    def test_add_service_with_metadata(self, config):
        """
        Verify that metadata values are correctly attached during service node insertion.

        Args:
            config: Test configuration fixture
        """
        from tracing.causal_graph import CausalGraph
        cg = CausalGraph(config)
        cg.add_service("svc-a", metadata={"team": "platform"})
        assert cg.graph.nodes["svc-a"]["metadata"]["team"] == "platform"

    def test_add_service_idempotent(self, config):
        """
        Verify that repeatedly adding the same service name acts idempotently.

        Args:
            config: Test configuration fixture
        """
        from tracing.causal_graph import CausalGraph
        cg = CausalGraph(config)
        cg.add_service("svc-a")
        cg.add_service("svc-a")
        assert cg.graph.number_of_nodes() == 1

    def test_add_dependency(self, config):
        """
        Verify that adding a dependency inserts the directed edge and initializes weight/latency.

        Args:
            config: Test configuration fixture
        """
        from tracing.causal_graph import CausalGraph
        cg = CausalGraph(config)
        cg.add_dependency("svc-a", "svc-b", call_count=10, latency=5.0)
        assert cg.graph.has_edge("svc-a", "svc-b")
        assert "svc-a" in cg.graph.nodes
        assert "svc-b" in cg.graph.nodes

    def test_dependency_weight(self, config):
        """
        Verify that directed call count mapping maps exactly to edge weights.

        Args:
            config: Test configuration fixture
        """
        from tracing.causal_graph import CausalGraph
        cg = CausalGraph(config)
        cg.add_dependency("a", "b", call_count=10)
        assert cg.graph["a"]["b"]["weight"] == 10

    def test_dependency_weight_accumulation(self, config):
        """
        Verify that adding dependency increments and decays weights correctly.

        Args:
            config: Test configuration fixture
        """
        from tracing.causal_graph import CausalGraph
        cg = CausalGraph(config)
        cg.add_dependency("a", "b", call_count=10)
        cg.add_dependency("a", "b", call_count=5)
        expected = 10 * config["tracing"]["graph"]["edge_weight_decay"] + 5
        assert abs(cg.graph["a"]["b"]["weight"] - expected) < 1e-6

    def test_latency_tracking(self, config):
        """
        Verify that average edge latency computation behaves correctly across calls.

        Args:
            config: Test configuration fixture
        """
        from tracing.causal_graph import CausalGraph
        cg = CausalGraph(config)
        cg.add_dependency("a", "b", latency=5.0)
        cg.add_dependency("a", "b", latency=10.0)
        assert cg.graph["a"]["b"]["avg_latency"] == 7.5

    def test_call_count_tracking(self, config):
        """
        Verify that total call counters are tracked properly in the lookup map.

        Args:
            config: Test configuration fixture
        """
        from tracing.causal_graph import CausalGraph
        cg = CausalGraph(config)
        cg.add_dependency("a", "b", call_count=3)
        cg.add_dependency("a", "b", call_count=7)
        assert cg.edge_call_counts[("a", "b")] == 10


class TestGraphTraversal:
    """
    Validates BFS-based downstream and upstream neighborhood exploration.
    """

    def test_downstream_services(self, causal_graph):
        """
        Verify that all transitive downstream dependents are identified up to a hop limit.

        Args:
            causal_graph: CausalGraph test fixture
        """
        downstream = causal_graph.get_downstream_services("gateway", max_hops=3)
        assert "svc-a" in downstream
        assert "svc-c" in downstream
        assert "svc-d" in downstream

    def test_downstream_max_hops_1(self, causal_graph):
        """
        Verify downstream exploration with a single-hop restriction.

        Args:
            causal_graph: CausalGraph test fixture
        """
        downstream = causal_graph.get_downstream_services("gateway", max_hops=1)
        assert downstream == {"svc-a"}

    def test_downstream_leaf_node(self, causal_graph):
        """
        Verify that downstream dependents of a terminal node is an empty set.

        Args:
            causal_graph: CausalGraph test fixture
        """
        assert causal_graph.get_downstream_services("svc-d", max_hops=3) == set()

    def test_downstream_missing_node(self, config):
        """
        Verify downstream dependents of an unregistered node name is empty.

        Args:
            config: Test configuration fixture
        """
        from tracing.causal_graph import CausalGraph
        cg = CausalGraph(config)
        assert cg.get_downstream_services("nonexistent") == set()

    def test_upstream_services(self, causal_graph):
        """
        Verify upstream caller propagation paths are resolved accurately.

        Args:
            causal_graph: CausalGraph test fixture
        """
        upstream = causal_graph.get_upstream_services("svc-d", max_hops=3)
        assert "svc-b" in upstream
        assert "svc-a" in upstream

    def test_upstream_max_hops_1(self, causal_graph):
        """
        Verify upstream callers with a single-hop limit.

        Args:
            causal_graph: CausalGraph test fixture
        """
        upstream = causal_graph.get_upstream_services("svc-d", max_hops=1)
        assert upstream == {"svc-b"}

    def test_upstream_root_node(self, causal_graph):
        """
        Verify upstream callers of a root node returns an empty set.

        Args:
            causal_graph: CausalGraph test fixture
        """
        assert causal_graph.get_upstream_services("gateway", max_hops=3) == set()

    def test_upstream_missing_node(self, config):
        """
        Verify upstream callers of a nonexistent service returns empty.

        Args:
            config: Test configuration fixture
        """
        from tracing.causal_graph import CausalGraph
        cg = CausalGraph(config)
        assert cg.get_upstream_services("nonexistent") == set()


class TestPathFinding:
    """
    Validates propagation path resolution between source and target nodes.
    """

    def test_shortest_path(self, causal_graph):
        """
        Verify shortest propagation path coordinates correctly across multiple nodes.

        Args:
            causal_graph: CausalGraph test fixture
        """
        path = causal_graph.get_propagation_path("gateway", "svc-d")
        assert path is not None
        assert path[0] == "gateway"
        assert path[-1] == "svc-d"
        assert len(path) == 4

    def test_no_path_reverse(self, causal_graph):
        """
        Verify that backward path exploration along directed links returns None.

        Args:
            causal_graph: CausalGraph test fixture
        """
        assert causal_graph.get_propagation_path("svc-d", "gateway") is None

    def test_no_path_missing_source(self, causal_graph):
        """
        Verify path finding returns None if source is missing.

        Args:
            causal_graph: CausalGraph test fixture
        """
        assert causal_graph.get_propagation_path("nonexistent", "svc-a") is None

    def test_no_path_missing_target(self, causal_graph):
        """
        Verify path finding returns None if target is missing.

        Args:
            causal_graph: CausalGraph test fixture
        """
        assert causal_graph.get_propagation_path("gateway", "nonexistent") is None

    def test_path_to_self(self, causal_graph):
        """
        Verify that path finding from a node to itself is returned immediately.

        Args:
            causal_graph: CausalGraph test fixture
        """
        path = causal_graph.get_propagation_path("svc-a", "svc-a")
        assert path == ["svc-a"]

    def test_direct_path(self, causal_graph):
        """
        Verify that adjacent nodes yield a direct path of length 2.

        Args:
            causal_graph: CausalGraph test fixture
        """
        path = causal_graph.get_propagation_path("gateway", "svc-a")
        assert path == ["gateway", "svc-a"]


class TestImpactScore:
    """
    Validates computation of service impact weights.
    """

    def test_gateway_highest_impact(self, causal_graph):
        """
        Verify that root gateway services yield higher impact weights than leaves.

        Args:
            causal_graph: CausalGraph test fixture
        """
        gw = causal_graph.get_impact_score("gateway")
        leaf = causal_graph.get_impact_score("svc-d")
        assert gw > leaf

    def test_leaf_has_zero_downstream(self, causal_graph):
        """
        Verify that leaf nodes carry non-negative impact values.

        Args:
            causal_graph: CausalGraph test fixture
        """
        score = causal_graph.get_impact_score("svc-d")
        assert score >= 0

    def test_missing_node_zero(self, config):
        """
        Verify that impact calculation for an unregistered service yields 0.

        Args:
            config: Test configuration fixture
        """
        from tracing.causal_graph import CausalGraph
        cg = CausalGraph(config)
        assert cg.get_impact_score("nonexistent") == 0.0


class TestAnomalyMarking:
    """
    Validates dynamic node state changes on anomaly flagging.
    """

    def test_mark_anomaly(self, causal_graph):
        """
        Verify that flagging a service correctly registers it as anomalous in the graph metadata.

        Args:
            causal_graph: CausalGraph test fixture
        """
        causal_graph.mark_anomaly("svc-a")
        assert "svc-a" in causal_graph.anomalous_services
        assert causal_graph.graph.nodes["svc-a"]["is_anomalous"] is True

    def test_clear_anomaly(self, causal_graph):
        """
        Verify that clearing flags updates graph attributes and anomaly tracking list.

        Args:
            causal_graph: CausalGraph test fixture
        """
        causal_graph.mark_anomaly("svc-a")
        causal_graph.clear_anomaly("svc-a")
        assert "svc-a" not in causal_graph.anomalous_services
        assert causal_graph.graph.nodes["svc-a"]["is_anomalous"] is False

    def test_mark_unknown_service(self, config):
        """
        Verify that marking an unknown service inserts the node dynamically.

        Args:
            config: Test configuration fixture
        """
        from tracing.causal_graph import CausalGraph
        cg = CausalGraph(config)
        cg.mark_anomaly("new-svc")
        assert "new-svc" in cg.graph.nodes
        assert "new-svc" in cg.anomalous_services


class TestEdgeDecayAndPruning:
    """
    Validates sliding window decay and weak edge pruning strategies.
    """

    def test_edge_weight_decay(self, config):
        """
        Verify that decay runs reduce edge weights across evaluation rounds.

        Args:
            config: Test configuration fixture
        """
        from tracing.causal_graph import CausalGraph
        cg = CausalGraph(config)
        cg.add_dependency("a", "b", call_count=100)
        initial = cg.graph["a"]["b"]["weight"]
        cg._apply_edge_decay()
        assert cg.graph["a"]["b"]["weight"] < initial

    def test_weak_edge_pruned(self, config):
        """
        Verify that edge weights falling below threshold limits are pruned from graph.

        Args:
            config: Test configuration fixture
        """
        from tracing.causal_graph import CausalGraph
        cg = CausalGraph(config)
        cg.add_dependency("a", "b", call_count=1)
        cg.graph["a"]["b"]["weight"] = 0.001
        cg._prune_weak_edges()
        assert not cg.graph.has_edge("a", "b")

    def test_strong_edge_not_pruned(self, config):
        """
        Confirm that edge weights above minimum limits are preserved.

        Args:
            config: Test configuration fixture
        """
        from tracing.causal_graph import CausalGraph
        cg = CausalGraph(config)
        cg.add_dependency("a", "b", call_count=100)
        cg._prune_weak_edges()
        assert cg.graph.has_edge("a", "b")

    def test_update_from_traces(self, config):
        """
        Verify graph building by parsing trace logs recursively.

        Args:
            config: Test configuration fixture
        """
        from tracing.causal_graph import CausalGraph
        cg = CausalGraph(config)
        traces = [
            {"dependencies": [
                {"from": "svc-a", "to": "svc-b", "duration": 5.0},
                {"from": "svc-b", "to": "svc-c", "duration": 3.0},
            ]}
        ]
        cg.update_from_traces(traces)
        assert cg.graph.has_edge("svc-a", "svc-b")
        assert cg.graph.has_edge("svc-b", "svc-c")


class TestSnapshotAndExport:
    """
    Validates serialization and snapshots creation limits.
    """

    def test_create_snapshot(self, causal_graph):
        """
        Verify snapshot records correct number of nodes and edges at capture.

        Args:
            causal_graph: CausalGraph test fixture
        """
        snap = causal_graph.create_snapshot()
        assert snap["num_nodes"] == 5
        assert snap["num_edges"] == 4

    def test_snapshot_history_bounded(self, config):
        """
        Verify snapshot history size limits to avoid memory growth leaks.

        Args:
            config: Test configuration fixture
        """
        from tracing.causal_graph import CausalGraph
        cg = CausalGraph(config)
        cg.add_dependency("a", "b")
        for _ in range(150):
            cg.create_snapshot()
        assert len(cg.snapshots) <= 100

    def test_export_graph_valid_json(self, causal_graph):
        """
        Verify that graph export generates syntactically valid JSON payload.

        Args:
            causal_graph: CausalGraph test fixture
        """
        data = json.loads(causal_graph.export_graph())
        assert "nodes" in data
        assert "edges" in data
        assert len(data["nodes"]) == 5
        assert len(data["edges"]) == 4

    def test_export_node_fields(self, causal_graph):
        """
        Verify node attribute properties exist in exported payload.

        Args:
            causal_graph: CausalGraph test fixture
        """
        data = json.loads(causal_graph.export_graph())
        node = data["nodes"][0]
        assert "id" in node
        assert "is_anomalous" in node

    def test_export_edge_fields(self, causal_graph):
        """
        Verify edge weight and layout metadata exist in exported payload.

        Args:
            causal_graph: CausalGraph test fixture
        """
        data = json.loads(causal_graph.export_graph())
        edge = data["edges"][0]
        assert "source" in edge
        assert "target" in edge
        assert "weight" in edge


class TestGraphStatistics:
    """
    Validates graph metrics statistics aggregation.
    """

    def test_statistics(self, causal_graph):
        """
        Verify statistical summary matches graph metadata counts.

        Args:
            causal_graph: CausalGraph test fixture
        """
        stats = causal_graph.get_statistics()
        assert stats["num_nodes"] == 5
        assert stats["num_edges"] == 4
        assert stats["avg_degree"] > 0

    def test_empty_graph_statistics(self, config):
        """
        Verify statistical summary on empty graphs yields zero values.

        Args:
            config: Test configuration fixture
        """
        from tracing.causal_graph import CausalGraph
        cg = CausalGraph(config)
        stats = cg.get_statistics()
        assert stats["num_nodes"] == 0
        assert stats["num_edges"] == 0
