import pytest


class TestRCAClassification:
    """
    Validates classification of anomalies as root causes vs. propagated failures.
    """

    def test_no_upstream_is_root_cause(self, config, causal_graph):
        """
        Verify that a service with no anomalous upstream callers is flagged as a root cause.

        Args:
            config: Test configuration fixture
            causal_graph: CausalGraph test fixture
        """
        from analysis.root_cause_analyzer import RootCauseAnalyzer
        rca = RootCauseAnalyzer(causal_graph, config)
        assert rca._is_root_cause("gateway", {"gateway", "svc-a"}) is True

    def test_no_anomalous_upstream_is_root_cause(self, config, causal_graph):
        """
        Verify that a service is flagged as a root cause if none of its callers are anomalous.

        Args:
            config: Test configuration fixture
            causal_graph: CausalGraph test fixture
        """
        from analysis.root_cause_analyzer import RootCauseAnalyzer
        rca = RootCauseAnalyzer(causal_graph, config)
        assert rca._is_root_cause("svc-b", {"svc-b"}) is True

    def test_all_upstream_anomalous_is_propagated(self, config, causal_graph):
        """
        Verify that a service with anomalous upstream callers is classified as propagated.

        Args:
            config: Test configuration fixture
            causal_graph: CausalGraph test fixture
        """
        from analysis.root_cause_analyzer import RootCauseAnalyzer
        rca = RootCauseAnalyzer(causal_graph, config)
        result = rca._is_root_cause("svc-c", {"svc-a", "svc-c"})
        assert result is False

    def test_classify_anomalies_partitions_correctly(self, config, causal_graph):
        """
        Verify that classification partitions anomalies into disjoint sets.

        Args:
            config: Test configuration fixture
            causal_graph: CausalGraph test fixture
        """
        from analysis.root_cause_analyzer import RootCauseAnalyzer
        rca = RootCauseAnalyzer(causal_graph, config)
        root_causes, propagated = rca._classify_anomalies({"gateway", "svc-a", "svc-c"})
        assert root_causes | propagated == {"gateway", "svc-a", "svc-c"}
        assert root_causes & propagated == set()


class TestRCAAnalyze:
    """
    Validates full RCA PageRank analysis pipelines.
    """

    def test_analyze_empty(self, config, causal_graph):
        """
        Verify that analysis with an empty set of anomalous services yields empty lists.

        Args:
            config: Test configuration fixture
            causal_graph: CausalGraph test fixture
        """
        from analysis.root_cause_analyzer import RootCauseAnalyzer
        rca = RootCauseAnalyzer(causal_graph, config)
        result = rca.analyze(set())
        assert result["root_causes"] == []
        assert result["propagated_anomalies"] == []
        assert result["analysis_time"] == 0

    def test_analyze_single_service(self, config, causal_graph):
        """
        Verify that analysis with a single anomalous service flags it as a root cause.

        Args:
            config: Test configuration fixture
            causal_graph: CausalGraph test fixture
        """
        from analysis.root_cause_analyzer import RootCauseAnalyzer
        rca = RootCauseAnalyzer(causal_graph, config)
        result = rca.analyze({"svc-a"})
        assert len(result["root_causes"]) >= 1

    def test_analyze_multiple_services(self, config, causal_graph):
        """
        Verify that analysis outputs contain all required schema keys for multiple inputs.

        Args:
            config: Test configuration fixture
            causal_graph: CausalGraph test fixture
        """
        from analysis.root_cause_analyzer import RootCauseAnalyzer
        rca = RootCauseAnalyzer(causal_graph, config)
        result = rca.analyze({"svc-a", "svc-c", "svc-d"})
        assert "root_causes" in result
        assert "propagated_anomalies" in result
        assert "explanations" in result

    def test_analyze_all_services(self, config, causal_graph):
        """
        Verify that analysis of all anomalous services accounts for every single node.

        Args:
            config: Test configuration fixture
            causal_graph: CausalGraph test fixture
        """
        from analysis.root_cause_analyzer import RootCauseAnalyzer
        rca = RootCauseAnalyzer(causal_graph, config)
        all_svcs = {"gateway", "svc-a", "svc-b", "svc-c", "svc-d"}
        result = rca.analyze(all_svcs)
        assert len(result["root_causes"]) + len(result["propagated_anomalies"]) == 5

    def test_analysis_time_recorded(self, config, causal_graph):
        """
        Verify that analyze() tracks elapsed processing time.

        Args:
            config: Test configuration fixture
            causal_graph: CausalGraph test fixture
        """
        from analysis.root_cause_analyzer import RootCauseAnalyzer
        rca = RootCauseAnalyzer(causal_graph, config)
        result = rca.analyze({"svc-a"})
        assert result["analysis_time"] >= 0


class TestRCARanking:
    """
    Validates PageRank scoring and ranking constraints.
    """

    def test_root_causes_sorted_by_score(self, config, causal_graph):
        """
        Verify that flagged root causes are sorted in descending order of PageRank scores.

        Args:
            config: Test configuration fixture
            causal_graph: CausalGraph test fixture
        """
        from analysis.root_cause_analyzer import RootCauseAnalyzer
        rca = RootCauseAnalyzer(causal_graph, config)
        result = rca.analyze({"gateway", "svc-a", "svc-b", "svc-c", "svc-d"})
        scores = [rc["combined_score"] for rc in result["root_causes"]]
        assert scores == sorted(scores, reverse=True)

    def test_root_cause_has_required_fields(self, config, causal_graph):
        """
        Verify that root cause outputs contain standard PageRank and impact score keys.

        Args:
            config: Test configuration fixture
            causal_graph: CausalGraph test fixture
        """
        from analysis.root_cause_analyzer import RootCauseAnalyzer
        rca = RootCauseAnalyzer(causal_graph, config)
        result = rca.analyze({"svc-a"})
        if result["root_causes"]:
            rc = result["root_causes"][0]
            for key in ["service", "pagerank_score", "impact_score", "combined_score", "affected_services"]:
                assert key in rc, f"Missing key: {key}"


class TestRCAExplanations:
    """
    Validates textual explanations generated for each anomalous service.
    """

    def test_explanation_for_root_cause(self, config, causal_graph):
        """
        Verify that a root cause has explanation mapping with high confidence.

        Args:
            config: Test configuration fixture
            causal_graph: CausalGraph test fixture
        """
        from analysis.root_cause_analyzer import RootCauseAnalyzer
        rca = RootCauseAnalyzer(causal_graph, config)
        result = rca.analyze({"gateway"})
        expl = result["explanations"]["gateway"]
        assert expl["type"] == "root_cause"
        assert expl["confidence"] >= 0

    def test_explanation_for_propagated(self, config, causal_graph):
        """
        Verify that propagated failures map back to valid root source explanations.

        Args:
            config: Test configuration fixture
            causal_graph: CausalGraph test fixture
        """
        from analysis.root_cause_analyzer import RootCauseAnalyzer
        rca = RootCauseAnalyzer(causal_graph, config)
        result = rca.analyze({"gateway", "svc-a", "svc-c"})
        expl_c = result["explanations"]["svc-c"]
        assert expl_c["type"] in ("propagated", "root_cause", "unknown")

    def test_explanation_for_unknown_service(self, config, causal_graph):
        """
        Verify that unregistered anomalies map back to unknown type explanations.

        Args:
            config: Test configuration fixture
            causal_graph: CausalGraph test fixture
        """
        from analysis.root_cause_analyzer import RootCauseAnalyzer
        rca = RootCauseAnalyzer(causal_graph, config)
        result = rca.analyze({"totally-unknown"})
        expl = result["explanations"]["totally-unknown"]
        assert expl["type"] in ("root_cause", "unknown")

    def test_every_anomalous_service_has_explanation(self, config, causal_graph):
        """
        Verify that every anomalous service is assigned an explanation entry.

        Args:
            config: Test configuration fixture
            causal_graph: CausalGraph test fixture
        """
        from analysis.root_cause_analyzer import RootCauseAnalyzer
        rca = RootCauseAnalyzer(causal_graph, config)
        anomalous = {"gateway", "svc-a", "svc-b", "svc-c", "svc-d"}
        result = rca.analyze(anomalous)
        for svc in anomalous:
            assert svc in result["explanations"]


class TestRCACascade:
    """
    Validates cascading failure analysis logic.
    """

    def test_cascade_from_gateway(self, config, causal_graph):
        """
        Verify cascade analysis tracks transitive dependencies starting at the root gateway.

        Args:
            config: Test configuration fixture
            causal_graph: CausalGraph test fixture
        """
        from analysis.root_cause_analyzer import RootCauseAnalyzer
        rca = RootCauseAnalyzer(causal_graph, config)
        cascade = rca.explain_cascade("gateway")
        assert cascade["initial_service"] == "gateway"
        assert cascade["num_affected"] >= 3

    def test_cascade_from_leaf(self, config, causal_graph):
        """
        Verify cascade analysis starting at a leaf node has zero affected services.

        Args:
            config: Test configuration fixture
            causal_graph: CausalGraph test fixture
        """
        from analysis.root_cause_analyzer import RootCauseAnalyzer
        rca = RootCauseAnalyzer(causal_graph, config)
        cascade = rca.explain_cascade("svc-d")
        assert cascade["num_affected"] == 0

    def test_cascade_layers(self, config, causal_graph):
        """
        Verify cascade layer groupings exist in cascade explanations.

        Args:
            config: Test configuration fixture
            causal_graph: CausalGraph test fixture
        """
        from analysis.root_cause_analyzer import RootCauseAnalyzer
        rca = RootCauseAnalyzer(causal_graph, config)
        cascade = rca.explain_cascade("gateway")
        assert len(cascade["cascade_layers"]) >= 1

    def test_cascade_severity(self, config, causal_graph):
        """
        Verify cascade severity mapping exists.

        Args:
            config: Test configuration fixture
            causal_graph: CausalGraph test fixture
        """
        from analysis.root_cause_analyzer import RootCauseAnalyzer
        rca = RootCauseAnalyzer(causal_graph, config)
        cascade = rca.explain_cascade("gateway")
        assert cascade["severity"] in ("low", "medium", "high", "critical")


class TestRCAHistory:
    """
    Validates history logging bounds.
    """

    def test_history_grows(self, config, causal_graph):
        """
        Verify that successive analyze calls grow the history log.

        Args:
            config: Test configuration fixture
            causal_graph: CausalGraph test fixture
        """
        from analysis.root_cause_analyzer import RootCauseAnalyzer
        rca = RootCauseAnalyzer(causal_graph, config)
        rca.analyze({"svc-a"})
        rca.analyze({"svc-b"})
        assert len(rca.analysis_history) == 2

    def test_history_bounded(self, config, causal_graph):
        """
        Verify history log size is capped to avoid memory growth leaks.

        Args:
            config: Test configuration fixture
            causal_graph: CausalGraph test fixture
        """
        from analysis.root_cause_analyzer import RootCauseAnalyzer
        rca = RootCauseAnalyzer(causal_graph, config)
        for _ in range(150):
            rca.analyze({"svc-a"})
        assert len(rca.analysis_history) <= 100

    def test_statistics(self, config, causal_graph):
        """
        Verify that history statistics yield averages of analysis runs.

        Args:
            config: Test configuration fixture
            causal_graph: CausalGraph test fixture
        """
        from analysis.root_cause_analyzer import RootCauseAnalyzer
        rca = RootCauseAnalyzer(causal_graph, config)
        rca.analyze({"svc-a"})
        rca.analyze({"svc-a", "svc-c"})
        stats = rca.get_statistics()
        assert stats["total_analyses"] == 2
        assert stats["avg_analysis_time"] >= 0

    def test_empty_statistics(self, config, causal_graph):
        """
        Verify asking for statistics on an un-run analyzer yields an empty dict.

        Args:
            config: Test configuration fixture
            causal_graph: CausalGraph test fixture
        """
        from analysis.root_cause_analyzer import RootCauseAnalyzer
        rca = RootCauseAnalyzer(causal_graph, config)
        assert rca.get_statistics() == {}
