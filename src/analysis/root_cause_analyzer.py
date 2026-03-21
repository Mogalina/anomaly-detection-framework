import networkx as nx
from typing import Dict, List, Optional, Set, Tuple
import time

from tracing.causal_graph import CausalGraph
from utils.logger import get_logger
from utils.config import get_config
from utils.metrics import get_metrics


logger = get_logger(__name__)


class RootCauseAnalyzer:
    """
    Root cause analysis engine using PageRank on causal graph.
    
    Identifies root causes of anomalies by analyzing propagation
    patterns in the service dependency graph.
    """
    
    def __init__(
        self,
        causal_graph: CausalGraph,
        config: Optional[Dict] = None
    ):
        """
        Initialize root cause analyzer.
        
        Args:
            causal_graph: Causal dependency graph
            config: Configuration dictionary
        """
        self.causal_graph = causal_graph
        self.config = config or get_config().to_dict()
        self.metrics = get_metrics()
        
        rca_config = self.config.get('root_cause', {})
        
        # PageRank parameters
        pagerank_config = rca_config.get('pagerank', {})
        self.alpha = pagerank_config.get('alpha', 0.85)
        self.max_iterations = pagerank_config.get('max_iterations', 100)
        self.tolerance = pagerank_config.get('tolerance', 1.0e-6)
        self.personalization_weight = pagerank_config.get('personalization_weight', 0.7)
        
        # Classification parameters
        classification_config = rca_config.get('classification', {})
        self.propagation_threshold = classification_config.get('propagation_threshold', 0.3)
        self.max_hops = classification_config.get('max_hops', 5)
        self.min_impact_score = classification_config.get('min_impact_score', 0.1)
        
        # Explanation parameters
        explanation_config = rca_config.get('explanation', {})
        self.max_chain_length = explanation_config.get('max_chain_length', 10)
        self.confidence_threshold = explanation_config.get('confidence_threshold', 0.6)
        
        # Analysis history
        self.analysis_history = []
        
        logger.info("RootCauseAnalyzer initialized")
    
    def analyze(self, anomalous_services: Set[str]) -> Dict:
        """
        Perform root cause analysis for anomalous services.
        
        Args:
            anomalous_services: Set of services with detected anomalies
        
        Returns:
            Analysis results with root causes and explanations
        """
        start_time = time.time()
        
        if not anomalous_services:
            return {
                'root_causes': [],
                'propagated_anomalies': [],
                'analysis_time': 0
            }
        
        logger.info(f"Analyzing root causes for {len(anomalous_services)} anomalous services")
        
        # Classify anomalies
        root_causes, propagated = self._classify_anomalies(anomalous_services)
        
        # Rank root causes
        ranked_root_causes = self._rank_root_causes(root_causes)
        
        # Generate explanations
        explanations = {}
        for service in anomalous_services:
            explanation = self._generate_explanation(service, ranked_root_causes)
            explanations[service] = explanation
        
        analysis_time = time.time() - start_time
        
        # Record metrics
        self.metrics.record_rca_latency(analysis_time)
        for _ in ranked_root_causes:
            self.metrics.record_root_cause()
        
        result = {
            'timestamp': time.time(),
            'anomalous_services': list(anomalous_services),
            'root_causes': ranked_root_causes,
            'propagated_anomalies': list(propagated),
            'explanations': explanations,
            'analysis_time': analysis_time
        }
        
        # Store in history
        self.analysis_history.append(result)
        if len(self.analysis_history) > 100:
            self.analysis_history = self.analysis_history[-100:]
        
        logger.info(
            f"Root cause analysis completed: "
            f"{len(ranked_root_causes)} root causes, "
            f"{len(propagated)} propagated anomalies, "
            f"time={analysis_time:.2f}s"
        )
        
        return result
    
    def _classify_anomalies(
        self,
        anomalous_services: Set[str]
    ) -> Tuple[Set[str], Set[str]]:
        """
        Classify anomalies as root causes or propagated.
        
        Args:
            anomalous_services: Set of anomalous services
        
        Returns:
            Tuple of (root_causes, propagated_anomalies)
        """
        root_causes = set()
        propagated = set()
        
        for service in anomalous_services:
            if self._is_root_cause(service, anomalous_services):
                root_causes.add(service)
            else:
                propagated.add(service)
        
        return root_causes, propagated
    
    def _is_root_cause(self, service: str, anomalous_services: Set[str]) -> bool:
        """
        Determine if a service is a root cause.
        
        A service is considered a root cause if:
        - It has no anomalous upstream dependencies, OR
        - The fraction of anomalous upstream dependencies is below threshold
        
        Args:
            service: Service to check
            anomalous_services: All anomalous services
        
        Returns:
            True if service is likely a root cause
        """
        upstream = self.causal_graph.get_upstream_services(service, max_hops=2)
        
        if not upstream:
            # No upstream dependencies - likely root cause
            return True
        
        anomalous_upstream = upstream & anomalous_services
        
        if not anomalous_upstream:
            # No anomalous upstream - definitely root cause
            return True
        
        # Check fraction of anomalous upstream
        fraction = len(anomalous_upstream) / len(upstream)
        
        return fraction < self.propagation_threshold
    
    def _rank_root_causes(self, root_causes: Set[str]) -> List[Dict]:
        """
        Rank root causes by impact using PageRank.
        
        Args:
            root_causes: Set of root cause services
        
        Returns:
            Ranked list of root causes with scores
        """
        if not root_causes:
            return []
        
        # Create personalization vector for PageRank
        # Higher weight for known root causes
        personalization = {}
        for node in self.causal_graph.graph.nodes():
            if node in root_causes:
                personalization[node] = self.personalization_weight
            else:
                personalization[node] = (1 - self.personalization_weight) / \
                                        (self.causal_graph.graph.number_of_nodes() - len(root_causes))
        
        try:
            # Compute PageRank
            pagerank_scores = nx.pagerank(
                self.causal_graph.graph,
                alpha=self.alpha,
                personalization=personalization,
                max_iter=self.max_iterations,
                tol=self.tolerance
            )
            
            # Filter and rank root causes
            ranked = []
            for service in root_causes:
                impact_score = self.causal_graph.get_impact_score(service)
                pagerank_score = pagerank_scores.get(service, 0)
                
                # Combined score
                combined_score = 0.6 * pagerank_score + 0.4 * (impact_score / 100)
                
                if combined_score >= self.min_impact_score:
                    downstream = self.causal_graph.get_downstream_services(
                        service,
                        max_hops=self.max_hops
                    )
                    
                    ranked.append({
                        'service': service,
                        'pagerank_score': pagerank_score,
                        'impact_score': impact_score,
                        'combined_score': combined_score,
                        'affected_services': len(downstream),
                        'downstream_services': list(downstream)
                    })
            
            # Sort by combined score
            ranked.sort(key=lambda x: x['combined_score'], reverse=True)
            
            return ranked
        
        except Exception as e:
            logger.error(f"Error computing PageRank: {e}")
            # Fallback: rank by impact score only
            return [
                {
                    'service': service,
                    'impact_score': self.causal_graph.get_impact_score(service),
                    'combined_score': 0,
                    'affected_services': 0
                }
                for service in root_causes
            ]
    
    def _generate_explanation(
        self,
        service: str,
        root_causes: List[Dict]
    ) -> Dict:
        """
        Generate explanation for why a service is anomalous.
        
        Args:
            service: Service to explain
            root_causes: Ranked root causes
        
        Returns:
            Explanation dictionary
        """
        # Find if service is in root causes
        is_root_cause = any(rc['service'] == service for rc in root_causes)
        
        if is_root_cause:
            # Explain as root cause
            root_cause_info = next(rc for rc in root_causes if rc['service'] == service)

            # Confidence based on combined score (0-1)
            confidence = min(1.0, root_cause_info.get('combined_score', 0))
            
            return {
                'type': 'root_cause',
                'service': service,
                'confidence': confidence,
                'message': (
                    f"{service} is identified as a root cause affecting "
                    f"{root_cause_info.get('affected_services', 0)} downstream services"
                ),
                'impact_score': root_cause_info.get('impact_score', 0),
                'downstream_services': root_cause_info.get('downstream_services', [])
            }
        
        # Find propagation paths from root causes
        propagation_paths = []
        
        for root_cause in root_causes:
            path = self.causal_graph.get_propagation_path(
                root_cause['service'],
                service
            )
            
            if path and len(path) <= self.max_chain_length:
                propagation_paths.append({
                    'root_cause': root_cause['service'],
                    'path': path,
                    'path_length': len(path) - 1,
                    'root_cause_score': root_cause.get('combined_score', 0)
                })
        
        if propagation_paths:
            # Sort by path length and root cause score
            propagation_paths.sort(
                key=lambda x: (x['path_length'], -x['root_cause_score'])
            )
            
            best_path = propagation_paths[0]
            confidence = min(
                0.9,
                best_path['root_cause_score'] / (best_path['path_length'] + 1)
            )
            
            return {
                'type': 'propagated',
                'service': service,
                'confidence': confidence,
                'message': (
                    f"{service} anomaly likely propagated from {best_path['root_cause']} "
                    f"via path: {' -> '.join(best_path['path'])}"
                ),
                'root_cause': best_path['root_cause'],
                'propagation_path': best_path['path'],
                'path_length': best_path['path_length'],
                'alternative_paths': len(propagation_paths) - 1
            }

        upstream_services = self.causal_graph.get_upstream_services(service, max_hops=2)

        # Estimate confidence from graph structure
        num_upstream = len(upstream_services)

        # If many upstream services exist, uncertainty increases
        confidence = max(0.1, min(0.5, 1 / (num_upstream + 1)))

        # No clear propagation path found
        return {
            'type': 'unknown',
            'service': service,
            'confidence': confidence,
            'message': f"Anomaly in {service} detected but propagation source unclear",
            'upstream_services': list(upstream_services)
        }
    
    def explain_cascade(self, initial_service: str) -> Dict:
        """
        Explain potential cascade effects from a service failure.
        
        Args:
            initial_service: Service that might fail
        
        Returns:
            Cascade analysis
        """
        downstream = self.causal_graph.get_downstream_services(
            initial_service,
            max_hops=self.max_hops
        )
        
        # Build cascade layers
        layers = self._build_cascade_layers(initial_service)
        
        # Estimate impact
        total_impact = self.causal_graph.get_impact_score(initial_service)
        
        return {
            'initial_service': initial_service,
            'potentially_affected': list(downstream),
            'num_affected': len(downstream),
            'cascade_layers': layers,
            'total_impact_score': total_impact,
            'severity': self._compute_severity(total_impact)
        }
    
    def _build_cascade_layers(self, service: str) -> List[List[str]]:
        """
        Build layers of cascade propagation.
        
        Args:
            service: Initial service
        
        Returns:
            List of layers, each containing services at that distance
        """
        if service not in self.causal_graph.graph:
            return []
        
        layers = []
        visited = {service}
        current_layer = [service]
        
        for hop in range(self.max_hops):
            next_layer = []
            
            for svc in current_layer:
                for neighbor in self.causal_graph.graph.successors(svc):
                    if neighbor not in visited:
                        visited.add(neighbor)
                        next_layer.append(neighbor)
            
            if not next_layer:
                break
            
            layers.append(next_layer)
            current_layer = next_layer
        
        return layers
    
    def _compute_severity(self, impact_score: float) -> str:
        """
        Compute severity level from impact score.
        
        Args:
            impact_score: Impact score
        
        Returns:
            Severity level
        """
        if impact_score < 5:
            return 'low'
        elif impact_score < 15:
            return 'medium'
        elif impact_score < 30:
            return 'high'
        else:
            return 'critical'
    
    def get_statistics(self) -> Dict:
        """
        Get analyzer statistics.
        
        Returns:
            Statistics dictionary
        """
        if not self.analysis_history:
            return {}
        
        total_analyses = len(self.analysis_history)
        total_root_causes = sum(
            len(analysis['root_causes'])
            for analysis in self.analysis_history
        )
        
        avg_analysis_time = sum(
            analysis['analysis_time'] 
            for analysis in self.analysis_history
        ) / total_analyses
        
        return {
            'total_analyses': total_analyses,
            'total_root_causes_identified': total_root_causes,
            'avg_root_causes_per_analysis': total_root_causes / total_analyses,
            'avg_analysis_time': avg_analysis_time
        }
