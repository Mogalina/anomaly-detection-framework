import networkx as nx
from typing import Dict, List, Set, Optional, Tuple
import time
from collections import defaultdict
import json

from utils.logger import get_logger
from utils.config import get_config
from utils.metrics import get_metrics


logger = get_logger(__name__)


class CausalGraph:
    """
    Dynamic service dependency graph for causal analysis.
    
    Builds and maintains a directed graph of service dependencies
    with edge weights representing call frequency and latency.
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize causal graph.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or get_config().to_dict()
        self.metrics = get_metrics()
        
        graph_config = self.config.get('tracing', {}).get('graph', {})
        
        # Graph parameters
        self.update_interval = graph_config.get('update_interval', 60)
        self.edge_weight_decay = graph_config.get('edge_weight_decay', 0.95)
        self.min_edge_weight = graph_config.get('min_edge_weight', 0.01)
        self.snapshot_interval = graph_config.get('snapshot_interval', 300)
        
        # Initialize graph
        self.graph = nx.DiGraph()
        
        # Edge attributes tracking
        self.edge_call_counts = defaultdict(int)
        self.edge_latencies = defaultdict(list)
        
        # Anomaly tracking
        self.anomalous_services: Set[str] = set()
        self.anomaly_timestamps: Dict[str, float] = {}
        
        # Snapshots
        self.snapshots = []
        self.last_snapshot_time = time.time()
        self.last_update_time = time.time()
        
        logger.info("CausalGraph initialized")
    
    def add_service(self, service: str, metadata: Optional[Dict] = None) -> None:
        """
        Add a service node to the graph.
        
        Args:
            service: Service name
            metadata: Optional service metadata
        """
        if service not in self.graph:
            self.graph.add_node(service, metadata=metadata or {})
            logger.debug(f"Added service node: {service}")
    
    def add_dependency(
        self,
        from_service: str,
        to_service: str,
        call_count: int = 1,
        latency: Optional[float] = None
    ) -> None:
        """
        Add or update a service dependency edge.
        
        Args:
            from_service: Source service
            to_service: Target service
            call_count: Number of calls
            latency: Optional call latency
        """
        # Ensure nodes exist
        self.add_service(from_service)
        self.add_service(to_service)
        
        edge = (from_service, to_service)
        
        # Update call count
        self.edge_call_counts[edge] += call_count
        
        # Update latency
        if latency is not None:
            self.edge_latencies[edge].append(latency)
            # Keep only recent latencies
            if len(self.edge_latencies[edge]) > 100:
                self.edge_latencies[edge] = self.edge_latencies[edge][-100:]
        
        # Update or add edge
        if self.graph.has_edge(from_service, to_service):
            # Update existing edge weight
            current_weight = self.graph[from_service][to_service].get('weight', 0)
            new_weight = current_weight * self.edge_weight_decay + call_count
            self.graph[from_service][to_service]['weight'] = new_weight
        else:
            # Add new edge
            self.graph.add_edge(
                from_service,
                to_service,
                weight=call_count,
                created_at=time.time()
            )
        
        # Update edge attributes
        avg_latency = (
            sum(self.edge_latencies[edge]) / len(self.edge_latencies[edge])
            if self.edge_latencies[edge] else None
        )
        
        self.graph[from_service][to_service]['call_count'] = self.edge_call_counts[edge]
        if avg_latency is not None:
            self.graph[from_service][to_service]['avg_latency'] = avg_latency
    
    def update_from_traces(self, traces: List[Dict]) -> None:
        """
        Update graph from trace data.
        
        Args:
            traces: List of parsed traces
        """
        for trace in traces:
            for dep in trace.get('dependencies', []):
                self.add_dependency(
                    from_service=dep['from'],
                    to_service=dep['to'],
                    call_count=1,
                    latency=dep.get('duration')
                )
        
        # Decay edge weights
        self._apply_edge_decay()
        
        # Remove weak edges
        self._prune_weak_edges()
        
        # Update metrics
        self.metrics.set_graph_size(self.graph.number_of_nodes())
        
        self.last_update_time = time.time()
        
        # Create snapshot if needed
        if time.time() - self.last_snapshot_time > self.snapshot_interval:
            self.create_snapshot()
    
    def _apply_edge_decay(self) -> None:
        """Apply time-based decay to edge weights."""
        for u, v in list(self.graph.edges()):
            current_weight = self.graph[u][v].get('weight', 0)
            decayed_weight = current_weight * self.edge_weight_decay
            self.graph[u][v]['weight'] = decayed_weight
    
    def _prune_weak_edges(self) -> None:
        """Remove edges with weight below threshold."""
        edges_to_remove = []
        
        for u, v in self.graph.edges():
            weight = self.graph[u][v].get('weight', 0)
            if weight < self.min_edge_weight:
                edges_to_remove.append((u, v))
        
        self.graph.remove_edges_from(edges_to_remove)
        
        if edges_to_remove:
            logger.debug(f"Pruned {len(edges_to_remove)} weak edges")
    
    def mark_anomaly(self, service: str) -> None:
        """
        Mark a service as anomalous.
        
        Args:
            service: Service name
        """
        if service not in self.graph:
            self.add_service(service)
        
        self.anomalous_services.add(service)
        self.anomaly_timestamps[service] = time.time()
        
        # Update node attribute
        self.graph.nodes[service]['is_anomalous'] = True
        self.graph.nodes[service]['anomaly_time'] = time.time()
        
        logger.info(f"Marked service as anomalous: {service}")
    
    def clear_anomaly(self, service: str) -> None:
        """
        Clear anomaly status for a service.
        
        Args:
            service: Service name
        """
        self.anomalous_services.discard(service)
        if service in self.anomaly_timestamps:
            del self.anomaly_timestamps[service]
        
        if service in self.graph:
            self.graph.nodes[service]['is_anomalous'] = False
    
    def get_downstream_services(self, service: str, max_hops: int = 3) -> Set[str]:
        """
        Get all downstream services within max_hops.
        
        Args:
            service: Source service
            max_hops: Maximum number of hops
        
        Returns:
            Set of downstream services
        """
        if service not in self.graph:
            return set()
        
        downstream = set()
        
        # BFS to find downstream services
        queue = [(service, 0)]
        visited = {service}
        
        while queue:
            current, hops = queue.pop(0)
            
            if hops >= max_hops:
                continue
            
            for neighbor in self.graph.successors(current):
                if neighbor not in visited:
                    visited.add(neighbor)
                    downstream.add(neighbor)
                    queue.append((neighbor, hops + 1))
        
        return downstream
    
    def get_upstream_services(self, service: str, max_hops: int = 3) -> Set[str]:
        """
        Get all upstream services within max_hops.
        
        Args:
            service: Target service
            max_hops: Maximum number of hops
        
        Returns:
            Set of upstream services
        """
        if service not in self.graph:
            return set()
        
        upstream = set()
        
        # BFS backwards
        queue = [(service, 0)]
        visited = {service}
        
        while queue:
            current, hops = queue.pop(0)
            
            if hops >= max_hops:
                continue
            
            for neighbor in self.graph.predecessors(current):
                if neighbor not in visited:
                    visited.add(neighbor)
                    upstream.add(neighbor)
                    queue.append((neighbor, hops + 1))
        
        return upstream
    
    def get_propagation_path(
        self,
        source: str,
        target: str
    ) -> Optional[List[str]]:
        """
        Find shortest propagation path between services.
        
        Args:
            source: Source service
            target: Target service
        
        Returns:
            Path as list of services, or None if no path exists
        """
        if source not in self.graph or target not in self.graph:
            return None
        
        try:
            path = nx.shortest_path(self.graph, source, target)
            return path
        except nx.NetworkXNoPath:
            return None
    
    def get_impact_score(self, service: str) -> float:
        """
        Compute impact score for a service based on downstream dependencies.
        
        Args:
            service: Service name
        
        Returns:
            Impact score
        """
        if service not in self.graph:
            return 0.0
        
        downstream = self.get_downstream_services(service, max_hops=5)
        
        # Weight by number of downstream services and edge weights
        impact = len(downstream)
        
        # Add edge weight contribution
        for neighbor in self.graph.successors(service):
            weight = self.graph[service][neighbor].get('weight', 0)
            impact += weight
        
        return impact
    
    def create_snapshot(self) -> Dict:
        """
        Create a snapshot of the current graph state.
        
        Returns:
            Graph snapshot
        """
        snapshot = {
            'timestamp': time.time(),
            'num_nodes': self.graph.number_of_nodes(),
            'num_edges': self.graph.number_of_edges(),
            'nodes': list(self.graph.nodes()),
            'edges': [
                {
                    'from': u,
                    'to': v,
                    'weight': self.graph[u][v].get('weight', 0),
                    'call_count': self.graph[u][v].get('call_count', 0),
                    'avg_latency': self.graph[u][v].get('avg_latency')
                }
                for u, v in self.graph.edges()
            ],
            'anomalous_services': list(self.anomalous_services)
        }
        
        self.snapshots.append(snapshot)
        self.last_snapshot_time = time.time()
        
        # Keep only recent snapshots
        if len(self.snapshots) > 100:
            self.snapshots = self.snapshots[-100:]
        
        logger.debug(f"Created graph snapshot: {snapshot['num_nodes']} nodes, {snapshot['num_edges']} edges")
        
        return snapshot
    
    def export_graph(self) -> str:
        """
        Export graph in JSON format.
        
        Returns:
            JSON string representation
        """
        data = {
            'nodes': [
                {
                    'id': node,
                    'is_anomalous': self.graph.nodes[node].get('is_anomalous', False),
                    'metadata': self.graph.nodes[node].get('metadata', {})
                }
                for node in self.graph.nodes()
            ],
            'edges': [
                {
                    'source': u,
                    'target': v,
                    'weight': self.graph[u][v].get('weight', 0),
                    'call_count': self.graph[u][v].get('call_count', 0),
                    'avg_latency': self.graph[u][v].get('avg_latency')
                }
                for u, v in self.graph.edges()
            ]
        }
        
        return json.dumps(data, indent=2)
    
    def get_statistics(self) -> Dict:
        """
        Get graph statistics.
        
        Returns:
            Statistics dictionary
        """
        return {
            'num_nodes': self.graph.number_of_nodes(),
            'num_edges': self.graph.number_of_edges(),
            'num_anomalous_services': len(self.anomalous_services),
            'avg_degree': (
                sum(dict(self.graph.degree()).values()) / self.graph.number_of_nodes()
                if self.graph.number_of_nodes() > 0 else 0
            ),
            'is_connected': nx.is_weakly_connected(self.graph) if self.graph.number_of_nodes() > 0 else False,
            'last_update': self.last_update_time
        }
