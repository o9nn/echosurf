import os
import json
import numpy as np
import logging
import datetime
import time
from pathlib import Path
from typing import Dict, List, Set, Tuple, Optional, Any, Union
from enum import Enum
from dataclasses import dataclass, field, asdict
from collections import defaultdict, deque
import networkx as nx
from deep_tree_echo import TreeNode

# Configure logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class MemoryType(Enum):
    DECLARATIVE = "declarative"  # Facts and concepts
    EPISODIC = "episodic"        # Personal experiences
    PROCEDURAL = "procedural"    # How to do things
    SEMANTIC = "semantic"        # General knowledge
    WORKING = "working"          # Short-term active processing
    SENSORY = "sensory"          # Perceptual information
    EMOTIONAL = "emotional"      # Feelings and emotional states
    ASSOCIATIVE = "associative"  # Connections between other memories

@dataclass
class MemoryNode:
    id: str
    content: str
    memory_type: MemoryType
    creation_time: float = field(default_factory=time.time)
    last_access_time: float = field(default_factory=time.time)
    access_count: int = 0
    salience: float = 0.5  # How important/noteworthy the memory is (0-1)
    echo_value: float = 0.0  # Related to Deep Tree Echo values
    source: str = "unknown"  # Where the memory came from
    metadata: Dict[str, Any] = field(default_factory=dict)
    embeddings: Optional[List[float]] = None

    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization"""
        data = asdict(self)
        data['memory_type'] = self.memory_type.value
        return data
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'MemoryNode':
        """Create from dictionary"""
        # Convert string memory_type back to enum
        if isinstance(data['memory_type'], str):
            data['memory_type'] = MemoryType(data['memory_type'])
        return cls(**data)
    
    def access(self):
        """Mark this memory as accessed"""
        self.last_access_time = time.time()
        self.access_count += 1

@dataclass
class MemoryEdge:
    from_id: str
    to_id: str
    relation_type: str
    weight: float = 0.5  # Strength of connection (0-1)
    creation_time: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization"""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'MemoryEdge':
        """Create from dictionary"""
        return cls(**data)

class HypergraphMemory:
    def __init__(self, storage_dir: str = "echo_memory"):
        """Initialize the hypergraph memory system
        
        Args:
            storage_dir: Directory to store memory files
        """
        self.nodes: Dict[str, MemoryNode] = {}
        self.edges: List[MemoryEdge] = []
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        
        # Index structures for efficient retrieval
        self.type_index: Dict[MemoryType, Set[str]] = {mem_type: set() for mem_type in MemoryType}
        self.source_index: Dict[str, Set[str]] = defaultdict(set)
        self.temporal_index: List[Tuple[float, str]] = []  # (timestamp, node_id)
        self.salience_index: List[Tuple[float, str]] = []  # (salience, node_id)
        self.echo_index: List[Tuple[float, str]] = []  # (echo_value, node_id)
        
        # Network representation for graph algorithms
        self.graph = nx.DiGraph()
        
        # Working memory (limited capacity active nodes)
        self.working_memory: deque = deque(maxlen=7)  # Miller's Law: 7±2 items
        
        # Load existing memories if available
        self.load()
    
    def add_node(self, node: MemoryNode) -> str:
        """Add a memory node to the system
        
        Args:
            node: The memory node to add
            
        Returns:
            The node ID
        """
        if node.id in self.nodes:
            logger.warning(f"Node with ID {node.id} already exists, updating")
            
        # Store the node
        self.nodes[node.id] = node
        
        # Update indices
        self.type_index[node.memory_type].add(node.id)
        self.source_index[node.source].add(node.id)
        self.temporal_index.append((node.creation_time, node.id))
        self.salience_index.append((node.salience, node.id))
        self.echo_index.append((node.echo_value, node.id))
        
        # Add to graph for network analysis
        self.graph.add_node(node.id, **{k: v for k, v in node.to_dict().items() 
                                      if k not in ['id', 'embeddings']})
        
        # Sort indices
        self._sort_indices()
        
        return node.id
    
    def remove_node(self, node_id: str) -> bool:
        """Remove a node from the system
        
        Args:
            node_id: ID of the node to remove
            
        Returns:
            True if node was removed, False if not found
        """
        if node_id not in self.nodes:
            return False
            
        node = self.nodes[node_id]
        
        # Remove from indices
        self.type_index[node.memory_type].remove(node_id)
        self.source_index[node.source].remove(node_id)
        
        # Remove from temporal and salience indices
        self.temporal_index = [(t, nid) for t, nid in self.temporal_index if nid != node_id]
        self.salience_index = [(s, nid) for s, nid in self.salience_index if nid != node_id]
        self.echo_index = [(e, nid) for e, nid in self.echo_index if nid != node_id]
        
        # Remove related edges
        self.edges = [edge for edge in self.edges if 
                     edge.from_id != node_id and edge.to_id != node_id]
        
        # Remove from graph
        if node_id in self.graph:
            self.graph.remove_node(node_id)
        
        # Remove from nodes dictionary
        del self.nodes[node_id]
        
        return True
    
    def add_edge(self, edge: MemoryEdge) -> None:
        """Add a connection between memory nodes
        
        Args:
            edge: The memory edge to add
        """
        if edge.from_id not in self.nodes or edge.to_id not in self.nodes:
            logger.warning(f"Cannot add edge: node {edge.from_id} or {edge.to_id} not found")
            return
            
        # Add to edges list
        self.edges.append(edge)
        
        # Add to graph
        self.graph.add_edge(edge.from_id, edge.to_id, 
                           relation=edge.relation_type, 
                           weight=edge.weight,
                           **edge.metadata)
    
    def update_node(self, node_id: str, **kwargs) -> bool:
        """Update node properties
        
        Args:
            node_id: ID of the node to update
            **kwargs: Properties to update
            
        Returns:
            True if node was updated, False if not found
        """
        if node_id not in self.nodes:
            return False
            
        node = self.nodes[node_id]
        
        # Update specified properties
        for key, value in kwargs.items():
            if hasattr(node, key):
                setattr(node, key, value)
        
        # Mark as accessed
        node.access()
        
        # Update indices that might have changed
        if 'salience' in kwargs:
            self.salience_index = [(s, nid) for s, nid in self.salience_index if nid != node_id]
            self.salience_index.append((node.salience, node_id))
            self._sort_indices()
            
        if 'echo_value' in kwargs:
            self.echo_index = [(e, nid) for e, nid in self.echo_index if nid != node_id]
            self.echo_index.append((node.echo_value, node_id))
            self._sort_indices()
            
        # Update graph node
        if node_id in self.graph:
            for key, value in kwargs.items():
                if key != 'id' and key != 'embeddings':
                    self.graph.nodes[node_id][key] = value
        
        return True
    
    def get_node(self, node_id: str) -> Optional[MemoryNode]:
        """Get a node by ID
        
        Args:
            node_id: The ID of the node to retrieve
            
        Returns:
            The memory node or None if not found
        """
        if node_id in self.nodes:
            node = self.nodes[node_id]
            node.access()  # Mark as accessed
            
            # Add to working memory
            if node_id not in self.working_memory:
                if len(self.working_memory) == self.working_memory.maxlen:
                    self.working_memory.popleft()  # Remove oldest item if full
                self.working_memory.append(node_id)
                
            return node
        return None
    
    def find_nodes(self, **filters) -> List[MemoryNode]:
        """Find nodes matching given filters
        
        Args:
            **filters: Filter criteria (memory_type, source, min_salience, etc.)
            
        Returns:
            List of matching memory nodes
        """
        results = set(self.nodes.keys())
        
        # Filter by memory type
        if 'memory_type' in filters:
            mem_type = filters['memory_type']
            if isinstance(mem_type, str):
                mem_type = MemoryType(mem_type)
            type_nodes = self.type_index.get(mem_type, set())
            results = results.intersection(type_nodes)
            
        # Filter by source
        if 'source' in filters:
            source_nodes = self.source_index.get(filters['source'], set())
            results = results.intersection(source_nodes)
            
        # Filter by min_salience
        if 'min_salience' in filters:
            min_sal = filters['min_salience']
            high_salience = {nid for _, nid in self.salience_index if _ >= min_sal}
            results = results.intersection(high_salience)
            
        # Filter by min_echo
        if 'min_echo' in filters:
            min_echo = filters['min_echo']
            high_echo = {nid for _, nid in self.echo_index if _ >= min_echo}
            results = results.intersection(high_echo)
            
        # Filter by time range
        if 'start_time' in filters or 'end_time' in filters:
            start = filters.get('start_time', 0)
            end = filters.get('end_time', float('inf'))
            time_nodes = {nid for t, nid in self.temporal_index if start <= t <= end}
            results = results.intersection(time_nodes)
            
        # Convert to nodes and sort by salience
        nodes = [self.nodes[nid] for nid in results]
        return sorted(nodes, key=lambda n: n.salience, reverse=True)
    
    def get_related_nodes(self, node_id: str, relation_type: Optional[str] = None,
                         max_depth: int = 1) -> List[MemoryNode]:
        """Get nodes related to the given node
        
        Args:
            node_id: ID of the source node
            relation_type: Optional filter for edge relation type
            max_depth: Maximum traversal depth (1 = direct connections only)
            
        Returns:
            List of related memory nodes
        """
        if node_id not in self.graph:
            return []
            
        if max_depth == 1:
            # Get direct connections only
            neighbors = list(self.graph.neighbors(node_id))
            
            # Filter by relation type if specified
            if relation_type:
                neighbors = [
                    n for n in neighbors 
                    if self.graph.edges[node_id, n].get('relation') == relation_type
                ]
                
            return [self.nodes[nid] for nid in neighbors if nid in self.nodes]
        else:
            # Use BFS to find nodes up to max_depth
            visited = set()
            queue = deque([(node_id, 0)])  # (node_id, depth)
            related_ids = []
            
            while queue:
                current_id, depth = queue.popleft()
                
                if current_id != node_id:
                    related_ids.append(current_id)
                
                if depth < max_depth:
                    for neighbor in self.graph.neighbors(current_id):
                        if neighbor not in visited:
                            # Check relation type if specified
                            if relation_type is None or self.graph.edges[current_id, neighbor].get('relation') == relation_type:
                                visited.add(neighbor)
                                queue.append((neighbor, depth + 1))
            
            return [self.nodes[nid] for nid in related_ids if nid in self.nodes]
    
    def find_paths(self, from_id: str, to_id: str, max_length: int = 5) -> List[List[str]]:
        """Find all paths between two nodes up to a maximum length
        
        Args:
            from_id: Starting node ID
            to_id: Target node ID
            max_length: Maximum path length
            
        Returns:
            List of paths (each path is a list of node IDs)
        """
        if from_id not in self.graph or to_id not in self.graph:
            return []
            
        try:
            paths = list(nx.all_simple_paths(
                self.graph, from_id, to_id, cutoff=max_length
            ))
            return paths
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            return []
    
    def find_strongest_path(self, from_id: str, to_id: str, max_length: int = 5) -> Optional[List[str]]:
        """Find the strongest path between two nodes based on edge weights
        
        Args:
            from_id: Starting node ID
            to_id: Target node ID
            max_length: Maximum path length
            
        Returns:
            The strongest path as a list of node IDs, or None if no path exists
        """
        if from_id not in self.graph or to_id not in self.graph:
            return None

        try:
            # Use Dijkstra's algorithm with negative weights (we want strongest path)
            # NetworkX finds shortest paths, so we invert weights
            inverted_graph = nx.DiGraph()
            for u, v, data in self.graph.edges(data=True):
                # Skip paths longer than max_length
                if len(nx.shortest_path(self.graph, from_id, u)) > max_length:
                    continue
                inverted_graph.add_edge(u, v, weight=1.0 - data.get('weight', 0.5))
                
            path = nx.shortest_path(inverted_graph, from_id, to_id, weight='weight')
            return path
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            return None
    
    def find_communities(self) -> Dict[int, List[str]]:
        """Find communities of densely connected nodes
        
        Returns:
            Dictionary mapping community ID to list of node IDs
        """
        if len(self.graph) < 3:
            # Not enough nodes for meaningful community detection
            return {0: list(self.graph.nodes())}
            
        # Use Louvain method for community detection
        try:
            from community import best_partition
            partition = best_partition(nx.Graph(self.graph))
            
            # Group nodes by community
            communities = defaultdict(list)
            for node, comm_id in partition.items():
                communities[comm_id].append(node)
                
            return dict(communities)
        except ImportError:
            # Fall back to connected components if community package not available
            logger.warning("Community detection package not available, falling back to connected components")
            components = list(nx.weakly_connected_components(self.graph))
            return {i: list(comp) for i, comp in enumerate(components)}
    
    def compute_centrality(self) -> Dict[str, float]:
        """Compute node centrality in the graph
        
        Returns:
            Dictionary mapping node ID to centrality score
        """
        if not self.graph.nodes():
            return {}
            
        # Use eigenvector centrality
        try:
            centrality = nx.eigenvector_centrality(self.graph, weight='weight', max_iter=1000)
            return centrality
        except (nx.PowerIterationFailedConvergence, nx.NetworkXPointlessConcept):
            # Fall back to degree centrality if eigenvector centrality fails
            logger.warning("Eigenvector centrality failed, falling back to degree centrality")
            return nx.degree_centrality(self.graph)
    
    def update_salience_by_centrality(self):
        """Update node salience based on graph centrality"""
        centrality = self.compute_centrality()
        for node_id, centrality_score in centrality.items():
            if node_id in self.nodes:
                # Blend current salience with centrality
                current_salience = self.nodes[node_id].salience
                new_salience = 0.7 * current_salience + 0.3 * centrality_score
                self.update_node(node_id, salience=new_salience)
    
    def prune_by_salience(self, threshold: float = 0.1):
        """Remove low-salience nodes from memory
        
        Args:
            threshold: Minimum salience value to keep
        """
        low_salience_nodes = [nid for sal, nid in self.salience_index if sal < threshold]
        for node_id in low_salience_nodes:
            self.remove_node(node_id)
    
    def save(self):
        """Save the memory graph to disk"""
        # Create the storage directory if it doesn't exist
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        
        # Save nodes
        nodes_file = self.storage_dir / 'nodes.json'
        nodes_data = {node_id: node.to_dict() for node_id, node in self.nodes.items()}
        with open(nodes_file, 'w') as f:
            json.dump(nodes_data, f, indent=2)
            
        # Save edges
        edges_file = self.storage_dir / 'edges.json'
        edges_data = [edge.to_dict() for edge in self.edges]
        with open(edges_file, 'w') as f:
            json.dump(edges_data, f, indent=2)
            
        # Save indices (just basic info, will be rebuilt on load)
        indices_file = self.storage_dir / 'indices.json'
        indices_data = {
            'temporal': self.temporal_index,
            'salience': self.salience_index,
            'echo': self.echo_index
        }
        with open(indices_file, 'w') as f:
            json.dump(indices_data, f, indent=2)
            
        # Save analytics
        analytics_file = self.storage_dir / 'analytics.json'
        analytics_data = {
            'node_count': len(self.nodes),
            'edge_count': len(self.edges),
            'edge_types': list(set(e.relation_type for e in self.edges)),
            'memory_types': {mem_type.value: len(nodes) for mem_type, nodes in self.type_index.items() if nodes},
            'avg_salience': sum(n.salience for n in self.nodes.values()) / max(1, len(self.nodes)),
            'avg_echo': sum(n.echo_value for n in self.nodes.values()) / max(1, len(self.nodes)),
            'timestamp': time.time()
        }
        with open(analytics_file, 'w') as f:
            json.dump(analytics_data, f, indent=2)
    
    def load(self):
        """Load the memory graph from disk"""
        # Check if storage directory exists
        if not self.storage_dir.exists():
            logger.info(f"Storage directory {self.storage_dir} does not exist, starting with empty memory")
            return
            
        # Load nodes
        nodes_file = self.storage_dir / 'nodes.json'
        if nodes_file.exists():
            try:
                with open(nodes_file, 'r') as f:
                    nodes_data = json.load(f)
                    
                for node_id, node_data in nodes_data.items():
                    node = MemoryNode.from_dict(node_data)
                    self.nodes[node_id] = node
                    
                    # Rebuild indices
                    self.type_index[node.memory_type].add(node_id)
                    self.source_index[node.source].add(node_id)
                    self.temporal_index.append((node.creation_time, node_id))
                    self.salience_index.append((node.salience, node_id))
                    self.echo_index.append((node.echo_value, node_id))
                    
                    # Add to graph
                    self.graph.add_node(node_id, **{k: v for k, v in node_data.items() 
                                                  if k != 'id' and k != 'embeddings'})
                    
                logger.info(f"Loaded {len(self.nodes)} memory nodes")
            except Exception as e:
                logger.error(f"Error loading nodes: {str(e)}")
                
        # Load edges
        edges_file = self.storage_dir / 'edges.json'
        if edges_file.exists():
            try:
                with open(edges_file, 'r') as f:
                    edges_data = json.load(f)
                    
                for edge_data in edges_data:
                    edge = MemoryEdge.from_dict(edge_data)
                    self.edges.append(edge)
                    
                    # Add to graph if both nodes exist
                    if edge.from_id in self.nodes and edge.to_id in self.nodes:
                        self.graph.add_edge(edge.from_id, edge.to_id, 
                                           relation=edge.relation_type, 
                                           weight=edge.weight,
                                           **edge.metadata)
                    
                logger.info(f"Loaded {len(self.edges)} memory edges")
            except Exception as e:
                logger.error(f"Error loading edges: {str(e)}")
                
        # Sort indices
        self._sort_indices()
    
    def _sort_indices(self):
        """Sort indices for efficient retrieval"""
        self.temporal_index.sort(reverse=True)  # Most recent first
        self.salience_index.sort(reverse=True)  # Highest salience first
        self.echo_index.sort(reverse=True)      # Highest echo value first

    def create_node_from_tree_node(self, tree_node: TreeNode, 
                                  memory_type: Union[MemoryType, str] = MemoryType.SEMANTIC,
                                  source: str = "deep_tree_echo") -> str:
        """Create a memory node from a Deep Tree Echo node
        
        Args:
            tree_node: The TreeNode to convert
            memory_type: Type of memory to create
            source: Source identifier
            
        Returns:
            ID of the created node
        """
        # Convert string memory type to enum if needed
        if isinstance(memory_type, str):
            memory_type = MemoryType(memory_type)
            
        # Create unique ID
        node_id = f"{int(time.time())}_{hash(tree_node.content[:50])}"
        
        # Create memory node
        node = MemoryNode(
            id=node_id,
            content=tree_node.content,
            memory_type=memory_type,
            echo_value=tree_node.echo_value,
            source=source,
            metadata={
                "tree_depth": tree_node.metadata.get("depth", 0),
                "parent_content": tree_node.parent.content[:100] if tree_node.parent else None,
                "children_count": len(tree_node.children)
            }
        )
        
        # Add to memory
        self.add_node(node)
        
        # Add connections to parent and children if they exist in memory
        if tree_node.parent:
            parent_id = f"{int(time.time())}_{hash(tree_node.parent.content[:50])}"
            if parent_id in self.nodes:
                self.add_edge(MemoryEdge(
                    from_id=node_id,
                    to_id=parent_id,
                    relation_type="child_of",
                    weight=0.8
                ))
                
        for i, child in enumerate(tree_node.children):
            child_id = f"{int(time.time())}_{hash(child.content[:50])}"
            if child_id in self.nodes:
                self.add_edge(MemoryEdge(
                    from_id=node_id,
                    to_id=child_id,
                    relation_type="parent_of",
                    weight=0.8 - (i * 0.1)  # Earlier children have stronger connections
                ))
        
        return node_id
    
    def import_tree(self, root_node: TreeNode, memory_type: Union[MemoryType, str] = MemoryType.SEMANTIC,
                   source: str = "deep_tree_echo") -> List[str]:
        """Import an entire Deep Tree Echo tree into the memory system
        
        Args:
            root_node: Root TreeNode to import
            memory_type: Type of memory to create
            source: Source identifier
            
        Returns:
            List of created node IDs
        """
        created_ids = []
        
        def process_node(node, depth=0):
            # Create node ID
            node_id = f"{int(time.time())}_{hash(node.content[:50])}"
            
            # Create memory node
            mem_node = MemoryNode(
                id=node_id,
                content=node.content,
                memory_type=memory_type if isinstance(memory_type, MemoryType) else MemoryType(memory_type),
                echo_value=node.echo_value,
                source=source,
                metadata={
                    "tree_depth": depth,
                    "parent_content": node.parent.content[:100] if node.parent else None,
                    "children_count": len(node.children)
                }
            )
            
            # Add to memory
            self.add_node(mem_node)
            created_ids.append(node_id)
            
            # Process children
            child_ids = []
            for child in node.children:
                child_id = process_node(child, depth + 1)
                child_ids.append(child_id)
                
                # Add parent-child relationship
                self.add_edge(MemoryEdge(
                    from_id=node_id,
                    to_id=child_id,
                    relation_type="parent_of",
                    weight=0.8
                ))
                
                # Add child-parent relationship
                self.add_edge(MemoryEdge(
                    from_id=child_id,
                    to_id=node_id,
                    relation_type="child_of",
                    weight=0.8
                ))
            
            # Add relationships between siblings
            for i in range(len(child_ids)):
                for j in range(i+1, len(child_ids)):
                    # Add bidirectional sibling relationships
                    self.add_edge(MemoryEdge(
                        from_id=child_ids[i],
                        to_id=child_ids[j],
                        relation_type="sibling",
                        weight=0.5
                    ))
                    self.add_edge(MemoryEdge(
                        from_id=child_ids[j],
                        to_id=child_ids[i],
                        relation_type="sibling",
                        weight=0.5
                    ))
            
            return node_id
        
        # Start processing from root
        process_node(root_node)
        
        # Update salience based on graph structure
        self.update_salience_by_centrality()
        
        return created_ids
    
    def export_to_tree(self, root_id: str, max_depth: int = 10) -> Optional[TreeNode]:
        """Export a subgraph as a Deep Tree Echo tree
        
        Args:
            root_id: ID of the root memory node
            max_depth: Maximum depth to export
            
        Returns:
            Root TreeNode or None if root_id not found
        """
        if root_id not in self.nodes:
            return None
            
        # Keep track of processed nodes to avoid cycles
        processed = set()
        
        def build_tree(node_id, depth=0):
            if depth > max_depth or node_id in processed:
                return None
                
            processed.add(node_id)
            node = self.nodes[node_id]
            
            # Create tree node
            tree_node = TreeNode(
                content=node.content,
                echo_value=node.echo_value,
                metadata={
                    "memory_type": node.memory_type.value,
                    "creation_time": node.creation_time,
                    "salience": node.salience,
                    "access_count": node.access_count,
                    "source": node.source,
                    "depth": depth
                }
            )
            
            # Find child nodes (nodes connected with "parent_of" relation)
            child_edges = [(e.to_id, e.weight) for e in self.edges 
                          if e.from_id == node_id and e.relation_type == "parent_of"]
            
            # Sort children by edge weight (descending)
            child_edges.sort(key=lambda x: x[1], reverse=True)
            
            # Process children
            for child_id, _ in child_edges:
                if child_id not in processed:
                    child_node = build_tree(child_id, depth + 1)
                    if child_node:
                        child_node.parent = tree_node
                        tree_node.children.append(child_node)
            
            return tree_node
        
        # Start building from root
        return build_tree(root_id)
    
    def generate_statistics(self) -> Dict:
        """Generate statistics about the memory system
        
        Returns:
            Dictionary of statistics
        """
        if not self.nodes:
            return {
                "node_count": 0,
                "edge_count": 0,
                "memory_type_distribution": {},
                "source_distribution": {},
                "avg_salience": 0,
                "avg_echo_value": 0,
                "avg_connections": 0,
                "recency_distribution": {}
            }
            
        # Calculate statistics
        node_count = len(self.nodes)
        edge_count = len(self.edges)
        
        # Memory type distribution
        memory_type_dist = {mem_type.value: len(nodes) for mem_type, nodes in self.type_index.items() if nodes}
        
        # Source distribution
        source_dist = {source: len(nodes) for source, nodes in self.source_index.items() if nodes}
        
        # Average values
        avg_salience = sum(node.salience for node in self.nodes.values()) / node_count
        avg_echo = sum(node.echo_value for node in self.nodes.values()) / node_count
        
        # Average connections per node
        if node_count > 0:
            avg_connections = edge_count / node_count
        else:
            avg_connections = 0
            
        # Recency distribution (group by day)
        now = time.time()
        day_seconds = 24 * 60 * 60
        recency_dist = {}
        for node in self.nodes.values():
            days_ago = int((now - node.creation_time) / day_seconds)
            recency_dist[days_ago] = recency_dist.get(days_ago, 0) + 1
            
        # Convert to more readable format
        recency_readable = {}
        for days, count in recency_dist.items():
            if days == 0:
                key = "Today"
            elif days == 1:
                key = "Yesterday"
            else:
                key = f"{days} days ago"
            recency_readable[key] = count
        
        return {
            "node_count": node_count,
            "edge_count": edge_count,
            "memory_type_distribution": memory_type_dist,
            "source_distribution": source_dist,
            "avg_salience": avg_salience,
            "avg_echo_value": avg_echo,
            "avg_connections": avg_connections,
            "recency_distribution": recency_readable
        }

# Create default memory instance
memory_system = HypergraphMemory(storage_dir="echo_memory")