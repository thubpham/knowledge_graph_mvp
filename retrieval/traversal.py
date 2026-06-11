from core.schema import *
from core.graph import KnowledgeGarden
import networkx as nx

def direct_lookup(kg: KnowledgeGarden, node: Node, 
                  relation: str, direction: str = 'out') -> list[Edge]:
    output = []
    if direction == 'out':
        for edge in kg.edges.values():
            if edge.source == node.id and edge.relation == relation and edge.valid_until is None:
                output.append(edge)
    else: 
        for edge in kg.edges.values():
            if edge.target == node.id and edge.relation == relation and edge.valid_until is None:
                output.append(edge)
    return output

def neighbor_expansion(kg: KnowledgeGarden, node: Node, depth: int = 2) -> list[Edge]:
    discovered_edges = set()
    visited_nodes = set()
    visited_nodes.add(node.id)
    queue = [(node.id, 0)] 
    while queue:
        current_node_id, current_depth = queue.pop(0)
        if current_depth < depth:
            for edge in kg.edges.values():
                if edge.source == current_node_id and edge.valid_until is None:
                    discovered_edges.add(edge)
                    if edge.target not in visited_nodes:
                        visited_nodes.add(edge.target)
                        queue.append((edge.target, current_depth + 1))
                elif edge.target == current_node_id and edge.valid_until is None:
                    discovered_edges.add(edge)
                    if edge.source not in visited_nodes:
                        visited_nodes.add(edge.source)
                        queue.append((edge.source, current_depth + 1))
    return list(discovered_edges)

def path_finding(kg: KnowledgeGarden, source: Node, target: Node) -> list[Edge]:
    discovered_nodes = nx.shortest_path(kg.graph.to_undirected(), source.id, target.id)
    discovered_edges = []
    for i in range(1, len(discovered_nodes)):
        for edge in kg.edges.values():
            if (edge.source == discovered_nodes[i-1] and edge.target == discovered_nodes[i] and edge.valid_until is None) or (edge.source == discovered_nodes[i] and edge.target == discovered_nodes[i-1] and edge.valid_until is None):
                discovered_edges.append(edge)
    return discovered_edges

def impact_traversal(kg: KnowledgeGarden, node: Node, relation: str, direction: str = 'out'):
    discovered_edges = set()
    visited_nodes = set()
    def dfs(current_node_id):
        visited_nodes.add(current_node_id)
        for edge in kg.edges.values():
            if direction == 'out' and edge.source == current_node_id and edge.relation == relation and edge.valid_until is None:
                discovered_edges.add(edge)
                if edge.target not in visited_nodes:
                    dfs(edge.target)
            elif direction == 'in' and edge.target == current_node_id and edge.relation == relation and edge.valid_until is None:
                discovered_edges.add(edge)
                if edge.source not in visited_nodes:
                    dfs(edge.source)
    dfs(node.id)
    return list(discovered_edges)

def history_traversal(kg: KnowledgeGarden, node: Node, relation: str, direction: str = 'out') -> list[Edge]:
    output = []
    if direction == 'out':
        for edge in kg.edges.values():
            if edge.source == node.id and edge.relation == relation:
                output.append(edge)
    else: 
        for edge in kg.edges.values():
            if edge.target == node.id and edge.relation == relation:
                output.append(edge)
    output.sort(key=lambda e: e.valid_from)
    return output