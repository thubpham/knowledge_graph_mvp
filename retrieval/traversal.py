from core.schema import *
from core.graph import KnowledgeGarden


def direct_lookup(kg: KnowledgeGarden, node: Node, relation: str, direction: str = 'out') -> list:
    if direction == 'out':
        return kg.get_outgoing_edges(node.id, relation=relation, active_only=True)
    else:
        return kg.get_incoming_edges(node.id, relation=relation, active_only=True)


def neighbor_expansion(kg: KnowledgeGarden, node: Node, depth: int = 2) -> list:
    seen_edge_ids = set()
    discovered_edges = []
    visited_nodes = set()
    visited_nodes.add(node.id)
    queue = [(node.id, 0)]
    while queue:
        current_node_id, current_depth = queue.pop(0)
        if current_depth < depth:
            for edge in kg.get_outgoing_edges(current_node_id, active_only=True):
                if edge.id not in seen_edge_ids:
                    seen_edge_ids.add(edge.id)
                    discovered_edges.append(edge)
                if edge.target not in visited_nodes:
                    visited_nodes.add(edge.target)
                    queue.append((edge.target, current_depth + 1))
            for edge in kg.get_incoming_edges(current_node_id, active_only=True):
                if edge.id not in seen_edge_ids:
                    seen_edge_ids.add(edge.id)
                    discovered_edges.append(edge)
                if edge.source not in visited_nodes:
                    visited_nodes.add(edge.source)
                    queue.append((edge.source, current_depth + 1))
    return discovered_edges


def path_finding(kg: KnowledgeGarden, source: Node, target: Node) -> list:
    return kg.get_shortest_path_edges(source.id, target.id)


def impact_traversal(kg: KnowledgeGarden, node: Node, relation: str, direction: str = 'out') -> list:
    seen_edge_ids = set()
    discovered_edges = []
    visited_nodes = set()

    def dfs(current_node_id):
        visited_nodes.add(current_node_id)
        if direction == 'out':
            edges = kg.get_outgoing_edges(current_node_id, relation=relation, active_only=True)
        else:
            edges = kg.get_incoming_edges(current_node_id, relation=relation, active_only=True)
        for edge in edges:
            if edge.id not in seen_edge_ids:
                seen_edge_ids.add(edge.id)
                discovered_edges.append(edge)
            next_id = edge.target if direction == 'out' else edge.source
            if next_id not in visited_nodes:
                dfs(next_id)

    dfs(node.id)
    return discovered_edges


def history_traversal(kg: KnowledgeGarden, node: Node, relation: str, direction: str = 'out') -> list:
    if direction == 'out':
        edges = kg.get_outgoing_edges(node.id, relation=relation, active_only=False)
    else:
        edges = kg.get_incoming_edges(node.id, relation=relation, active_only=False)
    return sorted(edges, key=lambda e: e.valid_from)
