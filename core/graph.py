import uuid
import networkx as nx
from .schema import *

class KnowledgeGarden:
    
    def __init__(self):
        self.graph = nx.MultiDiGraph()
        self.edges = {}
        self.nodes = {}

    def add_node(self, id: str, type: str, name: str):
        if id in self.nodes:
            raise ValueError(f"Node with id {id} already exists.")
        new_node = Node(id, type, name)
        self.graph.add_node(new_node)
        self.nodes[id] = new_node
        

    def add_edge(self, source: str, target: str, relation: str, fact:str, valid_from: datetime):
        edges = self.graph.get_edge_data(source, target)
        if edges is not None:
            for edge_attr in edges.values():
                if edge_attr['relation'] == relation and edge_attr['valid_until'] is None:
                    raise ValueError(f"Edge ({source})-({relation})->({target}) already exists. Update the edge or replace with new edge.")
        edge_id = uuid.uuid4()
        self.graph.add_edge(source, target,  key = str(edge_id), relation = relation, valid_until = None)
        new_edge = Edge(str(edge_id), source, target, relation, fact, valid_from)
        self.edges[str(edge_id)] = new_edge
        return str(edge_id)
    
    def get_current_facts(self, source: str):
        if source not in self.nodes:
            raise ValueError(f"Node with id {source} does not exist.")
        current_facts = []
        for edge in self.edges.values():
            if edge.source == source and edge.valid_until is None:
                current_facts.append(edge)
        return current_facts
    
    def get_facts_at(self, source: str, timestamp: datetime):
        if source not in self.nodes:
            raise ValueError(f"Node with id {source} does not exist.")
        facts_at_time = []
        for edge in self.edges.values():
            if edge.source == source and edge.valid_from <= timestamp and (edge.valid_until is None or edge.valid_until > timestamp):
                facts_at_time.append(edge)
        return facts_at_time

    def invalidate_edge(self, edge_id: str, timestamp: datetime):
        if edge_id not in self.edges:
            raise ValueError(f"Edge with id {edge_id} does not exist.")
        edge = self.edges[edge_id]
        edge.valid_until = timestamp
        self.graph[edge.source][edge.target][edge_id]['valid_until'] = timestamp
        


        


        
