import uuid
import os
from datetime import datetime
from dotenv import load_dotenv
from falkordb import FalkorDB
from .schema import Node, Edge, Episode

load_dotenv()


def _fmt_dt(dt: datetime) -> str | None:
    return dt.isoformat() if dt else None


def _parse_dt(s) -> datetime | None:
    return datetime.fromisoformat(s) if s else None


class KnowledgeGarden:

    def __init__(self):
        host = os.getenv('FALKORDB_HOST', 'localhost')
        port = int(os.getenv('FALKORDB_PORT', 6379))
        self._db = FalkorDB(host=host, port=port)
        self._graph = self._db.select_graph('knowledge_garden')
        self._create_indexes()

    def _create_indexes(self):
        indexes = [
            "CREATE INDEX FOR (n:Entity) ON (n.id)",
            "CREATE INDEX FOR (n:Episode) ON (n.id)",
            "CREATE INDEX FOR ()-[e:EDGE]-() ON (e.relation)",
            "CREATE INDEX FOR ()-[e:EDGE]-() ON (e.valid_until)",
            "CREATE INDEX FOR ()-[e:EDGE]-() ON (e.valid_from)",
        ]
        for idx in indexes:
            try:
                self._graph.query(idx)
            except Exception:
                pass

    # ── Deserializers ────────────────────────────────────────────────────────

    def _node_from_props(self, props: dict) -> Node:
        node = Node(props['id'], props['type'], props['name'])
        node.summary = props.get('summary')
        node.consolidated = props.get('consolidated', False)
        node.consolidation_run_id = props.get('consolidation_run_id')
        node.created_at = _parse_dt(props.get('created_at'))
        return node

    def _edge_from_props(self, props: dict) -> Edge:
        edge = Edge(
            props['id'],
            props['source'],
            props['target'],
            props['relation'],
            props['fact'],
            _parse_dt(props['valid_from']),
        )
        edge.valid_until = _parse_dt(props.get('valid_until'))
        edge.confidence = props.get('confidence', 1.0)
        edge.source_type = props.get('source_type')
        edge.source_id = props.get('source_id')
        edge.ingested_at = _parse_dt(props.get('ingested_at'))
        edge.extracted_by = props.get('extracted_by')
        return edge

    def _episode_from_props(self, props: dict) -> Episode:
        import json as _json
        ep = object.__new__(Episode)
        ep.id = props['id']
        ep.text = props['text']
        ep.source_type = props.get('source_type')
        ep.source_id = props.get('source_id')
        ep.reference_time = _parse_dt(props.get('reference_time'))
        ep.ingested_at = _parse_dt(props.get('ingested_at'))
        raw_meta = props.get('metadata')
        ep.metadata = _json.loads(raw_meta) if raw_meta else None
        return ep

    # ── Write methods ────────────────────────────────────────────────────────

    def add_node(self, id: str, type: str, name: str):
        if self.node_exists(id):
            raise ValueError(f"Node with id {id} already exists.")
        self._graph.query(
            "CREATE (:Entity {id: $id, type: $type, name: $name, "
            "summary: null, consolidated: false, consolidation_run_id: null, "
            "created_at: $created_at})",
            {'id': id, 'type': type, 'name': name, 'created_at': _fmt_dt(datetime.now())}
        )

    def add_edge(self, source: str, target: str, relation: str, fact: str, valid_from: datetime) -> str:
        result = self._graph.query(
            "MATCH (a {id: $source})-[e:EDGE]->(b {id: $target}) "
            "WHERE e.relation = $relation AND e.valid_until IS NULL RETURN e",
            {'source': source, 'target': target, 'relation': relation}
        )
        if result.result_set:
            raise ValueError(
                f"Edge ({source})-({relation})->({target}) already exists. "
                "Update the edge or replace with new edge."
            )
        edge_id = str(uuid.uuid4())
        self._graph.query(
            "MATCH (a {id: $source}), (b {id: $target}) "
            "CREATE (a)-[:EDGE {id: $id, source: $source, target: $target, "
            "relation: $relation, fact: $fact, valid_from: $valid_from, "
            "valid_until: null, confidence: 1.0, source_type: null, source_id: null, "
            "ingested_at: $ingested_at, extracted_by: null}]->(b)",
            {
                'source': source, 'target': target, 'id': edge_id,
                'relation': relation, 'fact': fact,
                'valid_from': _fmt_dt(valid_from),
                'ingested_at': _fmt_dt(datetime.now()),
            }
        )
        return edge_id

    def add_episode(self, episode: Episode) -> str:
        import json as _json
        self._graph.query(
            "CREATE (:Episode {id: $id, text: $text, source_type: $source_type, "
            "source_id: $source_id, reference_time: $reference_time, ingested_at: $ingested_at, "
            "metadata: $metadata})",
            {
                'id': episode.id,
                'text': episode.text,
                'source_type': episode.source_type,
                'source_id': episode.source_id,
                'reference_time': _fmt_dt(episode.reference_time),
                'ingested_at': _fmt_dt(episode.ingested_at),
                'metadata': _json.dumps(episode.metadata) if episode.metadata else None,
            }
        )
        return episode.id

    def invalidate_edge(self, edge_id: str, timestamp: datetime):
        result = self._graph.query(
            "MATCH ()-[e:EDGE {id: $id}]->() RETURN e", {'id': edge_id}
        )
        if not result.result_set:
            raise ValueError(f"Edge with id {edge_id} does not exist.")
        self._graph.query(
            "MATCH ()-[e:EDGE {id: $id}]->() SET e.valid_until = $ts",
            {'id': edge_id, 'ts': _fmt_dt(timestamp)}
        )

    def update_node(self, id: str, **kwargs):
        set_clauses = ", ".join(f"n.{k} = ${k}" for k in kwargs)
        self._graph.query(
            f"MATCH (n:Entity {{id: $id}}) SET {set_clauses}",
            {'id': id, **kwargs}
        )

    # ── Read methods ─────────────────────────────────────────────────────────

    def node_exists(self, id: str) -> bool:
        result = self._graph.query(
            "MATCH (n {id: $id}) RETURN n LIMIT 1", {'id': id}
        )
        return bool(result.result_set)

    def get_node(self, id: str) -> Node:
        result = self._graph.query(
            "MATCH (n:Entity {id: $id}) RETURN n", {'id': id}
        )
        if not result.result_set:
            return None
        return self._node_from_props(result.result_set[0][0].properties)

    def get_all_nodes(self) -> list:
        result = self._graph.query("MATCH (n:Entity) RETURN n")
        return [self._node_from_props(r[0].properties) for r in result.result_set]

    def get_episode(self, id: str) -> Episode:
        result = self._graph.query(
            "MATCH (ep:Episode {id: $id}) RETURN ep", {'id': id}
        )
        if not result.result_set:
            return None
        return self._episode_from_props(result.result_set[0][0].properties)

    def get_all_edges(self) -> list:
        result = self._graph.query("MATCH ()-[e:EDGE]->() RETURN e")
        return [self._edge_from_props(r[0].properties) for r in result.result_set]

    def get_outgoing_edges(self, node_id: str, relation: str = None, active_only: bool = True) -> list:
        params = {'node_id': node_id}
        where_parts = []
        if relation:
            where_parts.append("e.relation = $relation")
            params['relation'] = relation
        if active_only:
            where_parts.append("e.valid_until IS NULL")
        where = f" WHERE {' AND '.join(where_parts)}" if where_parts else ""
        result = self._graph.query(
            f"MATCH ({{id: $node_id}})-[e:EDGE]->(){ where} RETURN e", params
        )
        return [self._edge_from_props(r[0].properties) for r in result.result_set]

    def get_incoming_edges(self, node_id: str, relation: str = None, active_only: bool = True) -> list:
        params = {'node_id': node_id}
        where_parts = []
        if relation:
            where_parts.append("e.relation = $relation")
            params['relation'] = relation
        if active_only:
            where_parts.append("e.valid_until IS NULL")
        where = f" WHERE {' AND '.join(where_parts)}" if where_parts else ""
        result = self._graph.query(
            f"MATCH ()-[e:EDGE]->({{id: $node_id}}){where} RETURN e", params
        )
        return [self._edge_from_props(r[0].properties) for r in result.result_set]

    def get_current_facts(self, source: str) -> list:
        if not self.node_exists(source):
            raise ValueError(f"Node with id {source} does not exist.")
        return self.get_outgoing_edges(source, active_only=True)

    def get_facts_at(self, source: str, timestamp: datetime) -> list:
        if not self.node_exists(source):
            raise ValueError(f"Node with id {source} does not exist.")
        ts = _fmt_dt(timestamp)
        result = self._graph.query(
            "MATCH ({id: $source})-[e:EDGE]->() "
            "WHERE e.valid_from <= $ts AND (e.valid_until IS NULL OR e.valid_until > $ts) RETURN e",
            {'source': source, 'ts': ts}
        )
        return [self._edge_from_props(r[0].properties) for r in result.result_set]

    def get_episodes_for_entity(self, entity_id: str) -> list:
        result = self._graph.query(
            "MATCH ({id: $entity_id})-[e:EDGE {relation: 'MENTIONED_IN'}]->(ep:Episode) "
            "RETURN ep ORDER BY ep.reference_time",
            {'entity_id': entity_id}
        )
        return [self._episode_from_props(r[0].properties) for r in result.result_set]

    def get_episode_by_source(self, source_id: str) -> Episode | None:
        result = self._graph.query(
            "MATCH (ep:Episode {source_id: $source_id}) RETURN ep LIMIT 1",
            {'source_id': source_id}
        )
        if not result.result_set:
            return None
        return self._episode_from_props(result.result_set[0][0].properties)

    def update_episode(self, episode_id: str, **kwargs):
        import json as _json
        params = {'id': episode_id}
        set_parts = []
        for k, v in kwargs.items():
            params[k] = _json.dumps(v) if isinstance(v, dict) else v
            set_parts.append(f"ep.{k} = ${k}")
        self._graph.query(
            f"MATCH (ep:Episode {{id: $id}}) SET {', '.join(set_parts)}",
            params
        )

    def get_shortest_path_edges(self, source_id: str, target_id: str) -> list:
        result = self._graph.query(
            "MATCH p = shortestPath((a {id: $source})-[*]-(b {id: $target})) "
            "RETURN [n IN nodes(p) | n.id]",
            {'source': source_id, 'target': target_id}
        )
        if not result.result_set:
            return []
        path_ids = result.result_set[0][0]
        edges = []
        for i in range(1, len(path_ids)):
            a, b = path_ids[i - 1], path_ids[i]
            result2 = self._graph.query(
                "MATCH (a {id: $a})-[e:EDGE]->(b {id: $b}) WHERE e.valid_until IS NULL RETURN e "
                "UNION "
                "MATCH (a {id: $b})-[e:EDGE]->(b {id: $a}) WHERE e.valid_until IS NULL RETURN e",
                {'a': a, 'b': b}
            )
            for row in result2.result_set:
                edges.append(self._edge_from_props(row[0].properties))
        return edges
