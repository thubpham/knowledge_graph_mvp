from core.graph import KnowledgeGarden

def get_episode_for_entity(entity_id: str, kg: KnowledgeGarden):
    episodes = []
    for edge in kg.edges.values():
        if edge.relation == "MENTIONED_IN" and edge.source == entity_id:
            episode_id = edge.target
            episode = kg.episodes.get(episode_id)
            if episode:
                episodes.append(episode)
    episodes.sort(key = lambda e: e.reference_time)
    return episodes