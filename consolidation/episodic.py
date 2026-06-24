from core.graph import KnowledgeGarden


def get_episode_for_entity(entity_id: str, kg: KnowledgeGarden):
    return kg.get_episodes_for_entity(entity_id)
