from .similarity_util import calculate_cosine_similarity
import numpy as np
from config import Config

class Contextualizador:
    def __init__(self, embedding_model):
        self.embedding_model = embedding_model
        self.threshold = Config.CONTEXTUALIZER_SIMILARITY_THRESHOLD
        self.current_topic_vector = None
        self.current_topic_tags = set()
        self.current_conversation_block = []

    def _get_embedding_for_tags(self, tags: list):
        if not tags:
            return None
        tag_string = " ".join(tags)
        return self.embedding_model.encode([tag_string])

    def add_message_and_check_topic(self, new_message: dict, new_tags: list) -> dict | None:
        block_to_process = None
        
        if new_message['role'] == 'user':
            self.current_conversation_block.append(new_message)
        
        new_tags_set = set(new_tags)
        new_vector = self._get_embedding_for_tags(list(new_tags_set))

        if new_vector is None:
            return None

        if self.current_topic_vector is None:
            self.current_topic_vector = new_vector
            self.current_topic_tags = new_tags_set
            return None

        similarity = calculate_cosine_similarity(self.current_topic_vector, new_vector)
        print(f"[Contextualizador]: Similaridade Semântica calculada: {similarity:.2f}")

        if similarity >= self.threshold:
            self.current_topic_vector = (self.current_topic_vector * len(self.current_conversation_block) + new_vector) / (len(self.current_conversation_block) + 1)
            self.current_topic_tags.update(new_tags_set)
        else:
            if len(self.current_conversation_block) > 1:
                block_to_process = {
                    "block": self.current_conversation_block,
                    "tags": list(self.current_topic_tags)
                }
            
            self.current_conversation_block = [new_message] if new_message['role'] == 'user' else []
            self.current_topic_tags = new_tags_set
            self.current_topic_vector = new_vector

        return block_to_process
    
    def add_model_response_to_block(self, model_response: dict):
        self.current_conversation_block.append(model_response)