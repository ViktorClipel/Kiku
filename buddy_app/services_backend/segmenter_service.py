# Arquivo: services_backend/segmenter_service.py

from .ai_adapter import AI_Adapter
from config import Config
import json

class SegmenterService:
    def __init__(self, ai_adapter: AI_Adapter):
        self.ai_adapter = ai_adapter
        self.segmenter_model = Config.MODEL_CONFIG['segmenter']
        self.system_instruction = (
            "You are a conversation analysis expert. Your task is to read a conversation transcript "
            "and segment it into distinct topics. Each topic should be a self-contained block of messages. "
            "Respond ONLY with a valid JSON object. The JSON object should have keys representing the topic "
            "(e.g., 'topic_1', 'topic_2') and the value for each key should be an array of the message objects "
            "that belong to that topic."
        )
        print("[Segmenter Service]: Serviço de Segmentação inicializado.")

    def segment_conversation_by_topic(self, conversation_chunk: list) -> dict:
        try:
            prompt_text = json.dumps(conversation_chunk, indent=2, ensure_ascii=False)

            print("[Segmenter Service]: A solicitar segmentação de tópicos ao AI Adapter...")
            response_json_str = self.ai_adapter.get_completion_sync(
                model_name=self.segmenter_model,
                prompt=prompt_text,
                system_instruction=self.system_instruction,
                json_mode=True
            )
            
            segmented_topics = json.loads(response_json_str)
            print(f"[Segmente Service]: Conversa segmentada em {len(segmented_topics)} tópicos.")
            return segmented_topics

        except Exception as e:
            print(f"[Segmenter Service]: Erro ao segmentar conversa: {e}. A tratar o bloco inteiro como um único tópico.")
            return {"topic_1": conversation_chunk}