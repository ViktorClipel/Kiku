# Arquivo: services_backend/summarizer_service.py

from .ai_adapter import AI_Adapter
from config import Config

class SummarizerService:
    def __init__(self, ai_adapter: AI_Adapter):
        self.ai_adapter = ai_adapter
        self.summarizer_model = Config.MODEL_CONFIG['summarizer']
        self.system_instruction = (
            "You are a memory summarization expert. Your task is to read a conversation "
            "and create a concise, third-person summary of the key information. "
            "Focus on facts, user preferences, decisions made, and main topics discussed. "
            "Ignore greetings and pleasantries. The summary should be dense with information."
        )
        print("[Summarizer Service]: Serviço de Sumarização inicializado.")

    def summarize_conversation_chunk(self, conversation_chunk: list) -> str:
        try:
            prompt_text = "\n".join(
                f"{msg['role']}: {msg['parts'][0]}" for msg in conversation_chunk
            )

            print("[Summarizer Service]: A solicitar resumo ao AI Adapter...")
            summary = self.ai_adapter.get_completion_sync(
                model_name=self.summarizer_model,
                prompt=prompt_text,
                system_instruction=self.system_instruction
            )
            print("[Summarizer Service]: Resumo recebido com sucesso.")
            return summary

        except Exception as e:
            print(f"[Summarizer Service]: Erro ao gerar resumo: {e}")
            return None