# Arquivo: services_backend/tagger_service.py

from .ai_adapter import AI_Adapter
from config import Config 
import json

class TaggerService:
    def __init__(self, ai_adapter: AI_Adapter):
        self.ai_adapter = ai_adapter
        self.tagger_model = Config.MODEL_CONFIG['tagger']
        
        self.system_instruction_template = (
            "You are an expert librarian AI. Your job is to analyze a conversation summary and a list of candidate tags. "
            "Your goal is to create a final, clean list of 3-5 keywords that best describe the summary.\n\n"
            "{master_list_prompt_part}"
            "{session_list_prompt_part}"
            "RULE: Your primary goal is accuracy. If a new, more specific tag is better than reusing an old one, create it.\n\n"
            f"--- GENERAL TAGGING RULES ---\n{Config.TAGGING_RULES}\n\n"
            "Respond ONLY with a valid JSON array of strings."
        )
        print("[Tagger Service]: Serviço de Etiquetagem inicializado.")

    def refine_and_consolidate_tags(self, summary: str, candidate_tags: list, session_tags: list, master_tag_list: list) -> list:
        try:
            master_list_prompt = ""
            if master_tag_list:
                master_list_prompt = (
                    f"--- MASTER VOCABULARY (All Time) ---\n"
                    f"Here is a master list of all tags known to the system: {master_tag_list}\n"
                    f"RULE: Strongly prioritize reusing a tag from this master list to maintain long-term consistency.\n"
                )

            session_list_prompt = ""
            if session_tags:
                session_list_prompt = (
                    f"--- SESSION TAGS (This Session) ---\n"
                    f"Here are preferred tags from the current session: {session_tags}\n"
                    f"RULE: Also prioritize reusing a tag from this list if it's a perfect match.\n"
                )

            system_instruction = self.system_instruction_template.format(
                master_list_prompt_part=master_list_prompt,
                session_list_prompt_part=session_list_prompt
            )

            prompt_text = (
                f"Conversation Summary:\n---\n{summary}\n---\n\n"
                f"Candidate Tags from conversation: {candidate_tags}"
            )

            print("[Tagger Service]: A solicitar refinamento de tags ao AI Adapter...")
            response_json_str = self.ai_adapter.get_completion_sync(
                model_name=self.tagger_model,
                prompt=prompt_text,
                system_instruction=system_instruction,
                json_mode=True
            )
            
            final_tags = json.loads(response_json_str)
            print(f"[Tagger Service]: Tags refinadas recebidas: {final_tags}")
            return final_tags

        except Exception as e:
            print(f"[Tagger Service]: Erro ao refinar tags: {e}. A usar tags candidatas como fallback.")
            return candidate_tags