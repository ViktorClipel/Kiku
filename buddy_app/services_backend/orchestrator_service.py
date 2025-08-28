# Arquivo: services_backend/orchestrator_service.py

import threading
import os
# --- CORREÇÕES DE IMPORTAÇÃO ---
from models import ActionPlan
from .memory_service import MemoryService
from .prompt_builder import PromptBuilder
from .ai_adapter import AI_Adapter
from .summarizer_service import SummarizerService
from .tagger_service import TaggerService
from .segmenter_service import SegmenterService
from .utils.contextualizador import Contextualizador
from .utils.similarity_util import calculate_jaccard_similarity
from .utils.model_resolver import build_available_model_rankings
from config import Config
# --- FIM DAS CORREÇÕES ---
import json
from dotenv import set_key, get_key


class OrchestratorService:
    WORKBENCH_SIMILARITY_THRESHOLD = 0.1

    def __init__(self, memory_service: MemoryService, ai_adapter: AI_Adapter, embedding_model):
        print(f"[Orchestrator Service para Usuário {memory_service.user_data_path}]: A inicializar...")
        
        self.memory_service = memory_service
        self.ai_adapter = ai_adapter
        
        self.prompt_builder = PromptBuilder(self.memory_service)
        self.contextualizador = Contextualizador(embedding_model)
        
        self.is_model_initialized = True
        
        self.MODEL_CASCADES = { "DEFAULT": ["gemini-1.5-flash-latest"] }
        self.locked_models = set()
        self.prompt_counter = 0

        print(f"[Orchestrator Service para Usuário {memory_service.user_data_path}]: Serviço inicializado.")

    def add_to_history(self, message: dict):
        self.memory_service.add_to_history(message)

    def get_full_history(self) -> list:
        return self.memory_service.get_short_term_memory()

    def initialize_model(self):
        print("[Orchestrator Service]: A inicialização é tratada pelo AI_Adapter.")
        pass

    def get_api_key(self, key_name: str) -> str | None:
        try:
            dotenv_path = os.path.join(os.path.dirname(__file__), '..', '.env')
            if not os.path.exists(dotenv_path):
                open(dotenv_path, 'a').close()
            return get_key(dotenv_path, key_name)
        except Exception as e:
            print(f"Erro ao ler a chave {key_name}: {e}")
            return None

    def save_api_key_and_rebuild(self, key_name: str, key_value: str):
        if not key_name:
            return
        try:
            print(f"A guardar a chave {key_name} e a reconstruir os modelos...")
            dotenv_path = os.path.join(os.path.dirname(__file__), '..', '.env')
            
            set_key(dotenv_path, key_name, key_value or "")
            
            # Atualiza a configuração global do Flask App
            Config.DYNAMIC_MODEL_RANKINGS = build_available_model_rankings()

            self.ai_adapter._configure_apis()
            
            print("Modelos reconstruídos com sucesso.")
        except Exception as e:
            print(f"Erro ao guardar a chave e reconstruir modelos: {e}")

    def _get_action_plan(self, conversation_history: list) -> ActionPlan:
        try:
            if not conversation_history:
                return ActionPlan()

            recent_history = conversation_history[-4:]
            context_prompt = "\n".join(
                f"{msg['role']}: {msg['parts'][0]}" for msg in recent_history
            )

            classifier_model = Config.MODEL_CONFIG['classifier']
            available_specialties = list(Config.DYNAMIC_MODEL_RANKINGS.keys())

            classifier_system_prompt = (
                "You are an expert task analyzer. Your goal is to analyze the user's request and respond with a single JSON object. "
                "You must classify the request into one of the following specialties: "
                f"{available_specialties}. Use 'conversation' for simple questions, greetings, or conversational chat. "
                "The JSON object must have the following keys: 'specialty' (string), "
                "'needs_search' (boolean), 'needs_long_term_memory' (boolean), "
                "'tags' (a JSON array of 1-3 relevant keyword tags), and 'extracted_facts' (a JSON object or null)."
            )

            response_json_str = self.ai_adapter.get_completion_sync(
                model_name=classifier_model,
                prompt=context_prompt,
                system_instruction=classifier_system_prompt,
                json_mode=True
            )
            plan_data = json.loads(response_json_str)
            return ActionPlan(**plan_data)

        except Exception as e:
            print(f"[Orchestrator]: Erro ao classificar plano de ação: {e}. Usando fallback.")
            return ActionPlan()

    def _resolve_model_cascade(self, action_plan: ActionPlan) -> list:
        specialty = action_plan.specialty
        if not specialty or specialty not in Config.DYNAMIC_MODEL_RANKINGS:
            print(f"[Orchestrator]: Especialidade '{specialty}' inválida ou sem modelo disponível. Usando cascata DEFAULT.")
            return self.MODEL_CASCADES["DEFAULT"]

        rankings = Config.DYNAMIC_MODEL_RANKINGS[specialty]
        sorted_models = sorted(rankings, key=rankings.get, reverse=True)
        
        print(f"[Orchestrator]: Cascata de modelos resolvida para especialidade '{specialty}': {sorted_models}")
        return sorted_models

    def generate_response_stream(self):
        self.prompt_counter += 1
        
        try:
            system_instruction, conversation_history = self.prompt_builder.build_context()
            if not conversation_history:
                yield "Histórico de conversa vazio. Por favor, envie uma mensagem."
                yield "[STREAM_END]"
                return

            action_plan = self._get_action_plan(conversation_history)
            
            print(f"[Orchestrator]: Plano de Ação -> Especialidade: {action_plan.specialty}, Pesquisa Web: {action_plan.needs_search}, Memória Longo Prazo: {action_plan.needs_long_term_memory}, Tags: {action_plan.tags}")
            
            if action_plan.extracted_facts:
                self.memory_service.add_fact(action_plan.extracted_facts)
                print(f"[Orchestrator]: Facto extraído pelo classificador e salvo: {action_plan.extracted_facts}")

            last_user_message = conversation_history[-1]
            closed_block = self.contextualizador.add_message_and_check_topic(last_user_message, action_plan.tags)
            
            if closed_block:
                print(f"[Orchestrator]: Contextualizador detectou fim de tópico.")
                self.memory_service.add_block_to_workbench(closed_block)
                
                print("[Orchestrator]: Agendando arquivamento em uma nova thread...")
                archive_thread = threading.Thread(
                    target=self.memory_service.process_conversation_block_for_archiving,
                    args=(closed_block,)
                )
                archive_thread.start()

            if action_plan.tags:
                self.memory_service.add_predictive_tags(action_plan.tags)

            workbench_context = self._consult_workbench(action_plan.tags)
            if workbench_context:
                conversation_history.append({"role": "user", "parts": [workbench_context]})

            if action_plan.needs_long_term_memory:
                print("[Orchestrator]: A aceder à memória de longo prazo (RAG)...")
                retrieved_memories = self.memory_service.retrieve_relevant_memories(conversation_history[-1]['parts'][0])
                if retrieved_memories:
                    conversation_history.append({"role": "user", "parts": [f"<RECALLED_MEMORIES>\n{retrieved_memories}\n</RECALLED_MEMORIES>"]})
            
            cascade = self._resolve_model_cascade(action_plan)
            
            full_response = ""
            for chunk in self._execute_generation_cascade(cascade, system_instruction, conversation_history):
                if chunk != "[STREAM_END]":
                    full_response += chunk
                yield chunk

            if full_response:
                model_response_message = {"role": "model", "parts": [full_response]}
                self.contextualizador.add_model_response_to_block(model_response_message)

        except Exception as e:
            error_message = f"Ocorreu um erro fatal no Orquestrador: {e}"
            print(error_message)
            yield error_message
            yield "[STREAM_END]"

    def _consult_workbench(self, current_prompt_tags: list) -> str:
        workbench = self.memory_service.workbench
        if not workbench or not current_prompt_tags:
            return ""

        print("[Orchestrator]: Consultando a Bancada de Trabalho (Nível 2)...")
        
        best_match_block = None
        highest_similarity = 0.0
        current_tags_set = set(current_prompt_tags)

        for block in workbench:
            block_tags_set = set(block.get("tags", []))
            similarity = calculate_jaccard_similarity(current_tags_set, block_tags_set)

            if similarity > highest_similarity:
                highest_similarity = similarity
                best_match_block = block
        
        if best_match_block and highest_similarity > self.WORKBENCH_SIMILARITY_THRESHOLD:
            print(f"[Orchestrator]: Bloco relevante encontrado na Bancada com similaridade de {highest_similarity:.2f}.")
            
            context_header = f"<CONTEXTO_DA_SESSAO (Tópicos: {', '.join(best_match_block.get('tags', []))})>\n"
            conversation_text = "\n".join(
                f"{msg['role']}: {msg['parts'][0]}" for msg in best_match_block.get("block", [])
            )
            context_footer = "\n</CONTEXTO_DA_SESSAO>"
            
            return context_header + conversation_text + context_footer
        
        print("[Orchestrator]: Nenhum bloco relevante encontrado na Bancada.")
        return ""

    def _execute_generation_cascade(self, cascade, system_instruction, conversation_history):
        if self.prompt_counter > 1 and self.prompt_counter % 10 == 0 and self.locked_models:
            print(f"[Orchestrator]: Resetando travas de fallback após {self.prompt_counter} prompts.")
            self.locked_models.clear()

        last_error = None
        loop_count = 0
        while loop_count < 3:
            for model_name in cascade:
                if model_name in self.locked_models:
                    print(f"[Orchestrator]: A saltar modelo travado: {model_name}")
                    continue
                try:
                    print(f"[Orchestrator]: A tentar com o modelo: {model_name}")
                    yield from self.ai_adapter.get_completion_stream(
                        model_name=model_name,
                        conversation_history=conversation_history,
                        system_instruction=system_instruction
                    )
                    return 
                except Exception as e:
                    last_error = e
                    print(f"[Orchestrator]: Falha com o modelo {model_name}. A travar o modelo. Erro: {e}")
                    self.locked_models.add(model_name)
            
            loop_count += 1
            if loop_count < 3:
                print(f"[Orchestrator]: Todos os modelos disponíveis falharam. A repetir loop ({loop_count}/3)...")
        
        yield f"Erro: Todos os modelos falharam após 3 tentativas. Último erro: {last_error}"
        yield "[STREAM_END]"
        
    def validate_and_save_api_key(self, api_key: str) -> dict:
        provider_key_name = self.ai_adapter.identify_and_validate_key(api_key)

        if provider_key_name:
            self.save_api_key_and_rebuild(provider_key_name, api_key)
            provider_name = provider_key_name.replace("_API_KEY", "").capitalize()
            return {
                "success": True,
                "message": f"Chave da {provider_name} validada e guardada com sucesso.",
                "providers": self.get_active_providers()
            }
        else:
            return {
                "success": False,
                "message": "A chave de API fornecida é inválida ou não foi reconhecida.",
                "providers": self.get_active_providers()
            }

    def delete_api_key(self, provider_key_name: str) -> dict:
        print(f"A remover a chave para {provider_key_name}...")
        self.save_api_key_and_rebuild(provider_key_name, "")
        provider_name = provider_key_name.replace("_API_KEY", "").capitalize()
        return {
            "success": True,
            "message": f"Chave da {provider_name} removida com sucesso.",
            "providers": self.get_active_providers()
        }

    def get_active_providers(self) -> list:
        active_providers = []
        known_providers = {"Gemini": "GEMINI_API_KEY", "Openai": "OPENAI_API_KEY"}
        
        for name, key_env in known_providers.items():
            if self.get_api_key(key_env):
                active_providers.append({"name": name, "keyName": key_env})
        
        return active_providers