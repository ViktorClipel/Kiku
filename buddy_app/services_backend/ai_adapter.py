# Arquivo: buddy_app/backend/services/ai_adapter.py (Completo e Atualizado)

import google.generativeai as genai
import openai
import os
from dotenv import load_dotenv

class AI_Adapter:
    def __init__(self):
        self._configure_apis()
        print("[AI Adapter]: Adaptador de IA inicializado e pronto.")

    def _configure_apis(self):
        try:
            dotenv_path = os.path.join(os.path.dirname(__file__), '..', '.env')
            load_dotenv(dotenv_path=dotenv_path)
            
            gemini_api_key = os.getenv("GEMINI_API_KEY")
            if gemini_api_key:
                genai.configure(api_key=gemini_api_key)
            
            if not os.getenv("OPENAI_API_KEY") and not gemini_api_key:
                 print("[AI Adapter]: Nenhuma chave de API (GEMINI_API_KEY ou OPENAI_API_KEY) foi encontrada.")

        except Exception as e:
            print(f"[AI Adapter]: Falha fatal ao configurar as APIs: {e}")
            raise

    def identify_and_validate_key(self, api_key: str) -> str | None:
        """
        Tenta identificar a qual provedor uma chave de API pertence, fazendo
        uma chamada de teste leve.
        Retorna o nome da variável de ambiente (ex: "GEMINI_API_KEY") ou None.
        """
        try:
            print("[AI Adapter]: Tentando validar a chave como Google Gemini...")
            genai.configure(api_key=api_key)
            genai.get_model('models/gemini-1.5-flash-latest')
            print("[AI Adapter]: Chave validada com sucesso como GEMINI_API_KEY.")
            return "GEMINI_API_KEY"
        except Exception as e:
            print(f"[AI Adapter]: Validação como Gemini falhou. Erro: {e}")
            pass

        try:
            print("[AI Adapter]: Tentando validar a chave como OpenAI...")
            temp_openai_client = openai.OpenAI(api_key=api_key)
            temp_openai_client.models.list()
            print("[AI Adapter]: Chave validada com sucesso como OPENAI_API_KEY.")
            return "OPENAI_API_KEY"
        except Exception as e:
            print(f"[AI Adapter]: Validação como OpenAI falhou. Erro: {e}")
            pass
            
        print("[AI Adapter]: A chave não foi reconhecida por nenhum provedor.")
        return None

    def get_completion_stream(self, model_name: str, conversation_history: list, system_instruction: str = None):
        print(f"[AI Adapter]: Solicitando STREAM do modelo: {model_name}")
        if "gemini" in model_name:
            yield from self._get_gemini_completion(model_name, conversation_history, system_instruction, stream=True)
        elif "gpt" in model_name:
            yield from self._get_openai_completion(model_name, conversation_history, system_instruction, stream=True)
        else:
            raise NotImplementedError(f"Streaming para '{model_name}' não suportado.")

    def get_completion_sync(self, model_name: str, prompt: str, system_instruction: str = None, json_mode: bool = False) -> str:
        print(f"[AI Adapter]: Solicitando resposta SÍNCRONA do modelo: {model_name}")
        conversation_history = [{"role": "user", "parts": [prompt]}]
        if "gemini" in model_name:
            response_generator = self._get_gemini_completion(model_name, conversation_history, system_instruction, stream=False, json_mode=json_mode)
        elif "gpt" in model_name:
            response_generator = self._get_openai_completion(model_name, conversation_history, system_instruction, stream=False, json_mode=json_mode)
        else:
            raise NotImplementedError(f"Chamada síncrona para '{model_name}' não suportada.")
        
        return next(response_generator, "")

    def _get_gemini_completion(self, model_name: str, conversation_history: list, system_instruction: str, stream: bool, json_mode: bool = False):
        try:
            generation_config = genai.GenerationConfig(response_mime_type="application/json") if json_mode else None
            
            model = genai.GenerativeModel(
                model_name=model_name,
                system_instruction=system_instruction,
                generation_config=generation_config
            )
            response = model.generate_content(conversation_history, stream=stream)

            if stream:
                for chunk in response:
                    if chunk.text:
                        yield chunk.text
                yield "[STREAM_END]"
            else:
                yield response.text
        except Exception as e:
            error_message = f"Erro no AI Adapter ao chamar o modelo {model_name}: {e}"
            print(error_message)
            raise

    def _get_openai_completion(self, model_name: str, conversation_history: list, system_instruction: str, stream: bool, json_mode: bool = False):
        try:
            client = openai.OpenAI()
            
            messages = []
            if system_instruction:
                messages.append({"role": "system", "content": system_instruction})
            
            for msg in conversation_history:
                role = "assistant" if msg["role"] == "model" else "user"
                content = " ".join(msg["parts"])
                messages.append({"role": role, "content": content})

            response_format = {"type": "json_object"} if json_mode else {"type": "text"}

            response = client.chat.completions.create(
                model=model_name,
                messages=messages,
                stream=stream,
                response_format=response_format
            )

            if stream:
                for chunk in response:
                    content = chunk.choices[0].delta.content or ""
                    if content:
                        yield content
                yield "[STREAM_END]"
            else:
                yield response.choices[0].message.content
        except Exception as e:
            error_message = f"Erro no AI Adapter ao chamar o modelo {model_name}: {e}"
            print(error_message)
            raise