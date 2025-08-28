import os
from dotenv import load_dotenv

_MASTER_MODEL_LIST = [
    {
        "name": "gemini-1.5-pro-latest",
        "provider": "gemini",
        "api_key_name": "GEMINI_API_KEY",
        "rankings": {
            "conversation": 8,
            "creative_writing": 9,
            "logical_reasoning": 10,
            "code_generation": 10
        }
    },
    {
        "name": "gemini-1.5-flash-latest",
        "provider": "gemini",
        "api_key_name": "GEMINI_API_KEY",
        "rankings": {
            "conversation": 10,
            "creative_writing": 7,
            "logical_reasoning": 6,
            "code_generation": 8
        }
    },
    {
        "name": "gpt-4o",
        "provider": "openai",
        "api_key_name": "OPENAI_API_KEY",
        "rankings": {
            "conversation": 9,
            "creative_writing": 10,
            "logical_reasoning": 9,
            "code_generation": 9
        }
    }
]

def build_available_model_rankings():
    """
    Verifica as chaves de API disponíveis no ambiente e constrói dinamicamente
    o dicionário MODEL_RANKINGS apenas com os modelos utilizáveis.
    """
    dotenv_path = os.path.join(os.path.dirname(__file__), '..', '.env')
    load_dotenv(dotenv_path=dotenv_path, override=True)

    available_rankings = {}
    print("[Model Resolver]: Verificando chaves de API e construindo rankings de modelos...")
    
    for model_info in _MASTER_MODEL_LIST:
        api_key = os.getenv(model_info["api_key_name"])
        if api_key:
            model_name = model_info["name"]
            for specialty, rank in model_info["rankings"].items():
                if specialty not in available_rankings:
                    available_rankings[specialty] = {}
                available_rankings[specialty][model_name] = rank
    
    print(f"[Model Resolver]: Modelos disponíveis e ranqueados: {available_rankings}")
    return available_rankings