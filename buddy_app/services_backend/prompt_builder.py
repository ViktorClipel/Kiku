import json
from .memory_service import MemoryService

class PromptBuilder:
    def __init__(self, memory_service: MemoryService):
        self.memory_service = memory_service
        self.base_system_instruction = """You are Kiku, a desktop AI companion. Your personality is helpful and friendly. You need to respond the input in portuguese.
            Respond to the user concisely."""

    def build_context(self):
        known_facts = self.memory_service.load_facts()

        system_instruction_final = self.base_system_instruction
        if known_facts:
            facts_str = json.dumps(known_facts, ensure_ascii=False, indent=2)
            facts_context = (
                "\n\n--- ADDITIONAL INFORMATION YOU ALREADY KNOW ---\n"
                "Here are some facts you already know about the user. Do not ask about them again and use them to personalize the conversation.\n"
                f"Known Facts:\n{facts_str}\n"
                "--- END OF ADDITIONAL INFORMATION ---"
            )
            system_instruction_final += facts_context
        
        conversation_history = self.memory_service.get_short_term_memory()

        return system_instruction_final, conversation_history