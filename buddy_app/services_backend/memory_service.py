# Arquivo: services_backend/memory_service.py (versão multi-usuário)

import os
import json
import numpy as np
import faiss
import sqlite3
from config import Config 

class MemoryService:
    def __init__(self, user_id: int, embedding_model, summarizer, tagger, segmenter):
        if not user_id:
            raise ValueError("O ID do usuário é necessário para inicializar o MemoryService.")
        
        self.user_data_path = os.path.join(Config.BUDDY_DATA_BASE_PATH, str(user_id))
        self._ensure_user_directory_exists()

        self.config_path = os.path.join(self.user_data_path, 'buddy_config.json')
        self.history_path = os.path.join(self.user_data_path, 'history.json')
        self.index_path = os.path.join(self.user_data_path, 'memory.faiss')
        self.db_path = os.path.join(self.user_data_path, 'memory.db')

        self._ensure_files_exist()
        
        self.embedding_model = embedding_model
        
        self._create_memory_table()
        
        try:
            self.index = faiss.read_index(self.index_path)
            print(f"[Memory Service para Usuário {user_id}]: Índice FAISS carregado com {self.index.ntotal} memórias.")
        except RuntimeError:

            embedding_dim = self.embedding_model.get_sentence_embedding_dimension()
            self.index = faiss.IndexFlatL2(embedding_dim)
            print(f"[Memory Service para Usuário {user_id}]: Novo índice FAISS criado.")
        
        self.summarizer = summarizer
        self.tagger = tagger
        self.segmenter = segmenter
        
        self.predictive_tags_accumulator = []
        self.session_tags_cache = []
        
        self.workbench = []
        print(f"[Memory Service para Usuário {user_id}]: Bancada de Trabalho (Nível 2) inicializada.")

    def _ensure_user_directory_exists(self):
        """Garante que o diretório de dados do usuário exista."""
        if not os.path.exists(self.user_data_path):
            os.makedirs(self.user_data_path)
            print(f"[Memory Service]: Diretório criado para o usuário em: {self.user_data_path}")

    def _get_db_connection(self):
        return sqlite3.connect(self.db_path)

    def _create_memory_table(self):
        conn = self._get_db_connection()
        try:
            cursor = conn.cursor()
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS memories (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    summary TEXT NOT NULL,
                    tags TEXT,
                    original_chunk TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            conn.commit()
        finally:
            conn.close()

    def get_master_tag_list(self) -> list:
        conn = self._get_db_connection()
        try:
            cursor = conn.cursor()
            cursor.execute("SELECT tags FROM memories")
            master_tags = set()
            for row in cursor.fetchall():
                if row[0]:
                    tags = json.loads(row[0])
                    master_tags.update(tags)
            return sorted(list(master_tags))
        finally:
            conn.close()

    def add_block_to_workbench(self, block_data: dict):
        print(f"[Memory Service]: Bloco de tópico '{block_data.get('tags', [])}' adicionado à Bancada de Trabalho.")
        self.workbench.append(block_data)

    def process_conversation_block_for_archiving(self, block_data: dict):
        try:
            print(f"[Memory Service]: Recebido bloco para arquivamento em background.")
            conversation_chunk = block_data.get("block", [])
            
            if not conversation_chunk or len(conversation_chunk) < 2:
                print("[Memory Service]: Bloco recebido é muito curto para ser processado. Ignorando.")
                return

            segmented_topics = self.segmenter.segment_conversation_by_topic(conversation_chunk)

            for topic_name, topic_chunk in segmented_topics.items():
                print(f"[Memory Service]: A processar o tópico '{topic_name}' do bloco...")
                if not topic_chunk or len(topic_chunk) < 2:
                    continue
                summary = self.summarizer.summarize_conversation_chunk(topic_chunk)
                if summary:
                    master_tags = self.get_master_tag_list()
                    final_tags = self.tagger.refine_and_consolidate_tags(
                        summary=summary, 
                        candidate_tags=self.predictive_tags_accumulator,
                        session_tags=self.session_tags_cache,
                        master_tag_list=master_tags
                    )
                    
                    for tag in final_tags:
                        if tag not in self.session_tags_cache:
                            self.session_tags_cache.append(tag)
                    
                    self.add_to_long_term_memory(summary, final_tags, topic_chunk)
            
            self.predictive_tags_accumulator = []
            print(f"[Memory Service]: Arquivamento do bloco concluído.")
        
        except Exception as e:
            print(f"[Memory Service]: ERRO CRÍTICO DURANTE ARQUIVAMENTO EM BACKGROUND: {e}")

    def add_to_long_term_memory(self, summary: str, tags: list, original_chunk: list):
        conn = self._get_db_connection()
        try:
            summary_embedding = self.embedding_model.encode([summary])
            self.index.add(np.array(summary_embedding, dtype=np.float32))
            
            tags_json = json.dumps(tags)
            chunk_json = json.dumps(original_chunk, ensure_ascii=False)
            
            cursor = conn.cursor()
            cursor.execute(
                "INSERT INTO memories (summary, tags, original_chunk) VALUES (?, ?, ?)",
                (summary, tags_json, chunk_json)
            )
            conn.commit()
            
            faiss.write_index(self.index, self.index_path)
            print(f"[Memory Service]: Nova memória arquivada no FAISS e DB. Total: {self.index.ntotal}")
        except Exception as e:
            print(f"Erro ao salvar na memória de longo prazo: {e}")
        finally:
            conn.close()
            
    def retrieve_relevant_memories(self, user_prompt: str, n_results: int = 3) -> str:
        conn = self._get_db_connection()
        try:
            if self.index.ntotal == 0:
                return ""
            prompt_embedding = self.embedding_model.encode([user_prompt])
            distances, indices = self.index.search(np.array(prompt_embedding, dtype=np.float32), n_results)
            
            valid_indices = [i for i in indices[0] if i != -1]
            if not valid_indices:
                return ""
            
            db_ids = [i + 1 for i in valid_indices]
            placeholders = ','.join('?' for _ in db_ids)
            cursor = conn.cursor()
            cursor.execute(f"SELECT summary, tags FROM memories WHERE id IN ({placeholders})", db_ids)
            
            results = cursor.fetchall()
            found_memories = "\n".join(
                f"- Lembrete de uma conversa anterior (Tópicos: {', '.join(json.loads(res[1])) if res[1] else 'N/A'}): {res[0]}"
                for res in results
            )
            print(f"[Memory Service]: Memórias relevantes recuperadas do FAISS e DB.")
            return found_memories
        except Exception as e:
            print(f"Erro ao recuperar memórias: {e}")
            return "Ocorreu um erro enquanto eu tentava aceder à minha memória de longo prazo."
        finally:
            conn.close()

    def _ensure_files_exist(self):
        files_to_check = {
            self.config_path: '{}',
            self.history_path: '[]'
        }
        for path, default_content in files_to_check.items():
            if not os.path.exists(path):
                print(f"[Memory Service]: Ficheiro '{os.path.basename(path)}' não encontrado. A criar um novo.")
                with open(path, 'w', encoding='utf-8') as f:
                     f.write(default_content)
                     
    def add_predictive_tags(self, tags: list):
        self.predictive_tags_accumulator.extend(tags)
        self.predictive_tags_accumulator = list(set(self.predictive_tags_accumulator))
    
    def add_to_history(self, message: dict):
        history = self.get_short_term_memory()
        history.append(message)
        self._save_json(self.history_path, history)
        
    def load_facts(self):
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                content = f.read()
                return json.loads(content) if content else {}
        except (json.JSONDecodeError, IOError) as e:
            print(f"Erro ao carregar o ficheiro de factos: {e}")
            return {}
            
    def add_fact(self, new_facts: dict):
        if not isinstance(new_facts, dict):
            return
        current_facts = self.load_facts()
        current_facts.update(new_facts)
        self._save_json(self.config_path, current_facts)
        
    def get_short_term_memory(self):
        try:
            with open(self.history_path, 'r', encoding='utf-8') as f:
                content = f.read()
                return json.loads(content) if content else []
        except (json.JSONDecodeError, IOError) as e:
            print(f"Erro ao carregar o ficheiro de histórico: {e}")
            return []
            
    def _save_json(self, file_path: str, data: list | dict):
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
        except IOError as e:
            print(f"Erro ao salvar o ficheiro {os.path.basename(file_path)}: {e}")