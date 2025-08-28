# Kiku: Interface Web Segura para Agente de IA Conversacional

## 📖 Sobre o Projeto

Este projeto foi desenvolvido como parte da avaliação da disciplina de Desenvolvimento WEB 2 no Instituto Federal Rio Grandense, campus Farroupilha. "Kiku" é uma aplicação web segura, construída com o framework Flask, que serve como interface para um motor de IA conversacional subjacente.

O foco do trabalho foi a implementação dos requisitos de segurança web, a aplicação da arquitetura MVC, e a integração com um backend de IA não-convencional que possui funcionalidades avançadas de memória e orquestração, com todos os dados sendo gerenciados de forma isolada para cada usuário.

---

## ✨ Funcionalidades Implementadas

### Funcionalidades da Aplicação Web (Kiku)

* **Autenticação Segura de Usuários:**
    * Sistema completo de Registro, Login e Logout utilizando **Flask-Login**.
    * Senhas armazenadas de forma segura no banco de dados com **hash** (`werkzeug.security`).
    * Proteção de rotas, garantindo que apenas usuários autenticados possam aceder ao chat.

* **Segurança Web Completa:**
    * **Proteção contra CSRF:** Formulários de login e registro protegidos com tokens CSRF através de `Flask-WTF`.
    * **Proteção contra SQL Injection:** O uso exclusivo do ORM **SQLAlchemy** previne ataques de injeção de SQL.
    * **Proteção contra XSS:** O motor de templates Jinja2 e os formulários WTForms escapam automaticamente as entradas do usuário, prevenindo ataques de Cross-Site Scripting.

* **Interface de Chat em Tempo Real:**
    * Comunicação bidirecional e persistente através de **WebSockets (Flask-SocketIO)**.
    * Respostas da IA enviadas em *stream*, criando um efeito de "digitação" em tempo real para o usuário.
    * O histórico da conversa é salvo e recarregado automaticamente a cada nova sessão.

### Funcionalidades do Motor de IA

* **Orquestração Inteligente de Respostas:**
    * A cada prompt, o backend utiliza um modelo de IA para gerar um **"Plano de Ação" em JSON**.
    * Este plano determina a intenção do usuário, a necessidade de aceder à memória de longo prazo e **extrai factos em tempo real** da conversa (ex: nome do usuário).
    * O sistema implementa um **"Gerenciador de Fallback Inteligente"** que "trava" modelos de IA que falham, garantindo resiliência.

* **Memória Cognitiva Multi-Nível e por Usuário:**
    * Cada usuário possui uma "memória" de IA completamente isolada e persistente.
    * **Nível 4 (Armazenamento de Factos):** Factos explícitos sobre o utilizador são guardados num ficheiro `buddy_config.json` para personalizar a conversa.
    * **Nível 3 (Arquivo de Longo Prazo - RAG):** Tópicos de conversa são processados (resumidos e etiquetados) e arquivados num banco de dados vetorial  para recuperação semântica futura.
    * **Nível 2 (Contexto de Sessão - "Bancada de Trabalho"):** Um `Contextualizador Semântico` agrupa conversas em tópicos coesos. Um `"Curador"` consulta esta memória de sessão para injetar contexto relevante nas respostas.
    * **Nível 1 (Contexto Imediato):** O histórico recente da conversa é sempre utilizado para garantir a fluidez do diálogo.

---

## 🛠️ Tecnologias Utilizadas

#### **Backend**
* Python 3
* Flask, Flask-SocketIO, Flask-Login, Flask-WTF
* SQLAlchemy
* `sentence-transformers`
* `faiss-cpu`
* `google-generativeai` / `openai`
* `python-dotenv`

#### **Frontend**
* HTML, CSS, JavaScript
* Socket.IO Client

#### **Banco de Dados**
* SQLite

---

## 🚀 Configuração e Instalação

Siga os passos abaixo para configurar e executar o projeto localmente.

**Passos:**

1.  **Vá até o diretório correspondente em seu PC:**
    ```bash
    cd ...\buddy_app
    ```

2.  **Crie e configure o arquivo `.env`:**
    Na raiz do projeto, crie um arquivo chamado `.env` e adicione suas chaves de API:
    ```
    GEMINI_API_KEY=SUA_CHAVE_DO_GEMINI_AQUI
    OPENAI_API_KEY=SUA_CHAVE_DA_OPENAI_AQUI
    ```

3.  **Crie e ative o ambiente virtual:**
    ```bash
    # Criar
    python -m venv .venv

    # Ativar (Windows)
    .\.venv\Scripts\activate

    # Ativar (macOS/Linux)
    source .venv/bin/activate
    ```

4.  **Instale as dependências:**
    ```bash
    pip install -r requirements.txt
    ```

5.  **Inicialize o banco de dados:**
    ```bash
    python -m flask init-db
    ```

6.  **Execute a aplicação:**
    ```bash
    python -m flask run
    ```

7.  Acesse a aplicação no seu navegador em: **`http://127.0.0.1:5000`**

---

## 🕹️ Como Usar

1.  Acesse a página inicial e será redirecionado para a tela de login.
2.  Crie um novo usuário através da página de registro.
3.  Faça login com as suas credenciais.
4.  Você será direcionado para a interface de chat "Kiku" para conversar com o assistente.
5.  O histórico da sua conversa será salvo. Ao fazer logout e login novamente, as mensagens anteriores serão recarregadas.

---

## 👨‍💻 Autores

**Viktor Clipel, Carlos Leandro Goulart dos Santos Junior, Antônio Comin**