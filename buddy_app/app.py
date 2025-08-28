import os
import sys

project_root = os.path.abspath(os.path.dirname(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from flask import Flask, render_template, request, redirect, url_for, flash, jsonify
from config import Config
from models import db, User
from forms import LoginForm, RegistrationForm
from flask_login import LoginManager, login_user, logout_user, login_required, current_user
from flask_socketio import SocketIO, emit

from dotenv import load_dotenv
from services_backend.ai_adapter import AI_Adapter
from services_backend.memory_service import MemoryService
from services_backend.orchestrator_service import OrchestratorService
from services_backend.summarizer_service import SummarizerService
from services_backend.tagger_service import TaggerService
from services_backend.segmenter_service import SegmenterService
from services_backend.utils.model_resolver import build_available_model_rankings
from sentence_transformers import SentenceTransformer

app = Flask(__name__)
app.config.from_object(Config)
socketio = SocketIO(app)

try:
    os.makedirs(app.instance_path)
except OSError:
    pass

db.init_app(app)

login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login' 

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

print("[BuddyApp]: Carregando serviços de IA globais...")
load_dotenv()
app.config['DYNAMIC_MODEL_RANKINGS'] = build_available_model_rankings()

ai_adapter = AI_Adapter()
summarizer_service = SummarizerService(ai_adapter)
tagger_service = TaggerService(ai_adapter)
segmenter_service = SegmenterService(ai_adapter)
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

user_orchestrators = {}
print("[BuddyApp]: Serviços globais de IA prontos.")

def get_user_orchestrator():
    user_id = current_user.id
    if user_id not in user_orchestrators:
        print(f"[BuddyApp]: Criando nova instância de serviços para o usuário {user_id}...")
        memory = MemoryService(
            user_id=user_id,
            embedding_model=embedding_model,
            summarizer=summarizer_service,
            tagger=tagger_service,
            segmenter=segmenter_service
        )
        orchestrator = OrchestratorService(
            memory_service=memory,
            ai_adapter=ai_adapter,
            embedding_model=embedding_model
        )
        user_orchestrators[user_id] = orchestrator
    return user_orchestrators[user_id]

@app.route("/")
@login_required 
def home():
    get_user_orchestrator()
    return render_template("index.html", username=current_user.username)


@socketio.on('connect')
def handle_connect():
    """Acionado quando um cliente se conecta. Envia o histórico do chat."""
    if current_user.is_authenticated:
        orchestrator = get_user_orchestrator()
        history = orchestrator.get_full_history()
        emit('load_history', {'history': history})
        print(f"Histórico enviado para o usuário {current_user.id}")

@socketio.on('new_message')
def handle_new_message(data):
    user_message_text = data.get('message', '')
    if not user_message_text:
        return
    orchestrator = get_user_orchestrator()
    user_message = {"role": "user", "parts": [user_message_text]}
    orchestrator.add_to_history(user_message)
    emit('stream_start')
    response_generator = orchestrator.generate_response_stream()
    full_response = ""
    for chunk in response_generator:
        if chunk == "[STREAM_END]":
            break
        full_response += chunk
        emit('stream_chunk', {'data': chunk})
    emit('stream_end')
    if full_response:
        model_response = {"role": "model", "parts": [full_response]}
        orchestrator.add_to_history(model_response)

@app.route('/login', methods=['GET', 'POST'])
def login():
    if current_user.is_authenticated:
        return redirect(url_for('home'))
    form = LoginForm()
    if form.validate_on_submit():
        user = User.query.filter_by(username=form.username.data).first()
        if user and user.check_password(form.password.data):
            login_user(user)
            return redirect(url_for('home'))
        else:
            flash('Login sem sucesso. Verifique o nome de usuário e a senha.', 'danger')
    return render_template('login.html', form=form)

@app.route('/register', methods=['GET', 'POST'])
def register():
    if current_user.is_authenticated:
        return redirect(url_for('home'))
    form = RegistrationForm()
    if form.validate_on_submit():
        new_user = User(username=form.username.data, email=form.email.data)
        new_user.set_password(form.password.data)
        db.session.add(new_user)
        db.session.commit()
        flash('Sua conta foi criada! Agora você pode fazer o login.', 'success')
        return redirect(url_for('login'))
    return render_template('register.html', form=form)

@app.route('/logout')
@login_required
def logout():
    if current_user.id in user_orchestrators:
        del user_orchestrators[current_user.id]
        print(f"[BuddyApp]: Instância de serviços removida para o usuário {current_user.id}.")
    logout_user()
    flash('Você foi desconectado com sucesso.')
    return redirect(url_for('login'))

@app.cli.command("init-db")
def init_db_command():
    with app.app_context():
        db.create_all()
    print("Banco de dados inicializado.")

if __name__ == "__main__":
    socketio.run(app, debug=True)