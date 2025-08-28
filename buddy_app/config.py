
import os

basedir = os.path.abspath(os.path.dirname(__file__))

class Config:
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'uma-chave-secreta-muito-dificil-de-adivinhar'
    SQLALCHEMY_DATABASE_URI = os.environ.get('DATABASE_URL') or \
        'sqlite:///' + os.path.join(basedir, 'instance', 'app.db')
    SQLALCHEMY_TRACK_MODIFICATIONS = False

    BUDDY_DATA_BASE_PATH = os.path.join(basedir, 'user_data')

    TAGGING_RULES = (
        "CRITICAL RULE FOR TAGS: Tags must describe the CORE TOPIC of the text. "
        "DO NOT create tags about the language (e.g., 'portuguese-language'). "
        "DO NOT create generic tags like 'greeting', 'question', 'user-preference', or 'conversational'. "
        "Good tags are specific, hyphenated, lowercase concepts like 'memory-architecture', 'python-for-loop', or 'agile-methodology'. "
        "The primary goal is accuracy and specificity."
    )

    MODEL_CONFIG = {
        "classifier": "gemini-1.5-flash-latest",
        "summarizer": "gemini-1.5-flash-latest",
        "tagger": "gemini-1.5-flash-latest",
        "segmenter": "gemini-1.5-flash-latest"
    }

    CONTEXTUALIZER_SIMILARITY_THRESHOLD = 0.7

    DYNAMIC_MODEL_RANKINGS = {}