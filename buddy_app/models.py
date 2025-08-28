from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash
from flask_login import UserMixin
from pydantic import BaseModel, Field 
from typing import List, Literal, Optional, Dict, Any 

db = SQLAlchemy()

class User(db.Model, UserMixin):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password_hash = db.Column(db.String(256))

    def set_password(self, password):
        self.password_hash = generate_password_hash(password)

    def check_password(self, password):
        return check_password_hash(self.password_hash, password)

    def __repr__(self):
        return f'<User {self.username}>'

class Message(BaseModel):
    role: Literal["user", "model"]
    parts: List[str]

class ActionPlan(BaseModel):
    specialty: str = "conversation"
    needs_search: bool = False
    needs_long_term_memory: bool = False
    tags: List[str] = Field(default_factory=list)
    extracted_facts: Optional[Dict[str, Any]] = None