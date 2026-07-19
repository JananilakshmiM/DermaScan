from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash
import datetime

db = SQLAlchemy()

class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(150), unique=True, nullable=False)
    password_hash = db.Column(db.String(256), nullable=False)
    
    predictions = db.relationship('PredictionHistory', backref='user', lazy=True)

    def set_password(self, password):
        self.password_hash = generate_password_hash(password)

    def check_password(self, password):
        return check_password_hash(self.password_hash, password)

class PredictionHistory(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    
    original_image_path = db.Column(db.String(255), nullable=False)
    heatmap_image_path = db.Column(db.String(255), nullable=True)
    
    class_name = db.Column(db.String(100), nullable=False)
    confidence = db.Column(db.Float, nullable=False)
    risk_level = db.Column(db.String(50), nullable=False)
    
    timestamp = db.Column(db.DateTime, default=datetime.datetime.utcnow)
