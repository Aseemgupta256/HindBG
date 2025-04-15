# config.py
import os

class Config:
    SECRET_KEY = os.environ.get('SECRET_KEY', 'generate-a-secure-random-key-here')
    SQLALCHEMY_DATABASE_URI = os.environ.get('DATABASE_URL', 'sqlite:///app.db')
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    UPLOAD_FOLDER = os.path.join(os.getcwd(), 'static', 'uploads')
    PROCESSED_FOLDER = os.path.join(os.getcwd(), 'static', 'processed')
    MAX_CONTENT_LENGTH = 10 * 1024 * 1024  # 10MB max upload

class DevelopmentConfig(Config):
    DEBUG = True

class ProductionConfig(Config):
    DEBUG = False
    # Ensure SECRET_KEY and DATABASE_URL are set via environment variables in production.
