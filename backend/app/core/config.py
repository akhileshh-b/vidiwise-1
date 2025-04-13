import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Base directory
BASE_DIR = Path(__file__).resolve().parent.parent.parent

# API Settings
API_HOST = os.getenv("API_HOST", "0.0.0.0")
API_PORT = int(os.getenv("API_PORT", "8080"))
API_CORS_ORIGINS = os.getenv("FRONTEND_URL", "http://localhost:3000").split(",")

# Storage Settings
STORAGE_DIR = os.path.join(BASE_DIR, os.getenv("STORAGE_DIR", "video_findings"))

# API Keys
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY environment variable is not set")

# Logging
LOG_LEVEL = os.getenv("LOG_LEVEL", "DEBUG") 