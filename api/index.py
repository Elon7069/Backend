"""
Vercel serverless function entry point for FastAPI app.
"""
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app import app
from mangum import Mangum

# Vercel handler
handler = Mangum(app)

