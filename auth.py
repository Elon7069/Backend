"""
API Key authentication for HAWKEYE Fraud Detection API.

Frontend must send header:
"x-api-key": "<BACKEND_API_KEY>"
"""
from fastapi import Header, HTTPException
from fastapi.responses import JSONResponse
from typing import Optional
import os

# Load API key from environment
BACKEND_API_KEY = os.getenv("BACKEND_API_KEY")

# Warn if API key is missing (but don't fail startup)
if not BACKEND_API_KEY:
    import warnings
    warnings.warn(
        "WARNING: BACKEND_API_KEY not found in environment. "
        "API key authentication will fail. Please set BACKEND_API_KEY in .env file.",
        UserWarning
    )


async def verify_api_key(x_api_key: Optional[str] = Header(None, alias="x-api-key")):
    """
    Verify API key from request header.
    
    Raises HTTPException with 401 status if API key is missing or invalid.
    Returns error in format: { "error": "Unauthorized. Invalid or missing API key." }
    """
    if not BACKEND_API_KEY:
        raise HTTPException(
            status_code=500,
            detail="Server configuration error: API key not configured"
        )
    
    if x_api_key != BACKEND_API_KEY:
        # Raise HTTPException which FastAPI will convert to JSON
        # The response will be: {"detail": "Unauthorized. Invalid or missing API key."}
        # But we want {"error": ...}, so we'll use a custom exception handler
        raise HTTPException(
            status_code=401,
            detail="Unauthorized. Invalid or missing API key."
        )
    
    return True

