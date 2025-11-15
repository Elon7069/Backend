"""
Authentication module for HAWKEYE Fraud Detection API.
Provides API key authentication for protected endpoints.
"""
import os
from fastapi import HTTPException, Header
from typing import Optional

# Load API key from environment variable
BACKEND_API_KEY = os.getenv("BACKEND_API_KEY")


async def verify_api_key(x_api_key: Optional[str] = Header(None)) -> str:
    """
    Verify API key from request header.
    
    Args:
        x_api_key: API key from request header
        
    Returns:
        The validated API key
        
    Raises:
        HTTPException: If API key is invalid or missing
    """
    if not BACKEND_API_KEY:
        # If no BACKEND_API_KEY is set in environment, allow all requests
        # This is useful for local development
        return ""
    
    if not x_api_key:
        raise HTTPException(
            status_code=401, 
            detail="Missing API key. Include 'x-api-key' header in your request."
        )
    
    if x_api_key != BACKEND_API_KEY:
        raise HTTPException(
            status_code=401, 
            detail="Invalid API key"
        )
    
    return x_api_key
