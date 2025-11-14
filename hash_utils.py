"""
Hash utilities for canonical JSON hashing and SHA-256 computation.
"""

import json
import hashlib
from typing import Dict, Any


def canonical_json(data: Dict[str, Any]) -> str:
    """
    Convert Python dict to canonical JSON string.
    
    Rules:
    - Sort keys alphabetically
    - Remove spaces using separators=(",", ":")
    - ensure_ascii=False
    
    Args:
        data: Python dictionary to convert
        
    Returns:
        Canonical JSON string
    """
    return json.dumps(
        data,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False
    )


def compute_sha256(data: Dict[str, Any]) -> str:
    """
    Compute SHA-256 hash of canonical JSON representation.
    
    Args:
        data: Python dictionary to hash
        
    Returns:
        SHA-256 hex digest (64 characters)
    """
    canonical_str = canonical_json(data)
    hash_bytes = hashlib.sha256(canonical_str.encode('utf-8')).digest()
    return hash_bytes.hex()

