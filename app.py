from fastapi import FastAPI, UploadFile, File, HTTPException, Depends
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
import io
from fraud_explainer import (
    load_model,
    prepare_features,
    predict_fraud,
    get_risk_level,
    initialize_openai_client,
    explain_transactions_parallel,
    validate_csv_schema
)
import os
import warnings
from dotenv import load_dotenv
from hash_utils import compute_sha256, canonical_json
from aptos_client import (
    publish_hash_on_aptos,
    get_explorer_url,
    get_account_explorer_url,
    search_hash_on_aptos,
    get_aptos_account
)
from auth import verify_api_key, BACKEND_API_KEY

# ============================================================================
# API Key Authentication
# ============================================================================
# Frontend must send header: "x-api-key": "<BACKEND_API_KEY>"
# Protected endpoints: /detect, /explain, /report, /blockchain/publish, /verify
# Public endpoints: /health, /docs, /openapi.json, /
# ============================================================================

load_dotenv()

# Check for BACKEND_API_KEY and warn if missing
if not BACKEND_API_KEY:
    warnings.warn(
        "WARNING: BACKEND_API_KEY not found in environment. "
        "API key authentication will fail. Please set BACKEND_API_KEY in .env file.",
        UserWarning
    )

app = FastAPI(title="HAWKEYE Fraud Detection API")

# Add CORS middleware for frontend compatibility
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific frontend URLs
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Custom exception handler for API key authentication errors
@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc: HTTPException):
    """
    Custom exception handler to format API key authentication errors.
    Returns { "error": "..." } format for 401 errors.
    """
    if exc.status_code == 401:
        return JSONResponse(
            status_code=401,
            content={"error": "Unauthorized. Invalid or missing API key."}
        )
    # For other HTTP exceptions, use default format
    return JSONResponse(
        status_code=exc.status_code,
        content={"detail": exc.detail}
    )

# Maximum file size: 150MB
MAX_FILE_SIZE = 150 * 1024 * 1024  # 150MB in bytes

# Load model at startup - graceful handling for deployment
MODEL = None
try:
    # Get the directory where this script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(script_dir, "xgb_model.pkl")
    MODEL = load_model(model_path)
    print("Model loaded successfully")
except Exception as e:
    error_msg = f"WARNING: Failed to load model: {e}"
    print(error_msg)
    print("API will continue but fraud detection endpoints may not work")
    print("Note: Make sure xgb_model.pkl is in the Backend directory and committed to git")
    # Don't raise - allow API to start for health checks

# Initialize OpenAI client (will be created when needed)
openai_client = None


def get_openai_client():
    """Get or initialize OpenAI client."""
    global openai_client
    if openai_client is None:
        openai_client = initialize_openai_client()
    return openai_client


@app.get("/")
def home():
    return {"status": "running", "message": "HAWKEYE Fraud Detection API is live!"}


@app.get("/health")
async def health():
    """
    Health check endpoint - returns static status without heavy checks.
    """
    # Check OpenAI configuration
    openai_configured = "configured" if os.getenv("OPENAI_API_KEY") else "not_configured"
    
    # Check blockchain readiness
    blockchain_ready = "ready" if get_aptos_account() is not None else "not_configured"
    
    # Check model status
    model_status = "loaded" if MODEL is not None else "not_loaded"
    
    return {
        "status": "ok",
        "ml_model": model_status,
        "openai": openai_configured,
        "blockchain": blockchain_ready
    }


@app.post("/detect")
async def detect(
    file: UploadFile = File(...), 
    top_k: int = 10, 
    threshold: float = 0.9,
    api_key: str = Depends(verify_api_key)
):
    """
    Detect fraud based on fraud_score only (no GPT explanation).
    Fast endpoint for quick fraud detection without AI explanations.
    """
    if MODEL is None:
        raise HTTPException(status_code=500, detail="Model not loaded. Please check server logs.")
    
    # More lenient content type check - accept CSV files even if content_type is not set
    if file.content_type and file.content_type not in ["text/csv", "application/vnd.ms-excel", "text/plain", "application/csv"]:
        # Check file extension as fallback
        if not (file.filename and file.filename.lower().endswith('.csv')):
            raise HTTPException(status_code=400, detail="Upload a valid CSV file. Expected .csv file.")

    try:
        content = await file.read()
        if not content or len(content) == 0:
            raise HTTPException(status_code=400, detail="Uploaded file is empty.")
        
        # Check file size limit
        file_size = len(content)
        if file_size > MAX_FILE_SIZE:
            raise HTTPException(
                status_code=413,
                detail=f"File too large. Maximum size is {MAX_FILE_SIZE / (1024 * 1024):.1f}MB, received {file_size / (1024 * 1024):.2f}MB"
            )
        
        # Try to read CSV
        df = pd.read_csv(io.BytesIO(content))
        if df.empty:
            raise HTTPException(status_code=400, detail="CSV file is empty or has no data.")
    except HTTPException:
        raise  # Re-raise HTTPException (e.g., file size errors)
    except pd.errors.EmptyDataError:
        raise HTTPException(status_code=400, detail="CSV file is empty or has no data.")
    except pd.errors.ParserError as e:
        raise HTTPException(status_code=400, detail=f"Error parsing CSV file: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error loading CSV: {str(e)}")

    # Validate CSV schema
    is_valid, missing_columns = validate_csv_schema(df)
    if not is_valid:
        return JSONResponse(
            status_code=400,
            content={"error": f"Missing columns: {', '.join(missing_columns)}"}
        )

    # Prepare features (remove Class column if present)
    try:
        df_features = prepare_features(df.copy())
        results = predict_fraud(MODEL, df_features)
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": "Model inference failed", "details": str(e)}
        )
    
    # Filter and sort flagged transactions (use threshold parameter)
    flagged = results[results["fraud_score"] >= threshold].sort_values("fraud_score", ascending=False)
    top = flagged.head(top_k)
    
    # Convert to records with risk levels
    top_records = []
    for idx, row in top.iterrows():
        record = row.to_dict()
        record["risk_level"] = get_risk_level(record["fraud_score"])
        top_records.append(record)

    return {
        "flagged_count": len(flagged),
        "top_suspicious": top_records,
        "threshold_used": threshold
    }


@app.post("/explain")
async def explain(
    file: UploadFile = File(...), 
    top_k: int = 5, 
    threshold: float = 0.9, 
    parallel: int = 5,
    api_key: str = Depends(verify_api_key)
):
    """
    Detect fraud + Generate GPT explanations for flagged transactions.
    Uses parallel processing for faster explanation generation.
    """
    if MODEL is None:
        raise HTTPException(status_code=500, detail="Model not loaded. Please check server logs.")
    
    # Check for OpenAI API key
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return JSONResponse(
            status_code=500,
            content={"error": "OpenAI failed", "details": "Missing OpenAI API key in .env"}
        )

    # More lenient content type check
    if file.content_type and file.content_type not in ["text/csv", "application/vnd.ms-excel", "text/plain", "application/csv"]:
        if not (file.filename and file.filename.lower().endswith('.csv')):
            raise HTTPException(status_code=400, detail="Upload a valid CSV file. Expected .csv file.")

    try:
        content = await file.read()
        if not content or len(content) == 0:
            raise HTTPException(status_code=400, detail="Uploaded file is empty.")
        
        # Check file size limit
        file_size = len(content)
        if file_size > MAX_FILE_SIZE:
            raise HTTPException(
                status_code=413,
                detail=f"File too large. Maximum size is {MAX_FILE_SIZE / (1024 * 1024):.1f}MB, received {file_size / (1024 * 1024):.2f}MB"
            )
        
        df = pd.read_csv(io.BytesIO(content))
        if df.empty:
            raise HTTPException(status_code=400, detail="CSV file is empty or has no data.")
    except HTTPException:
        raise  # Re-raise HTTPException (e.g., file size errors)
    except pd.errors.EmptyDataError:
        raise HTTPException(status_code=400, detail="CSV file is empty or has no data.")
    except pd.errors.ParserError as e:
        raise HTTPException(status_code=400, detail=f"Error parsing CSV file: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error loading CSV: {str(e)}")

    # Validate CSV schema
    is_valid, missing_columns = validate_csv_schema(df)
    if not is_valid:
        return JSONResponse(
            status_code=400,
            content={"error": f"Missing columns: {', '.join(missing_columns)}"}
        )

    # Prepare features and predict
    try:
        df_features = prepare_features(df.copy())
        results = predict_fraud(MODEL, df_features)
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": "Model inference failed", "details": str(e)}
        )
    
    # Filter and sort flagged transactions (use threshold parameter)
    flagged = results[results["fraud_score"] >= threshold].sort_values("fraud_score", ascending=False)
    top_flagged = flagged.head(top_k)

    if len(top_flagged) == 0:
        return JSONResponse(content={
            "flagged_count": 0,
            "results": [],
            "message": "No transactions flagged above threshold"
        })

    # Get OpenAI client and generate explanations
    try:
        client = get_openai_client()
        explanations, risk_levels = explain_transactions_parallel(client, top_flagged, max_workers=parallel)
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": "OpenAI failed", "details": str(e)}
        )

    # Combine results
    combined = []
    for idx, (row_idx, row) in enumerate(top_flagged.iterrows()):
        txn_dict = row.to_dict()
        explanation = explanations[idx] if idx < len(explanations) else None
        risk_level = risk_levels[idx] if idx < len(risk_levels) else get_risk_level(row.get('fraud_score', 0.0))
        
        combined.append({
            "transaction_id": int(row_idx),
            "fraud_score": float(txn_dict.get("fraud_score", 0.0)),
            "risk_level": risk_level,
            "explanation": explanation,
            "transaction": txn_dict
        })

    return JSONResponse(content={
        "flagged_count": len(flagged),
        "results": combined
    })


class PublishRequest(BaseModel):
    """Request model for blockchain publish endpoint."""
    sha256: str


@app.post("/blockchain/publish")
async def publish_to_blockchain(
    request: PublishRequest,
    api_key: str = Depends(verify_api_key)
):
    """
    Publish SHA-256 hash to Aptos blockchain.
    
    Input:
    {
        "sha256": "aabbccddeeff..."
    }
    
    Output:
    {
        "sha256": "...",
        "aptos_tx": "<transaction_hash>",
        "aptos_explorer_url": "https://explorer.aptoslabs.com/txn/<hash>?network=testnet"
    }
    """
    try:
        # Validate hash format
        if len(request.sha256) != 64:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid hash length. Expected 64 characters (SHA-256), got {len(request.sha256)}"
            )
        
        # Validate hex format
        try:
            bytes.fromhex(request.sha256)
        except ValueError:
            raise HTTPException(
                status_code=400,
                detail="Invalid hex string format"
            )
        
        # Publish hash on Aptos
        aptos_tx = publish_hash_on_aptos(request.sha256)
        aptos_explorer_url = get_explorer_url(aptos_tx, network="testnet")
        
        return {
            "sha256": request.sha256,
            "aptos_tx": aptos_tx,
            "aptos_explorer_url": aptos_explorer_url
        }
    except HTTPException:
        raise
    except ValueError as e:
        # ValueError usually means account not found or insufficient balance
        error_msg = str(e)
        if "not found" in error_msg.lower() or "insufficient balance" in error_msg.lower() or "faucet" in error_msg.lower():
            return JSONResponse(
                status_code=400,
                content={
                    "error": "Aptos account issue",
                    "details": error_msg
                }
            )
        else:
            raise HTTPException(status_code=400, detail=error_msg)
    except Exception as e:
        error_msg = str(e)
        # Check if it's a known error type
        if "INSUFFICIENT_BALANCE" in error_msg or "balance" in error_msg.lower():
            return JSONResponse(
                status_code=400,
                content={
                    "error": "Insufficient balance",
                    "details": error_msg + "\n\nPlease fund your account using the Aptos testnet faucet."
                }
            )
        else:
            return JSONResponse(
                status_code=500,
                content={
                    "error": "Aptos error",
                    "details": error_msg
                }
            )


@app.post("/report")
async def full_report(
    file: UploadFile = File(...), 
    top_k: int = 10, 
    threshold: float = 0.9, 
    parallel: int = 5,
    api_key: str = Depends(verify_api_key)
):
    """
    Full fraud report: detection + explanation + risk scoring.
    Returns comprehensive report matching the format from fraud_explainer.py
    Includes SHA-256 hash and Aptos blockchain anchoring.
    """
    if MODEL is None:
        raise HTTPException(status_code=500, detail="Model not loaded. Please check server logs.")
    
    # Check for OpenAI API key
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return JSONResponse(
            status_code=500,
            content={"error": "OpenAI failed", "details": "Missing OpenAI API key in .env"}
        )

    # More lenient content type check
    if file.content_type and file.content_type not in ["text/csv", "application/vnd.ms-excel", "text/plain", "application/csv"]:
        if not (file.filename and file.filename.lower().endswith('.csv')):
            raise HTTPException(status_code=400, detail="Upload a valid CSV file. Expected .csv file.")

    try:
        content = await file.read()
        if not content or len(content) == 0:
            raise HTTPException(status_code=400, detail="Uploaded file is empty.")
        
        # Check file size limit
        file_size = len(content)
        if file_size > MAX_FILE_SIZE:
            raise HTTPException(
                status_code=413,
                detail=f"File too large. Maximum size is {MAX_FILE_SIZE / (1024 * 1024):.1f}MB, received {file_size / (1024 * 1024):.2f}MB"
            )
        
        df = pd.read_csv(io.BytesIO(content))
        if df.empty:
            raise HTTPException(status_code=400, detail="CSV file is empty or has no data.")
    except HTTPException:
        raise  # Re-raise HTTPException (e.g., file size errors)
    except pd.errors.EmptyDataError:
        raise HTTPException(status_code=400, detail="CSV file is empty or has no data.")
    except pd.errors.ParserError as e:
        raise HTTPException(status_code=400, detail=f"Error parsing CSV file: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error loading CSV: {str(e)}")

    # Validate CSV schema
    is_valid, missing_columns = validate_csv_schema(df)
    if not is_valid:
        return JSONResponse(
            status_code=400,
            content={"error": f"Missing columns: {', '.join(missing_columns)}"}
        )

    # Prepare features and predict
    try:
        df_features = prepare_features(df.copy())
        results = predict_fraud(MODEL, df_features)
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": "Model inference failed", "details": str(e)}
        )
    
    # Filter and sort flagged transactions (use threshold parameter)
    flagged = results[results["fraud_score"] >= threshold].sort_values("fraud_score", ascending=False)
    top_flagged = flagged.head(top_k)

    if len(top_flagged) == 0:
        return {
            "flagged_count": 0,
            "final_report": [],
            "message": "No transactions flagged above threshold"
        }

    # Get OpenAI client and generate explanations
    try:
        client = get_openai_client()
        explanations, risk_levels = explain_transactions_parallel(client, top_flagged, max_workers=parallel)
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": "OpenAI failed", "details": str(e)}
        )

    # Create full report (matching fraud_explainer.py format)
    final_report = []
    for idx, (row_idx, row) in enumerate(top_flagged.iterrows()):
        txn_dict = row.to_dict()
        fraud_score = txn_dict.get("fraud_score", 0.0)
        explanation = explanations[idx] if idx < len(explanations) else None
        risk_level = risk_levels[idx] if idx < len(risk_levels) else get_risk_level(fraud_score)
        
        final_report.append({
            "transaction_id": int(row_idx),
            "fraud_score": float(fraud_score),
            "risk_level": risk_level,
            "explanation": explanation,
            "raw_transaction": txn_dict
        })

    # Create report dictionary for hashing
    report_dict = {
        "flagged_count": len(flagged),
        "final_report": final_report
    }

    # Compute SHA-256 hash of the report
    sha256_hash = compute_sha256(report_dict)

    # Try to publish hash on Aptos blockchain
    aptos_tx = None
    aptos_explorer_url = None
    aptos_error = None

    try:
        aptos_tx = publish_hash_on_aptos(sha256_hash)
        aptos_explorer_url = get_explorer_url(aptos_tx, network="testnet")
    except ValueError as e:
        # ValueError usually means account not found or insufficient balance
        error_msg = str(e)
        aptos_error = {
            "error": "Aptos account issue",
            "details": error_msg,
            "action_required": "Please fund your account using the Aptos testnet faucet and try again."
        }
    except Exception as e:
        # If Aptos publishing fails, still return the hash so frontend can retry
        error_msg = str(e)
        if "INSUFFICIENT_BALANCE" in error_msg or "balance" in error_msg.lower():
            aptos_error = {
                "error": "Insufficient balance",
                "details": error_msg,
                "action_required": "Please fund your account using the Aptos testnet faucet."
            }
        else:
            aptos_error = {
                "error": "Aptos error",
                "details": error_msg
            }

    # Build response
    response = {
        "flagged_count": len(flagged),
        "final_report": final_report,
        "sha256": sha256_hash
    }

    # Add Aptos transaction info if successful
    if aptos_tx:
        response["aptos_tx"] = aptos_tx
        response["aptos_explorer_url"] = aptos_explorer_url
    elif aptos_error:
        response.update(aptos_error)

    return response


class VerifyRequest(BaseModel):
    """Request model for verify endpoint."""
    report: dict


@app.post("/verify")
async def verify(
    request: VerifyRequest,
    api_key: str = Depends(verify_api_key)
):
    """
    Verify a fraud report by checking if its hash exists on Aptos blockchain.
    
    Input:
    {
        "report": { ... full json report ... }
    }
    
    Output:
    {
        "status": "MATCH" | "MISMATCH" | "NOT_FOUND",
        "sha256": "<computed>",
        "aptos_search_url": "https://explorer.aptoslabs.com/account/<ADDRESS>?network=testnet"
    }
    """
    try:
        # Step 1: Canonicalize JSON
        canonical_str = canonical_json(request.report)
        
        # Step 2: Compute SHA-256
        computed_hash = compute_sha256(request.report)
        
        # Step 3: Search for hash on Aptos blockchain
        try:
            status, account_address = search_hash_on_aptos(computed_hash)
            
            # Build explorer URL
            aptos_search_url = None
            if account_address:
                aptos_search_url = get_account_explorer_url(account_address, network="testnet")
            else:
                # If no account, use a placeholder or check if we can get account
                aptos_account = get_aptos_account()
                if aptos_account:
                    aptos_search_url = get_account_explorer_url(aptos_account.address().hex(), network="testnet")
            
            # For now, since we can't directly verify the hash in transaction data,
            # we return NOT_FOUND. In a production system, you would check a hash registry.
            # If the hash was previously published and stored in a database, you could
            # return MATCH here.
            
            return {
                "status": status,
                "sha256": computed_hash,
                "aptos_search_url": aptos_search_url or "https://explorer.aptoslabs.com/?network=testnet"
            }
            
        except Exception as e:
            return JSONResponse(
                status_code=500,
                content={
                    "error": "Aptos error",
                    "details": str(e)
                }
            )
            
    except Exception as e:
        return JSONResponse(
            status_code=400,
            content={
                "error": "Verification failed",
                "details": str(e)
            }
        )


if __name__ == "__main__":
    import uvicorn
    # Use PORT environment variable for Render, fallback to 10000 for local development
    port = int(os.getenv("PORT", 10000))
    uvicorn.run(app, host="0.0.0.0", port=port)
