from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
import pandas as pd
import io
from fraud_explainer import (
    load_model,
    prepare_features,
    predict_fraud,
    get_risk_level,
    initialize_openai_client,
    explain_transactions_parallel
)
import os
from dotenv import load_dotenv

load_dotenv()

app = FastAPI(title="HAWKEYE Fraud Detection API")

# Maximum file size: 150MB
MAX_FILE_SIZE = 150 * 1024 * 1024  # 150MB in bytes

# Load model at startup - fail fast if model cannot be loaded
try:
    MODEL = load_model("xgb_model.pkl")
    print("Model loaded successfully")
except Exception as e:
    error_msg = f"CRITICAL: Failed to load model: {e}"
    print(error_msg)
    raise RuntimeError(error_msg) from e

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
def health():
    """
    Health check endpoint to verify model is loaded and API is ready.
    """
    if MODEL is None:
        return JSONResponse(
            status_code=503,
            content={
                "status": "unhealthy",
                "model_loaded": False,
                "message": "Model not loaded. Server may be starting or model file is missing."
            }
        )
    return {
        "status": "healthy",
        "model_loaded": True,
        "message": "API is ready to process requests"
    }


@app.post("/detect")
async def detect(file: UploadFile = File(...), top_k: int = 10, threshold: float = 0.9):
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

    # Prepare features (remove Class column if present)
    try:
        df_features = prepare_features(df.copy())
        results = predict_fraud(MODEL, df_features)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error processing CSV data: {str(e)}. Please ensure your CSV has the required columns (V1-V28, Time, Amount).")
    
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
async def explain(file: UploadFile = File(...), top_k: int = 5, threshold: float = 0.9, parallel: int = 5):
    """
    Detect fraud + Generate GPT explanations for flagged transactions.
    Uses parallel processing for faster explanation generation.
    """
    if MODEL is None:
        raise HTTPException(status_code=500, detail="Model not loaded. Please check server logs.")
    
    # Check for OpenAI API key
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise HTTPException(status_code=500, detail="Missing OpenAI API key in .env")

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

    # Prepare features and predict
    try:
        df_features = prepare_features(df.copy())
        results = predict_fraud(MODEL, df_features)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error processing CSV data: {str(e)}. Please ensure your CSV has the required columns (V1-V28, Time, Amount).")
    
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
    client = get_openai_client()
    explanations, risk_levels = explain_transactions_parallel(client, top_flagged, max_workers=parallel)

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


@app.post("/report")
async def full_report(file: UploadFile = File(...), top_k: int = 10, threshold: float = 0.9, parallel: int = 5):
    """
    Full fraud report: detection + explanation + risk scoring.
    Returns comprehensive report matching the format from fraud_explainer.py
    """
    if MODEL is None:
        raise HTTPException(status_code=500, detail="Model not loaded. Please check server logs.")
    
    # Check for OpenAI API key
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise HTTPException(status_code=500, detail="Missing OpenAI API key in .env")

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

    # Prepare features and predict
    try:
        df_features = prepare_features(df.copy())
        results = predict_fraud(MODEL, df_features)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error processing CSV data: {str(e)}. Please ensure your CSV has the required columns (V1-V28, Time, Amount).")
    
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
    client = get_openai_client()
    explanations, risk_levels = explain_transactions_parallel(client, top_flagged, max_workers=parallel)

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

    return {
        "flagged_count": len(flagged),
        "final_report": final_report
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
