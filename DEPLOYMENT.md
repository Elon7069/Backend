# Render Deployment Guide

## Prerequisites

1. **Model File**: Ensure `xgb_model.pkl` is included in your repository or uploaded to Render
   - If your `.gitignore` excludes `.pkl` files, you may need to temporarily allow it or use a different deployment method
   - Alternative: Upload the model file directly to Render's file system or use cloud storage

2. **Environment Variables**: Set the following in Render Dashboard → Environment:
   - `BACKEND_API_KEY`: Your API key for authentication
   - `OPENAI_API_KEY`: Your OpenAI API key (required for `/explain` and `/report` endpoints)
   - `PORT`: Automatically set by Render (don't override)
   - Aptos-related variables (if using blockchain features):
     - `APTOS_PRIVATE_KEY`: Your Aptos private key
     - `APTOS_NETWORK`: Network (default: testnet)

## Deployment Steps

### Option 1: Using render.yaml (Recommended)

1. Connect your GitHub repository to Render
2. Render will automatically detect `render.yaml` in the `Backend` directory
3. Configure environment variables in Render Dashboard
4. Deploy!

### Option 2: Manual Configuration

1. **Service Type**: Web Service
2. **Build Command**: `pip install -r requirements.txt`
3. **Start Command**: `uvicorn app:app --host 0.0.0.0 --port $PORT`
4. **Root Directory**: `Backend` (if deploying from repo root)
5. **Python Version**: 3.11.0 (or compatible)

## Important Notes

- **CORS**: Currently configured to allow all origins (`*`). For production, update `app.py` to specify your frontend domain(s)
- **File Size Limit**: Maximum upload size is 150MB
- **Health Check**: Use `/health` endpoint for Render health checks
- **Model Loading**: The model loads at startup. If it fails, the service won't start (fail-fast behavior)

## Testing After Deployment

1. Check health: `https://your-service.onrender.com/health`
2. Test API key authentication: Include `x-api-key` header in requests
3. Verify CORS: Test from your frontend application

## Troubleshooting

- **Model not found**: Ensure `xgb_model.pkl` is in the `Backend` directory
- **Port errors**: Render sets `PORT` automatically - don't override it
- **API key errors**: Verify environment variables are set correctly in Render Dashboard
- **Import errors**: Check that all dependencies in `requirements.txt` are correct

