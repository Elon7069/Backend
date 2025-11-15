# Render Deployment Guide

## 🔧 Prerequisites

Before deploying to Render, ensure you have:

1. ✅ **GitHub Repository**: Your code must be pushed to GitHub
2. ✅ **Model File**: `xgb_model.pkl` exists in the `Backend` directory and is committed to git
3. ✅ **Missing Files Fixed**: The `auth.py` file has been created (now included)
4. ✅ **Render Account**: Sign up at [render.com](https://render.com)

## 🚀 Deployment Steps

### Step 1: Connect Repository to Render

1. Go to [Render Dashboard](https://dashboard.render.com/)
2. Click **New +** → **Web Service**
3. Connect your GitHub account and select your repository
4. **Important**: Set **Root Directory** to `Backend`
   - This tells Render where to find your code

### Step 2: Configure Service

Render should auto-detect your `render.yaml` file. Verify these settings:

- **Name**: `hawkeye-backend` (or choose your own)
- **Runtime**: Python
- **Build Command**: `pip install -r requirements.txt`
- **Start Command**: `uvicorn app:app --host 0.0.0.0 --port $PORT`
- **Python Version**: 3.11.0

### Step 3: Set Environment Variables

In Render Dashboard → Environment, add these variables:

**Required:**
- `BACKEND_API_KEY`: Create a secure API key (e.g., `hawk_secret_12345`)
- `OPENAI_API_KEY`: Your OpenAI API key from [platform.openai.com](https://platform.openai.com)

**Optional (for blockchain features):**
- `APTOS_PRIVATE_KEY`: Your Aptos private key (without 0x prefix)
- `APTOS_NETWORK`: Set to `testnet` (default) or `mainnet`

**Note**: Do NOT set `PORT` - Render sets this automatically!

### Step 4: Deploy

1. Click **Create Web Service**
2. Render will:
   - Clone your repository
   - Install dependencies from `requirements.txt`
   - Start your FastAPI app with uvicorn
3. Wait for build to complete (2-5 minutes)
4. Your service will be live at `https://your-service-name.onrender.com`

### Step 5: Verify Deployment

Test your deployment:

```bash
# Health check
curl https://your-service-name.onrender.com/health

# Test detect endpoint (requires API key)
curl -X POST https://your-service-name.onrender.com/detect \
  -H "x-api-key: YOUR_BACKEND_API_KEY" \
  -F "file=@creditcard.csv"
```

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
