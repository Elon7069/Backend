# HAWKEYE - Fraud Detection Backend

A machine learning-powered fraud detection system that uses XGBoost to identify suspicious credit card transactions and OpenAI to generate detailed explanations.

## 🚀 Features

- **Fraud Detection**: XGBoost model for high-accuracy fraud prediction
- **AI-Powered Explanations**: OpenAI GPT integration for detailed fraud analysis
- **Risk Scoring**: Automatic risk level classification (Low/Medium/High/Critical)
- **Performance Optimized**: Parallel processing and CSV sampling for fast analysis
- **JSON Export**: Detailed fraud reports with transaction data and explanations
- **Secure**: Environment-based API key management

## 📋 Prerequisites

- Python 3.8 or higher
- OpenAI API key ([Get one here](https://platform.openai.com/api-keys))
- Trained XGBoost model (`xgb_model.pkl`)

## 🔧 Installation

1. **Clone the repository:**
   ```bash
   git clone <repository-url>
   cd HAWKEYE/Backend
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up environment variables:**
   ```bash
   # Copy the example file
   cp .env.example .env
   
   # Edit .env and add your OpenAI API key
   # OPENAI_API_KEY=sk-your-actual-key-here
   ```
   
   **Important:** Never commit your `.env` file! It's already in `.gitignore`.

## 🎯 Usage

### Basic Usage

```bash
# Analyze default creditcard.csv file
python fraud_explainer.py

# Use a different CSV file
python fraud_explainer.py your_data.csv

# Fast mode: Sample 5000 rows for quick analysis
python fraud_explainer.py --sample 5000

# Custom output file
python fraud_explainer.py --output my_report.json
```

### Advanced Options

```bash
# Full control with all options
python fraud_explainer.py \
  --csv data/transactions.csv \
  --sample 10000 \
  --parallel 10 \
  --top-n 10 \
  --output fraud_report.json \
  --model xgb_model.pkl
```

### Command-Line Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `csv_path` | Path to CSV file with transaction data | `creditcard.csv` |
| `--csv` | Alternative way to specify CSV file | - |
| `--model` | Path to trained XGBoost model | `xgb_model.pkl` |
| `--sample` | Number of rows to sample (for faster processing) | All rows |
| `--parallel` | Number of parallel OpenAI API calls | `5` |
| `--top-n` | Number of top flagged transactions to explain | `5` |
| `--output` | Output JSON file path | `fraud_report.json` |

## 🌐 API Usage

### Starting the API Server

```bash
# Run the FastAPI server
uvicorn app:app --reload

# Or with custom host/port
uvicorn app:app --host 0.0.0.0 --port 8000

# Or run directly
python app.py
```

The API will be available at:
- **API Base URL**: `http://localhost:8000`
- **Interactive Docs**: `http://localhost:8000/docs` (Swagger UI)
- **Alternative Docs**: `http://localhost:8000/redoc` (ReDoc)

### API Endpoints

#### 1. Health Check
```http
GET /
```

**Response:**
```json
{
  "status": "running",
  "message": "HAWKEYE Fraud Detection API is live!"
}
```

#### 2. Health Status
```http
GET /health
```

**Description**: Check if the API is healthy and the model is loaded.

**Response (Healthy):**
```json
{
  "status": "healthy",
  "model_loaded": true,
  "message": "API is ready to process requests"
}
```

**Response (Unhealthy):**
```json
{
  "status": "unhealthy",
  "model_loaded": false,
  "message": "Model not loaded. Server may be starting or model file is missing."
}
```

#### 3. Detect Fraud (Fast)
```http
POST /detect
```

**Description**: Fast fraud detection without AI explanations. Returns fraud scores and risk levels.

**Request:**
- **Content-Type**: `multipart/form-data`
- **Body**: 
  - `file` (file, required): CSV file with transaction data
  - `top_k` (int, optional): Number of top suspicious transactions to return (default: 10)
  - `threshold` (float, optional): Fraud score threshold (default: 0.9)

**Example using cURL:**
```bash
curl -X POST "http://localhost:8000/detect" \
  -F "file=@transactions.csv" \
  -F "top_k=10" \
  -F "threshold=0.9"
```

**Response:**
```json
{
  "flagged_count": 15,
  "top_suspicious": [
    {
      "Time": 0,
      "Amount": 149.62,
      "V1": -1.359807,
      "fraud_score": 0.9542,
      "fraud_flag": true,
      "risk_level": "Critical",
      ...
    }
  ]
}
```

#### 4. Explain Fraud (With AI)
```http
POST /explain
```

**Description**: Detect fraud and generate AI-powered explanations for flagged transactions.

**Request:**
- **Content-Type**: `multipart/form-data`
- **Body**: 
  - `file` (file, required): CSV file with transaction data
  - `top_k` (int, optional): Number of transactions to explain (default: 5)
  - `threshold` (float, optional): Fraud score threshold (default: 0.9)
  - `parallel` (int, optional): Number of parallel OpenAI API calls (default: 5)

**Example using cURL:**
```bash
curl -X POST "http://localhost:8000/explain" \
  -F "file=@transactions.csv" \
  -F "top_k=5" \
  -F "threshold=0.9" \
  -F "parallel=5"
```

**Response:**
```json
{
  "flagged_count": 15,
  "results": [
    {
      "transaction_id": 12345,
      "fraud_score": 0.9542,
      "risk_level": "Critical",
      "explanation": "1. **Summary Risk Rating: Critical**\n2. **Fraud Indicators:**\n   - Unusually high PCA component values...",
      "transaction": {
        "Time": 0,
        "Amount": 149.62,
        ...
      }
    }
  ]
}
```

#### 5. Full Report
```http
POST /report
```

**Description**: Complete fraud analysis report with detection, explanations, and risk scoring.

**Request:**
- **Content-Type**: `multipart/form-data`
- **Body**: 
  - `file` (file, required): CSV file with transaction data
  - `top_k` (int, optional): Number of transactions in report (default: 10)
  - `threshold` (float, optional): Fraud score threshold (default: 0.9)
  - `parallel` (int, optional): Number of parallel OpenAI API calls (default: 5)

**Example using cURL:**
```bash
curl -X POST "http://localhost:8000/report" \
  -F "file=@transactions.csv" \
  -F "top_k=10" \
  -F "threshold=0.9" \
  -F "parallel=5"
```

**Response:**
```json
{
  "flagged_count": 15,
  "final_report": [
    {
      "transaction_id": 12345,
      "fraud_score": 0.9542,
      "risk_level": "Critical",
      "explanation": "1. **Summary Risk Rating: Critical**\n...",
      "raw_transaction": {
        "Time": 0,
        "Amount": 149.62,
        ...
      }
    }
  ]
}
```

### API Error Responses

**400 Bad Request:**
```json
{
  "detail": "Upload a valid CSV file."
}
```

**413 Payload Too Large:**
```json
{
  "detail": "File too large. Maximum size is 150.0MB, received 200.0MB"
}
```

**500 Internal Server Error:**
```json
{
  "detail": "Model not loaded. Please check server logs."
}
```

```json
{
  "detail": "Missing OpenAI API key in .env"
}
```

### Using Python Requests

```python
import requests

# Detect fraud
with open('transactions.csv', 'rb') as f:
    response = requests.post(
        'http://localhost:8000/detect',
        files={'file': f},
        data={'top_k': 10, 'threshold': 0.9}
    )
    print(response.json())

# Get explanations
with open('transactions.csv', 'rb') as f:
    response = requests.post(
        'http://localhost:8000/explain',
        files={'file': f},
        data={'top_k': 5, 'parallel': 5}
    )
    print(response.json())
```

### CORS Configuration

CORS middleware is included but commented out by default. To enable for frontend integration:

1. Uncomment the CORS middleware in `app.py`
2. Update `allow_origins` with your frontend URL(s)

```python
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # Your frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

## 📊 Input Data Format

Your CSV file should have the same structure as the training data:

- **Required columns**: `Time`, `Amount`, `V1` through `V28` (PCA features)
- **Optional column**: `Class` (will be automatically removed if present)
- **Maximum file size**: 150MB

Example:
```csv
Time,Amount,V1,V2,V3,...,V28,Class
0,149.62,-1.359807,-0.072781,2.536347,...,0.133558,0
```

**Note**: The API will automatically validate your CSV format. If columns are missing, you'll receive a clear error message.

## 📁 Output

The script generates:

1. **Console Output**: Formatted fraud explanations with risk levels
2. **JSON Report** (`fraud_report.json`): Detailed analysis including:
   - Transaction ID
   - Fraud score (0.0 to 1.0)
   - Risk level (Low/Medium/High/Critical)
   - AI-generated explanation
   - Complete transaction data

### Example JSON Output

```json
[
  {
    "transaction_id": 12345,
    "fraud_score": 0.9542,
    "risk_level": "Critical",
    "explanation": "1. **Summary Risk Rating: Critical**\n2. **Fraud Indicators:**\n   - Unusually high PCA component values...",
    "raw_transaction": {
      "Time": 0,
      "Amount": 149.62,
      "V1": -1.359807,
      ...
    }
  }
]
```

## 🔒 Security

**IMPORTANT**: Never commit your `.env` file to version control!

### Security Features

- ✅ `.env` is in `.gitignore` (your API key is safe)
- ✅ `.env.example` is a template (safe to commit)
- ✅ No hardcoded API keys in source code
- ✅ File size limits (150MB max) to prevent DoS attacks
- ✅ Model validation at startup (fail-fast if model is missing)

### Before Pushing to Git

1. **Verify `.env` is ignored:**
   ```bash
   git status
   # .env should NOT appear in the list
   ```

2. **Check for sensitive data:**
   ```bash
   # Search for potential API keys (should return no results)
   grep -r "sk-[a-zA-Z0-9]\{20,\}" .
   ```

3. **Ensure test files are ignored:**
   - `test_api.py` is in `.gitignore`
   - Data files (`.csv`) are in `.gitignore`
   - Output files (`fraud_report.json`) are in `.gitignore`

## ⚡ Performance Tips

1. **Use Sampling**: For large datasets, use `--sample` to process fewer rows
   ```bash
   python fraud_explainer.py --sample 5000  # Much faster!
   ```

2. **Increase Parallel Requests**: Speed up OpenAI API calls
   ```bash
   python fraud_explainer.py --parallel 10  # More concurrent requests
   ```

3. **Expected Performance**:
   - Full dataset (284k rows): ~5-10 seconds
   - Sampled (5k rows): ~1-2 seconds
   - With parallel API calls: 2-3x faster explanations

## 🛠️ Project Structure

```
Backend/
├── app.py                  # FastAPI application (REST API)
├── fraud_explainer.py      # CLI fraud detection script
├── xgb_model.pkl           # Trained XGBoost model (gitignored if large)
├── requirements.txt        # Python dependencies
├── .env.example            # Environment template (safe to commit)
├── .env                    # Your actual API keys (gitignored - NEVER commit!)
├── .gitignore              # Git ignore rules
└── README.md               # This file
```

### Files NOT in Git (via .gitignore)
- `.env` - Your API keys (sensitive)
- `*.csv` - Data files (may be large/sensitive)
- `fraud_report.json` - Output files
- `test_api.py` - Test files
- `__pycache__/` - Python cache
- `*.pkl` - Model files (if large)

## 📝 Dependencies

### Core Dependencies
- `pandas>=2.0.0` - Data processing
- `numpy>=1.24.0` - Numerical computing
- `scikit-learn>=1.3.0` - Machine learning utilities
- `xgboost>=2.0.0` - XGBoost model
- `joblib>=1.3.0` - Model serialization

### API Framework
- `fastapi>=0.104.0` - Modern web framework
- `uvicorn[standard]>=0.24.0` - ASGI server

### External Services
- `openai>=1.0.0` - OpenAI API client
- `python-dotenv>=1.0.0` - Environment variable management

## 🐛 Troubleshooting

### "OPENAI_API_KEY not found"
- Make sure `.env` file exists in the Backend directory
- Verify the file contains: `OPENAI_API_KEY=sk-your-key-here`
- Check that the key starts with `sk-`

### "Model file not found"
- Ensure `xgb_model.pkl` is in the Backend directory
- Or specify path with `--model path/to/model.pkl`

### "CSV file not found"
- Check the file path is correct
- Use absolute path if needed: `python fraud_explainer.py C:/path/to/file.csv`

### Slow Performance
- Use `--sample` to reduce dataset size
- Increase `--parallel` for faster API calls
- Consider processing smaller batches

### File Size Errors
- Maximum file size is 150MB
- If you need to process larger files, split them into smaller chunks
- Consider using the CLI tool (`fraud_explainer.py`) for very large files

## 📄 License

[Add your license information here]

## 🤝 Contributing

[Add contribution guidelines if applicable]

## 📧 Support

[Add support/contact information if needed]

---

**Note**: This is a fraud detection system. Always verify results and use appropriate thresholds for your use case.

