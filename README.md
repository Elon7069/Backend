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

1. **Clone the repository** (if not already done)

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

## 📊 Input Data Format

Your CSV file should have the same structure as the training data:

- **Required columns**: `Time`, `Amount`, `V1` through `V28` (PCA features)
- **Optional column**: `Class` (will be removed if present)

Example:
```csv
Time,Amount,V1,V2,V3,...,V28,Class
0,149.62,-1.359807,-0.072781,2.536347,...,0.133558,0
```

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

- ✅ `.env` is in `.gitignore` (your API key is safe)
- ✅ `.env.example` is a template (safe to commit)
- ✅ No hardcoded API keys in source code

### Pre-Push Security Check

Run the security check before pushing:

```bash
python check_security.py
```

This verifies:
- `.env` is properly ignored
- No hardcoded API keys in code
- `.env` is not tracked by Git

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
├── fraud_explainer.py      # Main fraud detection script
├── xgb_model.pkl           # Trained XGBoost model
├── requirements.txt        # Python dependencies
├── .env.example            # Environment template
├── .gitignore              # Git ignore rules
├── check_security.py       # Security verification script
└── README.md               # This file
```

## 📝 Dependencies

- `openai>=1.0.0` - OpenAI API client
- `python-dotenv>=1.0.0` - Environment variable management
- `joblib>=1.3.0` - Model loading
- `pandas>=2.0.0` - Data processing

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

## 📄 License

[Add your license information here]

## 🤝 Contributing

[Add contribution guidelines if applicable]

## 📧 Support

[Add support/contact information if needed]

---

**Note**: This is a fraud detection system. Always verify results and use appropriate thresholds for your use case.

