"""
Fraud Detection Explainer with OpenAI Integration.
Loads trained XGBoost model, predicts fraud probabilities, and uses OpenAI
to generate explanations for flagged transactions.
"""

import argparse
import os
import sys
import json
from typing import Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
import time

import joblib
import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI, APIError, APITimeoutError


def load_model(model_path: str = 'xgb_model.pkl'):
    """Load the trained XGBoost model from a pickle file."""
    try:
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file '{model_path}' not found.")
        model = joblib.load(model_path)
        print(f"[OK] Successfully loaded model from '{model_path}'")
        return model
    except Exception as e:
        print(f"Error loading model: {e}", file=sys.stderr)
        sys.exit(1)


def load_data(csv_path: str = 'creditcard.csv', sample_size: Optional[int] = None, random_state: int = 42):
    """Load the credit card dataset from CSV file.
    
    Args:
        csv_path: Path to CSV file
        sample_size: If provided, randomly sample this many rows for faster processing
        random_state: Random seed for reproducible sampling
    """
    try:
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"CSV file '{csv_path}' not found.")
        
        # If sample_size is specified, sample rows for faster processing
        if sample_size:
            # For large files, read all and sample (memory efficient for very large files)
            # For smaller files, this is still fast
            df_full = pd.read_csv(csv_path)
            if len(df_full) > sample_size:
                df = df_full.sample(n=sample_size, random_state=random_state).reset_index(drop=True)
            else:
                df = df_full
        else:
            df = pd.read_csv(csv_path)
        
        print(f"[OK] Successfully loaded data from '{csv_path}'")
        print(f"  Dataset shape: {df.shape}")
        if sample_size and len(df) < sample_size:
            print(f"  Note: Requested {sample_size} rows but only {len(df)} available")
        return df
    except Exception as e:
        print(f"Error loading data: {e}", file=sys.stderr)
        sys.exit(1)


def prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    """Prepare features by removing the target column if it exists."""
    if 'Class' in df.columns:
        df = df.drop('Class', axis=1)
        print("[OK] Removed 'Class' column from dataset")
    return df


def get_risk_level(score: float) -> str:
    """Get risk level based on fraud score.
    
    Args:
        score: Fraud probability score (0.0 to 1.0)
    
    Returns:
        Risk level string: 'Low', 'Medium', 'High', or 'Critical'
    """
    if score < 0.20:
        return "Low"
    elif score < 0.60:
        return "Medium"
    elif score < 0.90:
        return "High"
    else:
        return "Critical"


def predict_fraud(model, df: pd.DataFrame) -> pd.DataFrame:
    """Predict fraud probabilities and add fraud_score and fraud_flag columns."""
    try:
        # Get predicted probabilities
        # predict_proba returns probabilities for all classes
        # For binary classification, we want the probability of class 1 (fraud)
        probabilities = model.predict_proba(df)
        
        # Extract fraud probability (class 1, second column)
        # If binary classification, probabilities shape is (n_samples, 2)
        if probabilities.shape[1] == 2:
            fraud_scores = probabilities[:, 1]
        else:
            # If multi-class, use the maximum probability
            fraud_scores = probabilities.max(axis=1)
        
        # Add fraud_score column
        df = df.copy()
        df['fraud_score'] = fraud_scores
        
        # Add fraud_flag column (True if fraud_score > 0.90)
        df['fraud_flag'] = df['fraud_score'] > 0.90
        
        flagged_count = df['fraud_flag'].sum()
        print(f"[OK] Predictions completed. Found {flagged_count} flagged transactions (fraud_score > 0.90)")
        return df
    except Exception as e:
        print(f"Error during prediction: {e}", file=sys.stderr)
        sys.exit(1)


def initialize_openai_client() -> OpenAI:
    """Initialize OpenAI client with API key from environment variables."""
    try:
        # Get the directory where this script is located
        script_dir = os.path.dirname(os.path.abspath(__file__))
        
        # Try multiple locations for .env file (case-insensitive on Windows)
        env_paths = [
            os.path.join(script_dir, '.env'),  # Same directory as script
            os.path.join(os.path.dirname(script_dir), '.env'),  # Parent directory
            os.path.join(os.getcwd(), '.env'),  # Current working directory
        ]
        
        # Also check case variations for Windows
        script_dir_variants = [
            script_dir,
            script_dir.replace('Backend', 'backend'),
            script_dir.replace('backend', 'Backend'),
        ]
        for variant_dir in script_dir_variants:
            variant_path = os.path.join(variant_dir, '.env')
            if variant_path not in env_paths and os.path.exists(variant_path):
                env_paths.append(variant_path)
        
        env_file_found = None
        api_key = None
        
        # First, try to load from .env file using python-dotenv
        for env_path in env_paths:
            if os.path.exists(env_path):
                env_file_found = env_path
                # Load with override=True to ensure it loads
                load_dotenv(env_path, override=True)
                print(f"[OK] Loading .env file from: {env_path}")
                
                # Try to read the file directly as backup
                try:
                    with open(env_path, 'r', encoding='utf-8', errors='ignore') as f:
                        lines = f.readlines()
                        for line in lines:
                            line = line.strip()
                            # Skip comments and empty lines
                            if not line or line.startswith('#'):
                                continue
                            # Parse key=value format
                            if '=' in line:
                                key, value = line.split('=', 1)
                                key = key.strip()
                                value = value.strip()
                                # Remove quotes if present
                                if value.startswith('"') and value.endswith('"'):
                                    value = value[1:-1]
                                elif value.startswith("'") and value.endswith("'"):
                                    value = value[1:-1]
                                
                                if key == 'OPENAI_API_KEY' and value:
                                    # Set in environment directly
                                    os.environ['OPENAI_API_KEY'] = value
                                    api_key = value
                                    print(f"[OK] Found API key in .env file (length: {len(value)})")
                                    break
                except Exception as e:
                    print(f"[WARNING] Could not read .env file directly: {e}")
                
                break
        
        # If no .env file found, try default load_dotenv() behavior
        if not env_file_found:
            load_dotenv()  # This will look in current directory and parent directories
            # Check common locations after load_dotenv()
            for env_path in env_paths:
                if os.path.exists(env_path):
                    env_file_found = env_path
                    break
        
        # Get API key from environment (from .env file or already set)
        if not api_key:
            api_key = os.getenv("OPENAI_API_KEY")
        
        # If still no key, provide detailed error message
        if not api_key:
            error_msg = f"OPENAI_API_KEY not found in environment variables.\n\n"
            
            if env_file_found:
                # Read file content to show what's actually in it
                try:
                    with open(env_file_found, 'r', encoding='utf-8', errors='ignore') as f:
                        content = f.read()
                        file_size = os.path.getsize(env_file_found)
                        
                    error_msg += f".env file found at: {env_file_found}\n"
                    error_msg += f"File size: {file_size} bytes\n\n"
                    
                    if file_size == 0:
                        error_msg += (
                            f"ERROR: The .env file is EMPTY (0 bytes).\n"
                            f"Please open the file and add your API key:\n"
                            f"  OPENAI_API_KEY=sk-your-actual-key-here\n\n"
                            f"To edit the file:\n"
                            f"  1. Open: {env_file_found}\n"
                            f"  2. Add the line: OPENAI_API_KEY=your_key_here\n"
                            f"  3. Save the file\n"
                            f"  4. Make sure the file is saved (not just closed without saving)\n\n"
                        )
                    elif 'OPENAI_API_KEY' not in content:
                        error_msg += (
                            f"ERROR: The .env file does not contain OPENAI_API_KEY.\n"
                            f"File content (first 200 chars):\n{content[:200]}\n\n"
                            f"Please add the following line to the file:\n"
                            f"  OPENAI_API_KEY=sk-your-actual-key-here\n\n"
                        )
                    else:
                        # Key exists in file but wasn't loaded
                        error_msg += (
                            f"ERROR: OPENAI_API_KEY found in file but could not be loaded.\n"
                            f"File content:\n{content[:500]}\n\n"
                            f"Please check:\n"
                            f"  - No spaces around the equals sign\n"
                            f"  - No quotes around the value (or remove them)\n"
                            f"  - The key is on a single line\n"
                            f"  - No special characters that break parsing\n\n"
                        )
                except Exception as e:
                    error_msg += f"ERROR: Could not read .env file: {e}\n\n"
            else:
                error_msg += (
                    f"ERROR: No .env file found in these locations:\n"
                )
                for env_path in env_paths:
                    exists = "EXISTS" if os.path.exists(env_path) else "NOT FOUND"
                    error_msg += f"  {exists}: {env_path}\n"
                error_msg += "\n"
                error_msg += (
                    f"Please create a .env file in the Backend directory.\n"
                    f"Example:\n"
                    f"  1. Create file: Backend/.env\n"
                    f"  2. Add line: OPENAI_API_KEY=sk-your-actual-key-here\n"
                    f"  3. Save the file\n\n"
                )
            
            error_msg += (
                f"Get your API key from: https://platform.openai.com/api-keys\n"
                f"Make sure to save the file after editing!"
            )
            raise ValueError(error_msg)
        
        # Check if API key is the placeholder value
        api_key_clean = api_key.strip()
        if api_key_clean == "YOUR_KEY_HERE" or api_key_clean == "":
            error_msg = (
                f"ERROR: OPENAI_API_KEY in .env file is still the placeholder value.\n"
                f"Please replace 'YOUR_KEY_HERE' with your actual OpenAI API key.\n"
                f"Found .env file at: {env_file_found}\n"
                f"Get your API key from: https://platform.openai.com/api-keys\n\n"
                f"After updating, make sure to SAVE the file!"
            )
            raise ValueError(error_msg)
        
        # Validate API key format (should start with sk-)
        if not api_key_clean.startswith('sk-'):
            print(f"[WARNING] API key doesn't start with 'sk-'. Make sure it's correct.")
        
        client = OpenAI(api_key=api_key)
        print("[OK] Successfully initialized OpenAI client")
        return client
    except ValueError as e:
        print(f"Error initializing OpenAI client: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Error initializing OpenAI client: {e}", file=sys.stderr)
        sys.exit(1)


def format_transaction_for_prompt(row: pd.Series) -> str:
    """Format transaction data as a simple key-value dictionary string for OpenAI prompt."""
    # Convert row to dictionary and format as key: value pairs
    transaction_dict = row.to_dict()
    formatted = "\n".join([f"{k}: {v}" for k, v in transaction_dict.items()])
    return formatted


def explain_transaction(client: OpenAI, row: pd.Series, transaction_id: int) -> Tuple[Optional[str], str]:
    """Generate fraud explanation for a transaction using OpenAI.
    
    Returns:
        Tuple of (explanation, risk_level) or (None, risk_level) on error
    """
    try:
        # Format transaction data
        transaction_dict = row.to_dict()
        formatted = "\n".join([f"{k}: {v}" for k, v in transaction_dict.items()])
        fraud_score = row.get('fraud_score', 0.0)
        risk_level = get_risk_level(fraud_score)
        
        # Create prompt matching the user's format
        prompt = f"""You are an expert financial fraud analyst with 15+ years of experience in banking, 
risk intelligence, behavioral analytics, and fraud forensics.

A machine learning model has flagged the following transaction as suspicious.

Fraud Model Score: {fraud_score}

Risk Level: {risk_level}

Interpretation Guide:
- 0.00–0.20 = Very Low Risk
- 0.20–0.60 = Medium Risk
- 0.60–0.90 = High Risk
- 0.90–1.00 = Critical Risk

Notes:
- V1–V28 are anonymized PCA components; large positive or negative values indicate outlier behavior.
- Be highly specific. Avoid vague or generic statements.
- Reference exact values and anomalies.
- Provide reasoning like a professional fraud investigator.

Possible Fraud Types:
- Card Not Present (CNP)
- Card skimming / cloning
- Bot-driven high velocity attempt
- Account takeover
- Large-amount anomaly
- Money laundering pattern

Your output MUST follow this exact structure:

1. **Summary Risk Rating (Low / Medium / High / Critical)**  

2. **Fraud Indicators (2–4 bullet points referencing exact features)**  

3. **Pattern Deviations (explain what is unusual)**  

4. **Most Likely Fraud Scenario (choose from list)**

Transaction:

{formatted}
"""
        
        # Call OpenAI API with timeout
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            timeout=30.0  # 30 second timeout to prevent hanging
        )
        
        explanation = response.choices[0].message.content
        return explanation, risk_level
        
    except (APIError, APITimeoutError) as e:
        error_type = "timeout" if isinstance(e, APITimeoutError) else "API error"
        print(f"  [ERROR] OpenAI {error_type} for transaction {transaction_id}: {e}", file=sys.stderr)
        fraud_score = row.get('fraud_score', 0.0)
        risk_level = get_risk_level(fraud_score)
        return None, risk_level
    except Exception as e:
        print(f"  [ERROR] Error generating explanation for transaction {transaction_id}: {e}", file=sys.stderr)
        fraud_score = row.get('fraud_score', 0.0)
        risk_level = get_risk_level(fraud_score)
        return None, risk_level


def explain_transactions_parallel(client: OpenAI, flagged_df: pd.DataFrame, max_workers: int = 5) -> Tuple[list, list]:
    """Generate explanations for multiple transactions in parallel for faster processing.
    
    Returns:
        Tuple of (explanations, risk_levels) lists
    """
    explanations = [None] * len(flagged_df)
    risk_levels = [None] * len(flagged_df)
    
    def explain_with_index(args):
        idx, (row_idx, row) = args
        explanation, risk_level = explain_transaction(client, row, row_idx)
        return idx, explanation, risk_level
    
    print(f"  Generating explanations in parallel (max {max_workers} concurrent requests)...")
    start_time = time.time()
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        future_to_idx = {
            executor.submit(explain_with_index, (idx, (row_idx, row))): idx
            for idx, (row_idx, row) in enumerate(flagged_df.iterrows())
        }
        
        # Collect results as they complete
        for future in as_completed(future_to_idx):
            idx = future_to_idx[future]
            try:
                result_idx, explanation, risk_level = future.result()
                explanations[result_idx] = explanation
                risk_levels[result_idx] = risk_level
                print(f"  [OK] Completed explanation {result_idx + 1}/{len(flagged_df)}")
            except Exception as e:
                print(f"  [ERROR] Error getting explanation for transaction {idx}: {e}", file=sys.stderr)
    
    elapsed_time = time.time() - start_time
    print(f"  [OK] All explanations completed in {elapsed_time:.2f} seconds")
    
    return explanations, risk_levels


def print_explanations(flagged_df: pd.DataFrame, explanations: list, risk_levels: list = None):
    """Print formatted explanations for flagged transactions."""
    print("\n" + "=" * 80)
    print(f"FRAUD EXPLANATIONS - Top {len(flagged_df)} Suspicious Transactions")
    print("=" * 80)
    
    if len(flagged_df) == 0:
        print("No transactions flagged as suspicious (fraud_score > 0.90)")
        return
    
    for idx, (row_idx, row) in enumerate(flagged_df.iterrows(), 1):
        explanation = explanations[idx - 1] if idx - 1 < len(explanations) else None
        risk_level = risk_levels[idx - 1] if risk_levels and idx - 1 < len(risk_levels) else get_risk_level(row.get('fraud_score', 0.0))
        
        print(f"\n{'-' * 80}")
        print(f"Transaction #{idx} (Index: {row_idx})")
        print(f"{'-' * 80}")
        print(f"Amount: ${row['Amount']:.2f}")
        print(f"Fraud Score: {row['fraud_score']:.4f}")
        print(f"Risk Level: {risk_level}")
        print(f"Time: {row.get('Time', 'N/A')}")
        
        if explanation:
            print(f"\nExplanation:")
            print(explanation)
        else:
            print("\n[WARNING] Could not generate explanation for this transaction.")
        
        print()


def save_fraud_report(flagged_df: pd.DataFrame, explanations: list, risk_levels: list, output_path: str = "fraud_report.json"):
    """Save fraud analysis results to JSON file."""
    final_reports = []
    
    for idx, (row_idx, row) in enumerate(flagged_df.iterrows()):
        txn_dict = row.to_dict()
        fraud_score = txn_dict.get("fraud_score", 0.0)
        explanation = explanations[idx] if idx < len(explanations) else None
        risk_level = risk_levels[idx] if idx < len(risk_levels) else get_risk_level(fraud_score)
        
        report = {
            "transaction_id": int(row_idx),
            "fraud_score": float(fraud_score),
            "risk_level": risk_level,
            "explanation": explanation,
            "raw_transaction": txn_dict
        }
        final_reports.append(report)
    
    try:
        with open(output_path, "w") as f:
            json.dump(final_reports, f, indent=4, default=str)
        print(f"\n[OK] Final report saved -> {output_path}")
        return output_path
    except Exception as e:
        print(f"[ERROR] Error saving report to {output_path}: {e}", file=sys.stderr)
        return None


def main():
    """Main function to run the fraud detection explainer."""
    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description='Fraud Detection Explainer with OpenAI Integration',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python fraud_explainer.py
  python fraud_explainer.py creditcard.csv
  python fraud_explainer.py --csv data/transactions.csv
  python fraud_explainer.py --csv data/transactions.csv --sample 5000
  python fraud_explainer.py --csv data/transactions.csv --sample 10000 --parallel 10
        """
    )
    parser.add_argument(
        'csv_path',
        nargs='?',
        default='creditcard.csv',
        help='Path to CSV file containing transaction data (default: creditcard.csv)'
    )
    parser.add_argument(
        '--csv',
        dest='csv_path',
        help='Alternative way to specify CSV file path (overrides positional argument)'
    )
    parser.add_argument(
        '--model',
        default='xgb_model.pkl',
        help='Path to trained model file (default: xgb_model.pkl)'
    )
    parser.add_argument(
        '--sample',
        type=int,
        default=None,
        help='Number of rows to randomly sample from CSV for faster processing (default: process all rows)'
    )
    parser.add_argument(
        '--parallel',
        type=int,
        default=5,
        help='Number of parallel OpenAI API calls for explanations (default: 5)'
    )
    parser.add_argument(
        '--top-n',
        type=int,
        default=5,
        help='Number of top flagged transactions to explain (default: 5)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='fraud_report.json',
        help='Output JSON file path for fraud report (default: fraud_report.json)'
    )
    
    args = parser.parse_args()
    
    # Print header
    print("=" * 80)
    print("FRAUD DETECTION EXPLAINER")
    print("=" * 80)
    print()
    
    start_time = time.time()
    
    # Load model
    model = load_model(args.model)
    
    # Load data (with optional sampling for speed)
    csv_path = args.csv_path
    if args.sample:
        print(f"[PERF] Performance mode: Sampling {args.sample} rows for faster processing")
        print(f"  Using CSV file: {csv_path}")
    df = load_data(csv_path, sample_size=args.sample)
    
    # Prepare features
    df_features = prepare_features(df.copy())
    
    # Make predictions
    prediction_start = time.time()
    df_results = predict_fraud(model, df_features)
    prediction_time = time.time() - prediction_start
    print(f"[OK] Prediction completed in {prediction_time:.2f} seconds")
    
    # Get flagged transactions (sorted by fraud_score, top N)
    flagged_df = df_results[df_results['fraud_flag'] == True].sort_values('fraud_score', ascending=False).head(args.top_n)
    
    if len(flagged_df) == 0:
        print("\n" + "=" * 80)
        print("No transactions flagged as suspicious (fraud_score > 0.90)")
        print("=" * 80)
        total_time = time.time() - start_time
        print(f"Total processing time: {total_time:.2f} seconds")
        return
    
    # Initialize OpenAI client
    client = initialize_openai_client()
    
    # Generate explanations for flagged transactions (in parallel for speed)
    print(f"\n[ALERT] Found {len(flagged_df)} suspicious transactions.")
    print(f"Generating explanations for {len(flagged_df)} flagged transactions...")
    explanation_start = time.time()
    
    if len(flagged_df) > 1 and args.parallel > 1:
        explanations, risk_levels = explain_transactions_parallel(client, flagged_df, max_workers=args.parallel)
    else:
        # Sequential for single transaction or if parallel disabled
        explanations = []
        risk_levels = []
        for row_idx, row in flagged_df.iterrows():
            explanation, risk_level = explain_transaction(client, row, row_idx)
            explanations.append(explanation)
            risk_levels.append(risk_level)
    
    explanation_time = time.time() - explanation_start
    print(f"[OK] Explanation generation completed in {explanation_time:.2f} seconds")
    
    # Print formatted explanations
    print_explanations(flagged_df, explanations, risk_levels)
    
    # Save JSON report
    save_fraud_report(flagged_df, explanations, risk_levels, args.output)
    
    total_time = time.time() - start_time
    print("=" * 80)
    print("Analysis completed!")
    print(f"Total processing time: {total_time:.2f} seconds")
    print("=" * 80)


if __name__ == '__main__':
    main()
