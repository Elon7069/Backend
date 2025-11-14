"""
Test script for HAWKEYE Fraud Detection API endpoints.
Run this after starting the API server to validate all endpoints.
"""

import requests
import json
import os
from pathlib import Path

# API base URL
BASE_URL = "http://localhost:8000"

def test_health_check():
    """Test the health check endpoint."""
    print("=" * 60)
    print("Testing GET / (Health Check)")
    print("=" * 60)
    try:
        response = requests.get(f"{BASE_URL}/")
        print(f"Status Code: {response.status_code}")
        print(f"Response: {json.dumps(response.json(), indent=2)}")
        assert response.status_code == 200
        assert response.json()["status"] == "running"
        print("✅ Health check passed!\n")
        return True
    except Exception as e:
        print(f"❌ Health check failed: {e}\n")
        return False

def test_detect_endpoint(csv_path):
    """Test the /detect endpoint."""
    print("=" * 60)
    print("Testing POST /detect")
    print("=" * 60)
    try:
        if not os.path.exists(csv_path):
            print(f"⚠️  CSV file not found: {csv_path}")
            print("   Skipping /detect test\n")
            return None
        
        with open(csv_path, 'rb') as f:
            files = {'file': f}
            data = {'top_k': 5, 'threshold': 0.9}
            response = requests.post(f"{BASE_URL}/detect", files=files, data=data)
        
        print(f"Status Code: {response.status_code}")
        if response.status_code == 200:
            result = response.json()
            print(f"Flagged Count: {result.get('flagged_count', 0)}")
            print(f"Top Suspicious: {len(result.get('top_suspicious', []))} transactions")
            if result.get('top_suspicious'):
                print(f"First Transaction Fraud Score: {result['top_suspicious'][0].get('fraud_score', 'N/A')}")
                print(f"First Transaction Risk Level: {result['top_suspicious'][0].get('risk_level', 'N/A')}")
            print("✅ /detect endpoint passed!\n")
            return True
        else:
            print(f"Response: {response.text}")
            print("❌ /detect endpoint failed!\n")
            return False
    except Exception as e:
        print(f"❌ /detect endpoint failed: {e}\n")
        return False

def test_explain_endpoint(csv_path):
    """Test the /explain endpoint."""
    print("=" * 60)
    print("Testing POST /explain")
    print("=" * 60)
    try:
        if not os.path.exists(csv_path):
            print(f"⚠️  CSV file not found: {csv_path}")
            print("   Skipping /explain test\n")
            return None
        
        with open(csv_path, 'rb') as f:
            files = {'file': f}
            data = {'top_k': 3, 'threshold': 0.9, 'parallel': 3}
            response = requests.post(f"{BASE_URL}/explain", files=files, data=data)
        
        print(f"Status Code: {response.status_code}")
        if response.status_code == 200:
            result = response.json()
            print(f"Flagged Count: {result.get('flagged_count', 0)}")
            print(f"Results: {len(result.get('results', []))} transactions with explanations")
            if result.get('results'):
                first_result = result['results'][0]
                print(f"First Transaction ID: {first_result.get('transaction_id', 'N/A')}")
                print(f"First Transaction Fraud Score: {first_result.get('fraud_score', 'N/A')}")
                print(f"First Transaction Risk Level: {first_result.get('risk_level', 'N/A')}")
                has_explanation = bool(first_result.get('explanation'))
                print(f"Has Explanation: {has_explanation}")
            print("✅ /explain endpoint passed!\n")
            return True
        else:
            print(f"Response: {response.text}")
            print("❌ /explain endpoint failed!\n")
            return False
    except Exception as e:
        print(f"❌ /explain endpoint failed: {e}\n")
        return False

def test_report_endpoint(csv_path):
    """Test the /report endpoint."""
    print("=" * 60)
    print("Testing POST /report")
    print("=" * 60)
    try:
        if not os.path.exists(csv_path):
            print(f"⚠️  CSV file not found: {csv_path}")
            print("   Skipping /report test\n")
            return None
        
        with open(csv_path, 'rb') as f:
            files = {'file': f}
            data = {'top_k': 3, 'threshold': 0.9, 'parallel': 3}
            response = requests.post(f"{BASE_URL}/report", files=files, data=data)
        
        print(f"Status Code: {response.status_code}")
        if response.status_code == 200:
            result = response.json()
            print(f"Flagged Count: {result.get('flagged_count', 0)}")
            print(f"Final Report: {len(result.get('final_report', []))} transactions")
            if result.get('final_report'):
                first_report = result['final_report'][0]
                print(f"First Transaction ID: {first_report.get('transaction_id', 'N/A')}")
                print(f"First Transaction Fraud Score: {first_report.get('fraud_score', 'N/A')}")
                print(f"First Transaction Risk Level: {first_report.get('risk_level', 'N/A')}")
                has_explanation = bool(first_report.get('explanation'))
                print(f"Has Explanation: {has_explanation}")
            print("✅ /report endpoint passed!\n")
            return True
        else:
            print(f"Response: {response.text}")
            print("❌ /report endpoint failed!\n")
            return False
    except Exception as e:
        print(f"❌ /report endpoint failed: {e}\n")
        return False

def test_error_handling():
    """Test error handling."""
    print("=" * 60)
    print("Testing Error Handling")
    print("=" * 60)
    
    # Test invalid file type
    try:
        files = {'file': ('test.txt', b'not a csv', 'text/plain')}
        response = requests.post(f"{BASE_URL}/detect", files=files)
        print(f"Invalid file type test - Status: {response.status_code}")
        if response.status_code == 400:
            print("✅ Invalid file type correctly rejected!\n")
            return True
        else:
            print("⚠️  Expected 400 but got different status\n")
            return False
    except Exception as e:
        print(f"❌ Error handling test failed: {e}\n")
        return False

def test_blockchain_publish():
    """Test the /blockchain/publish endpoint."""
    print("=" * 60)
    print("Testing POST /blockchain/publish")
    print("=" * 60)
    try:
        # Test with valid hash
        test_hash = "a" * 64  # 64 character hex string
        response = requests.post(
            f"{BASE_URL}/blockchain/publish",
            json={"sha256": test_hash},
            timeout=30
        )
        
        print(f"Status Code: {response.status_code}")
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Hash published successfully!")
            print(f"   SHA-256: {result.get('sha256', 'N/A')}")
            print(f"   Transaction Hash: {result.get('aptos_tx', 'N/A')}")
            print(f"   Explorer URL: {result.get('aptos_explorer_url', 'N/A')}")
            if result.get('aptos_explorer_url'):
                print(f"\n   Verify transaction at: {result['aptos_explorer_url']}")
            print("✅ /blockchain/publish endpoint passed!\n")
            return True
        elif response.status_code == 500:
            # Check if it's an Aptos error (private key not set, etc.)
            error_detail = response.json().get('detail', {})
            if isinstance(error_detail, dict) and error_detail.get('error') == "Aptos publishing failed":
                print("⚠️  Aptos publishing failed (check APTOS_PRIVATE_KEY in .env)")
                print(f"   Details: {error_detail.get('details', 'N/A')}")
                print("   This is expected if APTOS_PRIVATE_KEY is not set\n")
                return None
            else:
                print(f"Response: {response.text}")
                print("❌ /blockchain/publish endpoint failed!\n")
                return False
        else:
            print(f"Response: {response.text}")
            print("❌ /blockchain/publish endpoint failed!\n")
            return False
    except Exception as e:
        print(f"❌ /blockchain/publish endpoint failed: {e}\n")
        return False

def main():
    """Run all API tests."""
    print("\n" + "=" * 60)
    print("HAWKEYE API Test Suite")
    print("=" * 60)
    print("\nMake sure the API server is running:")
    print("  uvicorn app:app --reload")
    print("\n" + "=" * 60 + "\n")
    
    # Find CSV file
    csv_path = "creditcard.csv"
    if not os.path.exists(csv_path):
        # Try to find any CSV in current directory
        csv_files = list(Path('.').glob('*.csv'))
        if csv_files:
            csv_path = str(csv_files[0])
            print(f"Using CSV file: {csv_path}\n")
        else:
            print("⚠️  No CSV file found. Some tests will be skipped.\n")
    
    results = []
    
    # Run tests
    results.append(("Health Check", test_health_check()))
    results.append(("Detect Endpoint", test_detect_endpoint(csv_path)))
    results.append(("Explain Endpoint", test_explain_endpoint(csv_path)))
    results.append(("Report Endpoint", test_report_endpoint(csv_path)))
    results.append(("Error Handling", test_error_handling()))
    results.append(("Blockchain Publish", test_blockchain_publish()))
    
    # Summary
    print("=" * 60)
    print("Test Summary")
    print("=" * 60)
    passed = 0
    failed = 0
    skipped = 0
    
    for test_name, result in results:
        if result is True:
            print(f"✅ {test_name}: PASSED")
            passed += 1
        elif result is False:
            print(f"❌ {test_name}: FAILED")
            failed += 1
        else:
            print(f"⚠️  {test_name}: SKIPPED")
            skipped += 1
    
    print("=" * 60)
    print(f"Total: {passed} passed, {failed} failed, {skipped} skipped")
    print("=" * 60)
    
    if failed == 0:
        print("\n🎉 All tests passed!")
    else:
        print(f"\n⚠️  {failed} test(s) failed. Check the output above.")

if __name__ == "__main__":
    try:
        import requests
    except ImportError:
        print("❌ 'requests' library not found. Install it with:")
        print("   pip install requests")
        exit(1)
    
    main()

