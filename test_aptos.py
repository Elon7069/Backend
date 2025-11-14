"""
Test script for Aptos blockchain integration.
Tests hash computation and blockchain publishing functionality.
"""

import os
import sys
import json
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def test_hash_utils():
    """Test hash utilities."""
    print("=" * 60)
    print("Testing Hash Utils")
    print("=" * 60)
    try:
        from hash_utils import canonical_json, compute_sha256
        
        # Test data
        test_data = {
            "flagged_count": 2,
            "final_report": [
                {
                    "transaction_id": 1,
                    "fraud_score": 0.95,
                    "risk_level": "Critical"
                }
            ]
        }
        
        # Test canonical JSON
        canonical = canonical_json(test_data)
        print(f"✅ Canonical JSON: {canonical[:100]}...")
        
        # Test SHA-256 computation
        hash_value = compute_sha256(test_data)
        print(f"✅ SHA-256 Hash: {hash_value}")
        print(f"✅ Hash length: {len(hash_value)} characters")
        
        if len(hash_value) != 64:
            print(f"❌ Invalid hash length. Expected 64, got {len(hash_value)}")
            return False
        
        # Test determinism (same input should produce same hash)
        hash2 = compute_sha256(test_data)
        if hash_value != hash2:
            print("❌ Hash is not deterministic!")
            return False
        
        print("✅ Hash is deterministic\n")
        return True
        
    except Exception as e:
        print(f"❌ Hash utils test failed: {e}\n")
        import traceback
        traceback.print_exc()
        return False


def test_aptos_client():
    """Test Aptos client functions."""
    print("=" * 60)
    print("Testing Aptos Client")
    print("=" * 60)
    
    try:
        from aptos_client import get_aptos_account, get_explorer_url, publish_hash_on_aptos
        from aptos_sdk.client import RestClient
        
        # Check if private key is set
        private_key = os.getenv("APTOS_PRIVATE_KEY")
        if not private_key:
            print("⚠️  APTOS_PRIVATE_KEY not found in .env file")
            print("   Skipping Aptos client tests")
            print("   To test, add APTOS_PRIVATE_KEY to your .env file\n")
            return None
        
        print(f"✅ APTOS_PRIVATE_KEY found (length: {len(private_key)})")
        
        # Test account loading
        try:
            account = get_aptos_account()
            if account is None:
                print("❌ Failed to load account")
                return False
            print(f"✅ Account loaded: {account.address()}")
        except Exception as e:
            print(f"❌ Failed to load account: {e}")
            return False
        
        # Test connection to Aptos testnet
        try:
            client = RestClient("https://fullnode.testnet.aptoslabs.com/v1")
            account_data = client.account(account.address())
            balance = account_data.get("data", {}).get("coin", {}).get("value", "0")
            print(f"✅ Connected to Aptos testnet")
            print(f"✅ Account balance: {balance} APT")
            
            # Check if account has enough balance for transactions
            # Transaction fee is usually around 100-1000 octas (0.0001-0.001 APT)
            if int(balance) < 1000:
                print("⚠️  Warning: Low balance. You may need testnet APT for transactions.")
                print("   Get testnet APT from: https://faucet.testnet.aptoslabs.com/")
        except Exception as e:
            print(f"❌ Failed to connect to Aptos testnet: {e}")
            return False
        
        # Test explorer URL generation
        test_tx_hash = "0x1234567890abcdef" * 4
        explorer_url = get_explorer_url(test_tx_hash, network="testnet")
        print(f"✅ Explorer URL: {explorer_url}")
        
        print("\n✅ Aptos client tests passed!\n")
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("   Install aptos-sdk: pip install aptos-sdk==0.6.3\n")
        return False
    except Exception as e:
        print(f"❌ Aptos client test failed: {e}\n")
        import traceback
        traceback.print_exc()
        return False


def test_publish_hash():
    """Test publishing a hash to Aptos blockchain."""
    print("=" * 60)
    print("Testing Hash Publishing to Aptos")
    print("=" * 60)
    
    try:
        from hash_utils import compute_sha256
        from aptos_client import publish_hash_on_aptos, get_explorer_url
        
        # Check if private key is set
        private_key = os.getenv("APTOS_PRIVATE_KEY")
        if not private_key:
            print("⚠️  APTOS_PRIVATE_KEY not found in .env file")
            print("   Skipping hash publishing test")
            print("   To test, add APTOS_PRIVATE_KEY to your .env file\n")
            return None
        
        # Create test data and compute hash
        test_data = {
            "test": "fraud_report",
            "timestamp": "2024-01-01T00:00:00Z",
            "flagged_count": 1
        }
        
        hash_value = compute_sha256(test_data)
        print(f"✅ Computed hash: {hash_value}")
        
        # Ask user if they want to publish (to avoid spamming testnet)
        print("\n⚠️  This will publish a transaction to Aptos testnet")
        print("   You need testnet APT in your account for transaction fees")
        response = input("   Do you want to publish the hash? (yes/no): ").strip().lower()
        
        if response not in ['yes', 'y']:
            print("   Skipping hash publishing test\n")
            return None
        
        # Publish hash
        try:
            print("\n📤 Publishing hash to Aptos testnet...")
            tx_hash = publish_hash_on_aptos(hash_value)
            explorer_url = get_explorer_url(tx_hash, network="testnet")
            
            print(f"✅ Transaction published!")
            print(f"✅ Transaction hash: {tx_hash}")
            print(f"✅ Explorer URL: {explorer_url}")
            print(f"\n   You can verify the transaction at:")
            print(f"   {explorer_url}\n")
            
            return True
            
        except Exception as e:
            print(f"❌ Failed to publish hash: {e}")
            print(f"   Error type: {type(e).__name__}")
            import traceback
            traceback.print_exc()
            return False
        
    except Exception as e:
        print(f"❌ Hash publishing test failed: {e}\n")
        import traceback
        traceback.print_exc()
        return False


def test_api_endpoint():
    """Test the /blockchain/publish API endpoint."""
    print("=" * 60)
    print("Testing /blockchain/publish API Endpoint")
    print("=" * 60)
    
    try:
        import requests
        
        BASE_URL = "http://localhost:8000"
        
        # Test if API server is running
        try:
            response = requests.get(f"{BASE_URL}/", timeout=5)
            if response.status_code != 200:
                print(f"⚠️  API server returned status {response.status_code}")
                print("   Make sure the API server is running: uvicorn app:app --reload\n")
                return None
        except requests.exceptions.ConnectionError:
            print("⚠️  Cannot connect to API server")
            print("   Make sure the API server is running: uvicorn app:app --reload\n")
            return None
        
        print("✅ API server is running")
        
        # Test hash publishing endpoint
        test_hash = "a" * 64  # 64 character hex string (valid format)
        
        print(f"\n📤 Testing /blockchain/publish endpoint...")
        print(f"   Hash: {test_hash}")
        
        response = requests.post(
            f"{BASE_URL}/blockchain/publish",
            json={"sha256": test_hash},
            timeout=30
        )
        
        print(f"   Status Code: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Hash published successfully!")
            print(f"   Transaction hash: {result.get('aptos_tx')}")
            print(f"   Explorer URL: {result.get('aptos_explorer_url')}")
            print(f"\n   You can verify at: {result.get('aptos_explorer_url')}\n")
            return True
        else:
            print(f"❌ Failed to publish hash")
            print(f"   Response: {response.text}")
            return False
        
    except ImportError:
        print("❌ 'requests' library not found")
        print("   Install with: pip install requests\n")
        return False
    except Exception as e:
        print(f"❌ API endpoint test failed: {e}\n")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    print("\n" + "=" * 60)
    print("APTOS BLOCKCHAIN INTEGRATION TESTS")
    print("=" * 60)
    print()
    
    results = []
    
    # Test hash utils
    results.append(("Hash Utils", test_hash_utils()))
    
    # Test Aptos client
    results.append(("Aptos Client", test_aptos_client()))
    
    # Test hash publishing (optional, requires user confirmation)
    results.append(("Hash Publishing", test_publish_hash()))
    
    # Test API endpoint (optional, requires API server)
    results.append(("API Endpoint", test_api_endpoint()))
    
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
    main()

