"""
Aptos blockchain client for hash anchoring.
"""

import os
from typing import Optional, Tuple
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

try:
    from aptos_sdk.account import Account
    from aptos_sdk.client import RestClient
except ImportError as e:
    raise ImportError(
        f"Aptos SDK not installed. Install with: pip install aptos-sdk==0.6.3\n"
        f"Original error: {e}"
    )

# Aptos testnet fullnode URL
APTOS_NODE_URL = "https://fullnode.testnet.aptoslabs.com/v1"


def get_aptos_account() -> Optional[Account]:
    """
    Load Aptos account from private key in environment variables.
    
    Returns:
        Account object or None if private key is not set
    """
    private_key_hex = os.getenv("APTOS_PRIVATE_KEY")
    
    if not private_key_hex:
        return None
    
    try:
        # Remove '0x' prefix if present
        if private_key_hex.startswith('0x'):
            private_key_hex = private_key_hex[2:]
        
        # Remove 'ed25519-priv-' prefix if present (Petra wallet format)
        if private_key_hex.startswith('ed25519-priv-'):
            private_key_hex = private_key_hex.replace('ed25519-priv-', '')
            if private_key_hex.startswith('0x'):
                private_key_hex = private_key_hex[2:]
        
        # Load account from hex string
        account = Account.load_key(private_key_hex)
        return account
    except Exception as e:
        raise ValueError(f"Failed to load Aptos account from private key: {e}")


def publish_hash_on_aptos(hash_hex: str) -> str:
    """
    Publish SHA-256 hash to Aptos blockchain.
    
    This function creates a verifiable transaction on-chain that serves as an anchor
    for the hash. The transaction hash can be used to verify the hash was anchored
    at a specific time on the blockchain.
    
    Note: The hash itself is not stored in the transaction data in this implementation.
    For production use, a custom Move module would be needed to store the hash on-chain.
    This implementation creates a transaction that can be associated with the hash
    for verification purposes.
    
    Args:
        hash_hex: SHA-256 hash as hex string (64 characters)
        
    Returns:
        Transaction hash (can be used to verify the transaction on Aptos explorer)
        
    Raises:
        ValueError: If private key is not configured or hash format is invalid
        Exception: If transaction submission fails
    """
    # Get account
    account = get_aptos_account()
    if account is None:
        raise ValueError(
            "APTOS_PRIVATE_KEY not found in environment variables. "
            "Please set APTOS_PRIVATE_KEY in your .env file."
        )
    
    # Initialize client
    client = RestClient(APTOS_NODE_URL)
    
    try:
        # Validate hash hex format
        if len(hash_hex) != 64:
            raise ValueError(f"Invalid hash length. Expected 64 characters (SHA-256), got {len(hash_hex)}")
        
        # Convert hash hex to bytes to validate it's valid hex
        try:
            hash_bytes = bytes.fromhex(hash_hex)
        except ValueError as e:
            raise ValueError(f"Invalid hex string: {e}")
        
        # Get account address
        account_address = account.address()
        account_address_hex = account_address.hex()
        
        # Check if account exists (this will fail if account doesn't exist on-chain)
        try:
            account_data = client.account(account_address)
        except Exception as account_error:
            error_str = str(account_error)
            # Account might not exist on testnet - provide helpful error
            if "not found" in error_str.lower() or "404" in error_str or "does not exist" in error_str.lower():
                raise ValueError(
                    f"Account not found on Aptos testnet.\n"
                    f"Address: {account_address_hex}\n\n"
                    f"To fund your account:\n"
                    f"1. Visit the Aptos faucet: https://faucet.testnet.aptoslabs.com/\n"
                    f"2. Enter your address: {account_address_hex}\n"
                    f"3. Request testnet APT tokens\n"
                    f"4. Wait for confirmation and try again\n\n"
                    f"Original error: {error_str}"
                )
            else:
                # Other error - might be network issue
                raise ValueError(
                    f"Failed to check account on Aptos testnet.\n"
                    f"Address: {account_address_hex}\n"
                    f"Error: {error_str}\n\n"
                    f"This might be a network issue. Please try again."
                )
        
        # Try to check balance (optional - if this fails, we'll try the transaction anyway)
        balance_checked = False
        try:
            resources = client.account_resources(account_address)
            balance = 0
            for resource in resources:
                resource_type = resource.get("type", "")
                if "0x1::coin::CoinStore" in resource_type and "AptosCoin" in resource_type:
                    coin_data = resource.get("data", {})
                    coin_value = coin_data.get("coin", {}).get("value", "0")
                    balance = int(coin_value) if coin_value else 0
                    balance_checked = True
                    break
            
            # Check if account has sufficient balance (need at least 10000 octa = 0.00001 APT for gas)
            if balance_checked:
                min_balance = 10000  # Minimum balance needed for gas (conservative estimate)
                if balance < min_balance:
                    raise ValueError(
                        f"Insufficient balance to pay for transaction fees.\n"
                        f"Account balance: {balance} octa ({balance / 100_000_000:.8f} APT)\n"
                        f"Minimum required: {min_balance} octa ({min_balance / 100_000_000:.8f} APT)\n\n"
                        f"To fund your account:\n"
                        f"1. Visit: https://faucet.testnet.aptoslabs.com/?address={account_address_hex}\n"
                        f"2. Request testnet APT tokens\n"
                        f"3. Wait for confirmation and try again\n\n"
                        f"Account address: {account_address_hex}"
                    )
        except ValueError:
            raise  # Re-raise ValueError (balance errors)
        except Exception:
            # If balance check fails, proceed with transaction attempt
            # The transaction will fail with a clearer error if balance is insufficient
            pass
        
        # Create transfer payload
        # Transfer 1 octa (smallest unit) to self - this creates a minimal transaction
        # Using 1 octa instead of 0 to avoid rejection
        payload = {
            "type": "entry_function_payload",
            "function": "0x1::coin::transfer",
            "type_arguments": ["0x1::aptos_coin::AptosCoin"],
            "arguments": [
                account_address.hex(),
                "1"  # Transfer 1 octa (minimum) to self
            ]
        }
        
        # Submit transaction
        try:
            transaction_hash = client.submit_transaction(account, payload)
            
            # Wait for transaction confirmation
            client.wait_for_transaction(transaction_hash)
            
            return transaction_hash
        except Exception as tx_error:
            error_msg = str(tx_error)
            
            # Provide helpful error messages
            if "INSUFFICIENT_BALANCE" in error_msg or "balance" in error_msg.lower():
                raise ValueError(
                    f"Insufficient balance to pay for transaction fees.\n"
                    f"Please fund your account using the Aptos faucet:\n"
                    f"https://faucet.testnet.aptoslabs.com/?address={account_address.hex()}\n"
                    f"Your account address: {account_address.hex()}\n"
                    f"Original error: {error_msg}"
                )
            elif "SEQUENCE_NUMBER" in error_msg or "sequence" in error_msg.lower():
                raise ValueError(
                    f"Transaction sequence number error. This might be a network issue.\n"
                    f"Please try again in a few moments.\n"
                    f"Original error: {error_msg}"
                )
            else:
                raise Exception(
                    f"Transaction submission failed: {error_msg}\n"
                    f"Account address: {account_address.hex()}\n"
                    f"Network: testnet\n"
                    f"If this persists, check:\n"
                    f"1. Account has sufficient balance: https://explorer.aptoslabs.com/account/{account_address.hex()}?network=testnet\n"
                    f"2. Network connectivity\n"
                    f"3. Aptos testnet status"
                )
        
    except ValueError:
        raise  # Re-raise ValueError (validation errors)
    except Exception as e:
        # Wrap other exceptions with context
        error_msg = str(e)
        if "APTOS_PRIVATE_KEY" in error_msg or "private key" in error_msg.lower():
            raise ValueError(error_msg)
        else:
            raise Exception(f"Failed to publish hash to Aptos: {error_msg}")


def get_explorer_url(tx_hash: str, network: str = "testnet") -> str:
    """
    Get Aptos explorer URL for a transaction.
    
    Args:
        tx_hash: Transaction hash
        network: Network name (devnet, testnet, mainnet)
        
    Returns:
        Explorer URL
    """
    return f"https://explorer.aptoslabs.com/txn/{tx_hash}?network={network}"


def get_account_explorer_url(account_address: str, network: str = "testnet") -> str:
    """
    Get Aptos explorer URL for an account.
    
    Args:
        account_address: Account address (hex string)
        network: Network name (devnet, testnet, mainnet)
        
    Returns:
        Explorer URL
    """
    return f"https://explorer.aptoslabs.com/account/{account_address}?network={network}"


def search_hash_on_aptos(hash_hex: str) -> Tuple[str, Optional[str]]:
    """
    Search for a hash in Aptos blockchain transactions.
    
    Since the current implementation doesn't store the hash directly in transaction data,
    this function checks if the account has been used to publish transactions.
    For a proper implementation, a custom Move module would be needed to store hashes.
    
    Args:
        hash_hex: SHA-256 hash as hex string (64 characters)
        
    Returns:
        Tuple of (status, account_address)
        - status: "MATCH" if hash found, "NOT_FOUND" if not found
        - account_address: Account address if found, None otherwise
        
    Raises:
        ValueError: If hash format is invalid
        Exception: If blockchain query fails
    """
    # Validate hash format
    if len(hash_hex) != 64:
        raise ValueError(f"Invalid hash length. Expected 64 characters (SHA-256), got {len(hash_hex)}")
    
    try:
        bytes.fromhex(hash_hex)
    except ValueError as e:
        raise ValueError(f"Invalid hex string: {e}")
    
    # Get account
    account = get_aptos_account()
    if account is None:
        # If no account configured, we can't verify
        return "NOT_FOUND", None
    
    # Initialize client
    client = RestClient(APTOS_NODE_URL)
    
    try:
        # Get account address
        account_address = account.address().hex()
        
        # Check if account exists and has transactions
        # Note: Since we don't store the hash in transaction data,
        # we can only verify that the account is accessible.
        # For a proper implementation, we would need to:
        # 1. Store hash -> transaction mapping in a database, or
        # 2. Use a custom Move module that stores hashes on-chain
        
        # For now, we'll check if we can access the account
        # This indicates blockchain connectivity
        account_data = client.account(account_address)
        
        # Since we can't directly search for the hash in transaction data,
        # we return NOT_FOUND. In a production system, you would:
        # - Query a database that maps hashes to transaction hashes, or
        # - Use a custom Move module that stores hashes and allows querying
        
        # For demonstration, we'll return NOT_FOUND
        # In a real implementation, you would check a hash registry
        return "NOT_FOUND", account_address
        
    except Exception as e:
        raise Exception(f"Failed to search hash on Aptos: {str(e)}")

