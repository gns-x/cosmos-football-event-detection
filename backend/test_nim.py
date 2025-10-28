#!/usr/bin/env python3

"""
NVIDIA NIM API Test Script
Tests the connection to NVIDIA NIM and Cosmos-Reason1-7B model
"""

import os
import requests
import json
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def test_nvidia_nim():
    """Test NVIDIA NIM API connection"""
    
    # Get configuration
    api_key = os.getenv("NVIDIA_API_KEY")
    base_url = os.getenv("NIM_BASE_URL", "https://api.nim.nvidia.com/v1")
    model_name = os.getenv("NIM_MODEL_NAME", "cosmos-reason1-7b")
    
    if not api_key:
        print("❌ NVIDIA_API_KEY not found in environment variables")
        print("🔑 Please set your NVIDIA API key in the .env file")
        return False
    
    if api_key == "your_nvidia_api_key_here":
        print("❌ Please update .env file with your actual NVIDIA API key")
        print("🔑 Get your API key from: https://build.nvidia.com/")
        return False
    
    print(f"🧪 Testing NVIDIA NIM API connection...")
    print(f"📍 Base URL: {base_url}")
    print(f"🤖 Model: {model_name}")
    
    # Prepare test request
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    payload = {
        "model": model_name,
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "Hello! Can you analyze this simple scenario: A football player kicks a ball towards the goal. What happens next?"
                    }
                ]
            }
        ],
        "max_tokens": 100,
        "temperature": 0.7
    }
    
    try:
        print("📡 Sending test request to NVIDIA NIM...")
        response = requests.post(
            f"{base_url}/chat/completions",
            headers=headers,
            json=payload,
            timeout=30
        )
        
        if response.status_code == 200:
            result = response.json()
            answer = result["choices"][0]["message"]["content"]
            
            print("✅ NVIDIA NIM API connection successful!")
            print(f"🤖 Model response: {answer[:100]}...")
            print("🎉 Ready to use Cosmos-Reason1-7B for video analysis!")
            return True
            
        elif response.status_code == 401:
            print("❌ Authentication failed - Invalid API key")
            print("🔑 Please check your NVIDIA API key")
            return False
            
        elif response.status_code == 404:
            print("❌ Model not found")
            print(f"🤖 Model '{model_name}' may not be available")
            print("📋 Check available models at: https://docs.nvidia.com/nim/")
            return False
            
        else:
            print(f"❌ API request failed with status {response.status_code}")
            print(f"📄 Response: {response.text}")
            return False
            
    except requests.exceptions.Timeout:
        print("❌ Request timeout - API may be slow or unavailable")
        return False
        
    except requests.exceptions.ConnectionError:
        print("❌ Connection error - Check your internet connection")
        return False
        
    except Exception as e:
        print(f"❌ Unexpected error: {str(e)}")
        return False

def main():
    """Main test function"""
    print("🚀 NVIDIA NIM Cosmos-Reason1-7B Test Script")
    print("=" * 50)
    
    success = test_nvidia_nim()
    
    print("\n" + "=" * 50)
    if success:
        print("🎉 All tests passed! Your NVIDIA NIM setup is ready.")
        print("🚀 You can now run: python main_nim.py")
    else:
        print("❌ Tests failed. Please check your configuration.")
        print("📚 See README_NIM.md for setup instructions")

if __name__ == "__main__":
    main()
