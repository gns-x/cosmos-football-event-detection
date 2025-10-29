#!/usr/bin/env python3
"""
Test script for NVIDIA NIM integration
Tests the backend API endpoints
"""

import requests
import json
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv('backend/.env')

API_BASE_URL = 'http://localhost:8000'

def test_health():
    """Test health endpoint"""
    print("🔍 Testing health endpoint...")
    try:
        response = requests.get(f"{API_BASE_URL}/health")
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Health check passed: {data}")
            return True
        else:
            print(f"❌ Health check failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Health check error: {e}")
        return False

def test_model_info():
    """Test model info endpoint"""
    print("\n🔍 Testing model info endpoint...")
    try:
        response = requests.get(f"{API_BASE_URL}/model-info")
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Model info: {data}")
            return True
        else:
            print(f"❌ Model info failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Model info error: {e}")
        return False

def test_text_analysis():
    """Test text analysis endpoint"""
    print("\n🔍 Testing text analysis endpoint...")
    try:
        payload = {
            "prompt": "Analyze this football match and tell me about the goals scored",
            "system_prompt": "You are a professional football analyst."
        }
        
        response = requests.post(
            f"{API_BASE_URL}/analyze-text",
            data=payload
        )
        
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Text analysis successful:")
            print(f"   Answer: {data['answer'][:100]}...")
            print(f"   Confidence: {data['confidence']}")
            print(f"   Actor: {data['actor']}")
            return True
        else:
            print(f"❌ Text analysis failed: {response.status_code}")
            print(f"   Error: {response.text}")
            return False
    except Exception as e:
        print(f"❌ Text analysis error: {e}")
        return False

def test_nim_connection():
    """Test direct NVIDIA NIM connection"""
    print("\n🔍 Testing direct NVIDIA NIM connection...")
    
    api_key = os.getenv('NVIDIA_API_KEY')
    if not api_key:
        print("❌ NVIDIA_API_KEY not found in environment")
        return False
    
    try:
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": "nvidia/cosmos-reason1-7b",
            "messages": [
                {"role": "user", "content": "Hello, test connection"}
            ],
            "max_tokens": 10,
            "temperature": 0.1
        }
        
        response = requests.post(
            "https://integrate.api.nvidia.com/v1/chat/completions",
            headers=headers,
            json=payload,
            timeout=30
        )
        
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Direct NIM connection successful:")
            print(f"   Response: {data['choices'][0]['message']['content']}")
            return True
        else:
            print(f"❌ Direct NIM connection failed: {response.status_code}")
            print(f"   Error: {response.text}")
            return False
    except Exception as e:
        print(f"❌ Direct NIM connection error: {e}")
        return False

def main():
    """Run all tests"""
    print("🚀 Starting NVIDIA NIM Integration Tests...")
    print("=" * 50)
    
    tests = [
        ("Health Check", test_health),
        ("Model Info", test_model_info),
        ("Text Analysis", test_text_analysis),
        ("Direct NIM Connection", test_nim_connection)
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name} crashed: {e}")
            results.append((test_name, False))
    
    print("\n" + "=" * 50)
    print("📊 Test Results Summary:")
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"   {test_name}: {status}")
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    print(f"\n🎯 Overall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! NVIDIA NIM integration is working!")
    else:
        print("⚠️  Some tests failed. Check the errors above.")

if __name__ == "__main__":
    main()
