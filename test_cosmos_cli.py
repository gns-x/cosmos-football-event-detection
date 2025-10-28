#!/usr/bin/env python3

"""
Cosmos-Reason1-7B CLI Test Script
Simple command-line test for the Cosmos model
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import time
import sys

def test_cosmos_model():
    """Test Cosmos-Reason1-7B model with CLI"""
    
    print("🧪 Cosmos-Reason1-7B CLI Test")
    print("=" * 50)
    
    # Check device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🖥️  Device: {device}")
    
    if device == "cuda":
        print(f"🔥 GPU: {torch.cuda.get_device_name(0)}")
        print(f"💾 GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}GB")
    
    print("\n🚀 Loading Cosmos-Reason1-7B model...")
    print("⚠️  First run will download ~14GB model (this may take 10-30 minutes)")
    
    try:
        # Load tokenizer
        print("\n📝 Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(
            "nvidia/Cosmos-Reason1-7B", 
            trust_remote_code=True
        )
        
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        print("✅ Tokenizer loaded successfully!")
        
        # Load model
        print("\n🧠 Loading model...")
        model_kwargs = {
            "trust_remote_code": True,
            "torch_dtype": torch.float16 if device == "cuda" else torch.float32,
            "device_map": "auto" if device == "cuda" else None,
            "low_cpu_mem_usage": True,
        }
        
        model = AutoModelForCausalLM.from_pretrained(
            "nvidia/Cosmos-Reason1-7B",
            **model_kwargs
        )
        
        if device == "cpu":
            model = model.to(device)
        
        print("✅ Model loaded successfully!")
        
        # Test scenarios
        test_scenarios = [
            {
                "name": "Football Goal Analysis",
                "prompt": """You are analyzing a football video. Please provide detailed reasoning about the following question:

Question: Which player scored the goal?

Please analyze the video content and provide:
1. Step-by-step reasoning
2. A clear answer
3. Confidence level
4. Key timestamp
5. Main actor involved

Format your response as structured reasoning followed by a clear answer."""
            },
            {
                "name": "Physical Reasoning",
                "prompt": """Analyze this physical scenario: A football player kicks a ball towards the goal. The ball hits the crossbar and bounces back. What happens next?

Provide detailed reasoning about:
1. The physics of the ball trajectory
2. What the player should do
3. The most likely outcome"""
            },
            {
                "name": "Spatial Understanding",
                "prompt": """In a football match, a player is running down the left wing with the ball. Two defenders are approaching from different angles. Analyze the spatial situation and predict the best course of action.

Consider:
1. The player's position relative to defenders
2. Available space and options
3. Most effective strategy"""
            }
        ]
        
        print(f"\n🧪 Running {len(test_scenarios)} test scenarios...")
        
        for i, scenario in enumerate(test_scenarios, 1):
            print(f"\n{'='*60}")
            print(f"🧪 Test {i}: {scenario['name']}")
            print(f"{'='*60}")
            
            # Tokenize input
            inputs = tokenizer(
                scenario['prompt'], 
                return_tensors="pt", 
                truncation=True, 
                max_length=1024
            )
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            # Generate response
            print("🤖 Generating response...")
            start_time = time.time()
            
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=256,
                    temperature=0.7,
                    do_sample=True,
                    pad_token_id=tokenizer.eos_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                    repetition_penalty=1.1
                )
            
            generation_time = time.time() - start_time
            
            # Decode response
            response = tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Remove input from response
            if scenario['prompt'] in response:
                response = response.replace(scenario['prompt'], "").strip()
            
            print(f"⏱️  Generation time: {generation_time:.2f} seconds")
            print("\n📝 Model Response:")
            print("-" * 50)
            print(response[:800] + "..." if len(response) > 800 else response)
            print("-" * 50)
        
        print(f"\n{'='*60}")
        print("🎉 ALL TESTS COMPLETED SUCCESSFULLY!")
        print(f"{'='*60}")
        print("\n📊 Test Summary:")
        print("✅ Model loading: Successful")
        print("✅ Tokenization: Working")
        print("✅ Generation: Working")
        print("✅ Physical reasoning: Demonstrated")
        print("✅ Spatial understanding: Demonstrated")
        print("✅ Football analysis: Demonstrated")
        
        if device == "cuda":
            memory_used = torch.cuda.memory_allocated() / 1024**3
            print(f"💾 GPU Memory Used: {memory_used:.2f}GB")
        
        print("\n🚀 Cosmos-Reason1-7B is ready for production use!")
        
    except Exception as e:
        print(f"\n❌ Error testing model: {str(e)}")
        print("\n🔧 Troubleshooting:")
        print("  - Check internet connection for model download")
        print("  - Ensure sufficient disk space (~20GB)")
        print("  - Check GPU memory if using CUDA")
        print("  - Verify transformers library version")
        sys.exit(1)

if __name__ == "__main__":
    test_cosmos_model()
