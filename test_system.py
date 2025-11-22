#!/usr/bin/env python3
"""Test script to verify system functionality"""

import os
import sys
import subprocess
import json

def run_command(cmd, cwd=None):
    """Run command and return success status"""
    try:
        result = subprocess.run(cmd, shell=True, cwd=cwd, capture_output=True, text=True)
        return result.returncode == 0, result.stdout, result.stderr
    except Exception as e:
        return False, "", str(e)

def test_system():
    """Test all system components"""
    print("[TEST] Testing Fuel Price Optimization System")
    
    # Test 1: Generate data
    print("\n1. Testing data generation...")
    success, stdout, stderr = run_command("python generate_sample_data.py", cwd="src")
    if success:
        print("[PASS] Data generation: PASS")
    else:
        print(f"[FAIL] Data generation: FAIL - {stderr}")
        return False
    
    # Test 2: Main pipeline
    print("\n2. Testing main pipeline...")
    success, stdout, stderr = run_command("python main.py", cwd="src")
    if success:
        print("[PASS] Main pipeline: PASS")
    else:
        print(f"[FAIL] Main pipeline: FAIL - {stderr}")
        return False
    
    # Test 3: Check outputs
    print("\n3. Testing outputs...")
    required_files = [
        "data/prediction_result.json",
        "models/price_optimizer.pkl"
    ]
    
    for file_path in required_files:
        if os.path.exists(file_path):
            print(f"[PASS] {file_path}: EXISTS")
        else:
            print(f"[FAIL] {file_path}: MISSING")
            return False
    
    # Test 4: Validate prediction result
    print("\n4. Testing prediction result...")
    try:
        with open("data/prediction_result.json", 'r') as f:
            result = json.load(f)
        
        required_keys = ["recommended_price", "expected_volume", "expected_profit"]
        for key in required_keys:
            if key in result:
                print(f"[PASS] {key}: {result[key]}")
            else:
                print(f"[FAIL] {key}: MISSING")
                return False
    except Exception as e:
        print(f"[FAIL] JSON validation: FAIL - {e}")
        return False
    
    print("\n[SUCCESS] ALL TESTS PASSED - System is ready for GitHub!")
    return True

if __name__ == "__main__":
    success = test_system()
    sys.exit(0 if success else 1)