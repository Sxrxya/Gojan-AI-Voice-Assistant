"""
Retriever Acceptance Test Suite
===============================
Tests 10 predefined questions against the FAISS vector database.
Verifies if the retriever's top result contains the expected keyword.
"""

import sys
import os

# Ensure we can import phase_b_local modules
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from phase_b_local.services.retriever import load_retriever, retrieve

# Test matrix: Question -> Expected substring in top chunk
TEST_CASES = {
    "What is the TNEA code?": "1123",
    "Where is Gojan located?": "redhills",
    "Who is the chairman of Gojan?": "natarajan",
    "Tell me about hostel": "hostel",
    "What are the bus routes?": "bus route",
    "What courses are offered?": "engineering",
    "Tell me about placements": "placement",
    "Is the college NAAC accredited?": "naac",
    "What is the official email address?": "gsbt",
    "Tell me about sports": "sport",
}

def run_tests():
    print("Loading Retriever...")
    try:
        idx, docs, emb = load_retriever()
    except Exception as e:
        print(f"FAILED to load retriever: {e}")
        return

    print(f"[OK] FAISS Index Loaded: {idx.ntotal} documents")
    print("-" * 50)
    
    passed = 0
    total = len(TEST_CASES)

    for question, expected_keyword in TEST_CASES.items():
        results = retrieve(question, idx, docs, emb)
        
        if not results:
            print(f"[FAIL] {question}")
            print(f"       -> Expected: '{expected_keyword}'")
            print(f"       -> Got: NO RESULTS (Distance filter may be too strict)")
            continue
            
        top_result = results[0].lower()
        
        if expected_keyword.lower() in top_result:
            print(f"[PASS] {question}")
            passed += 1
        else:
            print(f"[FAIL] {question}")
            print(f"       -> Expected: '{expected_keyword}'")
            # Print first 100 chars of actual result
            clean_top = top_result[:100].replace('\n', ' ')
            print(f"       -> Got top chunk: '{clean_top}...'")

    print("-" * 50)
    print(f"Results: {passed}/{total} Passed")
    
    if passed == total:
        print("Score: 100% - PERFECT!")
    else:
        print(f"Score: {(passed/total)*100:.1f}%")

if __name__ == "__main__":
    run_tests()
