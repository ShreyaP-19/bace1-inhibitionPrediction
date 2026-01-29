
import sys
import os
import json

# Add project root to sys.path
sys.path.append(os.getcwd())

from main.predict import predict

def run_verification():
    smiles = "C1=CC=C(C=C1)C(=O)O" # Benzoic Acid
    print(f"Verifying prediction for {smiles}...")
    
    try:
        prob, pic50 = predict(smiles)
        
        result = {
            "smiles": smiles,
            "probability": prob,
            "pIC50": pic50,
            "class": "ACTIVE" if prob > 0.5 else "INACTIVE"
        }
        
        with open("verification_result.json", "w") as f:
            json.dump(result, f, indent=4)
            
        print("Verification successful. Result saved to verification_result.json")
        
    except Exception as e:
        print(f"Verification failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    run_verification()
