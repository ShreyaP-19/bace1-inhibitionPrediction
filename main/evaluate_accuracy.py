
import pandas as pd
import torch
import sys
import os
import math

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from main.predict import load_model, predict_with_model

def calculate_metrics(csv_path, model, tokenizer, device):
    print(f"Loading data from {csv_path}...")
    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        print(f"Error: {csv_path} not found.")
        return

    correct_predictions = 0
    total_samples = 0
    squared_error_sum = 0
    
    # Check column names
    # Assuming 'mol' for SMILES, 'Class' for label, 'pIC50' for regression target
    smiles_col = "mol"
    class_col = "Class"
    reg_col = "pIC50"
    
    print(f"evaluating {len(df)} samples...")
    
    for i, row in df.iterrows():
        smiles = row[smiles_col]
        true_class = int(row[class_col])
        true_pic50 = float(row[reg_col])
        
        # Predict (verbose=False)
        prob, pred_pic50 = predict_with_model(smiles, model, tokenizer, device, verbose=False)
        
        if prob is None:
            continue
            
        # Classification Accuracy
        pred_class = 1 if prob > 0.5 else 0
        if pred_class == true_class:
            correct_predictions += 1
            
        # Regression RMSE
        squared_error_sum += (pred_pic50 - true_pic50) ** 2
        total_samples += 1
        
        if i % 100 == 0:
            print(f"Processed {i}/{len(df)}...")

    accuracy = correct_predictions / total_samples if total_samples > 0 else 0
    rmse = math.sqrt(squared_error_sum / total_samples) if total_samples > 0 else 0
    
    print("-" * 30)
    print(f"Results for {csv_path}")
    print("-" * 30)
    print(f"Total Samples: {total_samples}")
    print(f"Accuracy     : {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"RMSE         : {rmse:.4f}")
    print("-" * 30)
    return accuracy, rmse

def main():
    model, tokenizer, device = load_model("bace_model.pth")
    
    print("\n=== Training Set Evaluation ===")
    calculate_metrics("data/train.csv", model, tokenizer, device)
    
    print("\n=== Test Set Evaluation ===")
    calculate_metrics("data/test.csv", model, tokenizer, device)

if __name__ == "__main__":
    main()
