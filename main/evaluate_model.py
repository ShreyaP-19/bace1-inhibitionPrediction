import torch
from torch_geometric.loader import DataLoader
from main.bace_dataset import BACEDataset
from models.gnn import GCN
from sklearn.metrics import roc_auc_score, mean_squared_error
import numpy as np
import sys
import os

def evaluate_split(model, loader, device, split_name):
    model.eval()
    all_preds_class, all_labels_class = [], []
    all_preds_reg, all_labels_reg = [], []
    
    print(f"Processing {split_name} set...")
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            out_class, out_reg = model(
                x=batch.x, 
                edge_index=batch.edge_index, 
                batch=batch.batch,
                input_ids=batch.input_ids,
                attention_mask=batch.attention_mask,
                heavy_atom_mask=batch.heavy_atom_mask,
                extra_features=batch.extra_features
            )
            # Classification Outputs
            all_preds_class.extend(torch.sigmoid(out_class).view(-1).cpu().numpy())
            all_labels_class.extend(batch.y_class.view(-1).cpu().numpy())
            # Regression Outputs
            all_preds_reg.extend(out_reg.view(-1).cpu().numpy())
            all_labels_reg.extend(batch.y_reg.view(-1).cpu().numpy())

    # Metrics
    try:
        auc = roc_auc_score(all_labels_class, all_preds_class)
    except:
        auc = 0.0
    
    rmse = np.sqrt(mean_squared_error(all_labels_reg, all_preds_reg))
    
    print(f"--- {split_name} Results ---")
    print(f"Classification AUC: {auc:.4f}")
    print(f"Regression RMSE   : {rmse:.4f}")
    print("-" * 30)

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. Load Data
    train_dataset = BACEDataset("data/train.csv")
    test_dataset = BACEDataset("data/test.csv")
    
    # Larger batch size for evaluation speed
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    # 2. Load Model
    # Get dims from a sample
    sample = train_dataset[0]
    extra_dim = sample.extra_features.shape[1]
    input_dim = sample.x.shape[1]
    
    model = GCN(
        input_dim=input_dim, 
        hidden_dim=128, 
        extra_features_dim=extra_dim
    ).to(device)
    
    model_path = "bace_model.pth"
    print(f"Loading weights from {model_path}...")
    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
        print("Model loaded successfully.")
    except FileNotFoundError:
        print(f"ERROR: {model_path} not found. Please run training first.")
        return

    # 3. Evaluate
    evaluate_split(model, train_loader, device, "TRAINING")
    evaluate_split(model, test_loader, device, "TESTING")

if __name__ == "__main__":
    main()
