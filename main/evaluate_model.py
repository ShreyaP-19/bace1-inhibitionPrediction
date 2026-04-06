import torch
from torch_geometric.loader import DataLoader
from main.bace_dataset import BACEDataset
from models.gnn import GCN
from sklearn.metrics import roc_auc_score, mean_squared_error, accuracy_score, roc_curve, confusion_matrix
import numpy as np
import sys
import os
import matplotlib.pyplot as plt

def evaluate_split(model, loader, device, split_name):
    model.eval()
    
    all_probs = []
    all_preds_class = []
    all_labels_class = []
    
    all_preds_reg = []
    all_labels_reg = []

    print(f"\nProcessing {split_name} set...")

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

            # Convert logits → probabilities
            probs = torch.sigmoid(out_class).view(-1)

            # Classification
            preds_class = (probs > 0.5).float()

            all_probs.extend(probs.cpu().numpy())
            all_preds_class.extend(preds_class.cpu().numpy())
            all_labels_class.extend(batch.y_class.view(-1).cpu().numpy())

            # Regression
            all_preds_reg.extend(out_reg.view(-1).cpu().numpy())
            all_labels_reg.extend(batch.y_reg.view(-1).cpu().numpy())

    # ✅ Metrics
    accuracy = accuracy_score(all_labels_class, all_preds_class)

    try:
        auc = roc_auc_score(all_labels_class, all_probs)
    except:
        auc = 0.0

    rmse = np.sqrt(mean_squared_error(all_labels_reg, all_preds_reg))

    # ✅ Print Results
    print("-" * 40)
    print(f"{split_name} RESULTS")
    print("-" * 40)
    print(f"Accuracy           : {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"ROC-AUC            : {auc:.4f}")
    print(f"Regression RMSE    : {rmse:.4f}")
    print("-" * 40)

    # ROC Curve
    fpr, tpr, thresholds = roc_curve(all_labels_class, all_probs)

    plt.figure()
    plt.plot(fpr, tpr, label=f"AUC = {auc:.4f}")
    plt.plot([0, 1], [0, 1], linestyle='--')  # diagonal line (random model)

    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC Curve - {split_name}")
    plt.legend(loc="lower right")

    # Save image
    plt.savefig(f"roc_curve_{split_name.lower()}.png")
    plt.close()

    # Confusion Matrix
    cm = confusion_matrix(all_labels_class, all_preds_class)

    plt.figure()
    plt.imshow(cm)

    plt.title(f"Confusion Matrix - {split_name}")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")

    # Add numbers inside the matrix
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, cm[i, j], ha="center", va="center")

    # Axis labels
    plt.xticks([0, 1])
    plt.yticks([0, 1])

    plt.colorbar()
    plt.tight_layout()

    # Save image
    plt.savefig(f"confusion_matrix_{split_name.lower()}.png")
    plt.close()

    return accuracy, auc, rmse


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load datasets
    train_dataset = BACEDataset("data/train.csv")
    test_dataset = BACEDataset("data/test.csv")

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    # Get dimensions from sample
    sample = train_dataset[0]
    input_dim = sample.x.shape[1]
    extra_dim = sample.extra_features.shape[1]

    # Load model
    model = GCN(
        input_dim=input_dim,
        hidden_dim=128,
        extra_features_dim=extra_dim
    ).to(device)

    model_path = "bace_model.pth"
    print(f"Loading model from {model_path}...")

    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
        print("Model loaded successfully.")
    except FileNotFoundError:
        print("ERROR: Model file not found.")
        return

    # Evaluate
    evaluate_split(model, train_loader, device, "TRAINING")
    evaluate_split(model, test_loader, device, "TESTING")


if __name__ == "__main__":
    main()