import torch
from torch_geometric.loader import DataLoader
from torch.utils.data import random_split
from main.bace_dataset import BACEDataset
from models.gnn import GCN
from sklearn.metrics import roc_auc_score, mean_squared_error
import numpy as np

# 1. Dataset & Splitting
# Load explicit train/test splits
print("Loading train/test datasets...")
train_dataset = BACEDataset("data/train.csv")
test_dataset = BACEDataset("data/test.csv")

# Loaders
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 2. Model Initialization
# Compute extra features dim from dataset
# Note: Since dataset is huge, we can peak one item or trust our known columns (12)
sample = train_dataset[0]
extra_dim = sample.extra_features.shape[1]

model = GCN(
    input_dim=sample.x.shape[1], # From Rdkit graph features
    hidden_dim=128, 
    transformer_name="DeepChem/ChemBERTa-77M-MTR",
    extra_features_dim=extra_dim
).to(device)

# 3. Loss Functions
criterion_class = torch.nn.BCEWithLogitsLoss()
criterion_reg = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

def train(loader):
    model.train()
    total_loss = 0
    all_preds_class, all_labels_class = [], []
    all_preds_reg, all_labels_reg = [], []

    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()

        # Reshape input_ids/mask for transformer [Batch, Seq]
        # PyG batching stacks them. We need to check if they are correctly stacked.
        # Since we padded in Dataset, 'input_ids' is [Batch*Seq], but we need [Batch, Seq]?
        # Actually PyG batching for custom tensor attributes concats them in dim 0.
        # So 'input_ids' becomes (N*Seq, ). We need to reshape.
        # BUT 'input_ids' from Dataset was (1, 128). PyG concat -> (Batch, 128). CORRECT.
        
        out_class, out_reg = model(
            x=batch.x, 
            edge_index=batch.edge_index, 
            batch=batch.batch,
            input_ids=batch.input_ids,
            attention_mask=batch.attention_mask,
            heavy_atom_mask=batch.heavy_atom_mask,
            extra_features=batch.extra_features
        )

        # Loss Calculation
        # Class Targets
        loss_cls = criterion_class(out_class.view(-1), batch.y_class.view(-1))
        # Regression Targets
        loss_reg = criterion_reg(out_reg.view(-1), batch.y_reg.view(-1))
        
        # Combined Loss
        loss = loss_cls + loss_reg
        
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        
        # Metrics Collection
        all_preds_class.extend(torch.sigmoid(out_class).view(-1).detach().cpu().numpy())
        all_labels_class.extend(batch.y_class.view(-1).detach().cpu().numpy())
        
        all_preds_reg.extend(out_reg.view(-1).detach().cpu().numpy())
        all_labels_reg.extend(batch.y_reg.view(-1).detach().cpu().numpy())

    # Epoch Metrics
    try:
        auc = roc_auc_score(all_labels_class, all_preds_class)
    except:
        auc = 0.5 # Handle single class batch edge case
        
    rmse = np.sqrt(mean_squared_error(all_labels_reg, all_preds_reg))
    
    return total_loss / len(loader), auc, rmse

def evaluate(loader):
    model.eval()
    all_preds_class, all_labels_class = [], []
    all_preds_reg, all_labels_reg = [], []
    
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
            
            all_preds_class.extend(torch.sigmoid(out_class).view(-1).cpu().numpy())
            all_labels_class.extend(batch.y_class.view(-1).cpu().numpy())
            all_preds_reg.extend(out_reg.view(-1).cpu().numpy())
            all_labels_reg.extend(batch.y_reg.view(-1).cpu().numpy())

    try:
        auc = roc_auc_score(all_labels_class, all_preds_class)
    except:
        auc = 0.5
        
    rmse = np.sqrt(mean_squared_error(all_labels_reg, all_preds_reg))
    return auc, rmse

# Training Loop
print("Starting Multi-Task Training...")
best_loss = float('inf')

for epoch in range(1, 21): # 20 Epochs
    train_loss, train_auc, train_rmse = train(train_loader)
    test_auc, test_rmse = evaluate(test_loader)
    
    print(f"Epoch {epoch:03d} | Loss: {train_loss:.4f} | "
          f"Train AUC: {train_auc:.4f} RMSE: {train_rmse:.4f} | "
          f"Test AUC: {test_auc:.4f} RMSE: {test_rmse:.4f}")
    
    # Save best model
    if train_loss < best_loss:
        best_loss = train_loss
        torch.save(model.state_dict(), "bace_model.pth")
        print("  -> Model Saved!")
