import torch
from torch_geometric.loader import DataLoader
from main.bace_dataset import BACEDataset
from models.gnn import GCN
from sklearn.metrics import roc_auc_score, mean_squared_error
import numpy as np

# =========================
# 1. Dataset Loading
# =========================

print("Loading train/test datasets...")

train_dataset = BACEDataset("data/train.csv")
test_dataset = BACEDataset("data/test.csv")

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =========================
# 2. Model Initialization
# =========================

sample = train_dataset[0]
extra_dim = sample.extra_features.shape[1]

model = GCN(
    input_dim=sample.x.shape[1],
    hidden_dim=128,
    transformer_name="DeepChem/ChemBERTa-77M-MTR",
    extra_features_dim=extra_dim
).to(device)

# =========================
# 3. Loss Functions
# =========================

criterion_class = torch.nn.BCEWithLogitsLoss()
criterion_reg = torch.nn.MSELoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.0001,
    weight_decay=1e-5
)

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode='max',
    factor=0.5,
    patience=3
)

# =========================
# 4. Training Function
# =========================

def train(loader):

    model.train()

    total_loss = 0
    all_preds_class, all_labels_class = [], []
    all_preds_reg, all_labels_reg = [], []

    for batch in loader:

        batch = batch.to(device)
        optimizer.zero_grad()

        out_class, out_reg = model(
            x=batch.x,
            edge_index=batch.edge_index,
            batch=batch.batch,
            input_ids=batch.input_ids,
            attention_mask=batch.attention_mask,
            heavy_atom_mask=batch.heavy_atom_mask,
            extra_features=batch.extra_features
        )

        # classification loss
        loss_cls = criterion_class(
            out_class.view(-1),
            batch.y_class.view(-1)
        )

        # regression loss
        loss_reg = criterion_reg(
            out_reg.view(-1),
            batch.y_reg.view(-1)
        )

        # weighted multitask loss
        loss = 0.7 * loss_cls + 0.3 * loss_reg

        loss.backward()

        # gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        optimizer.step()

        total_loss += loss.item()

        # store predictions
        all_preds_class.extend(
            torch.sigmoid(out_class).view(-1).detach().cpu().numpy()
        )
        all_labels_class.extend(
            batch.y_class.view(-1).detach().cpu().numpy()
        )

        all_preds_reg.extend(
            out_reg.view(-1).detach().cpu().numpy()
        )
        all_labels_reg.extend(
            batch.y_reg.view(-1).detach().cpu().numpy()
        )

    # compute metrics
    try:
        auc = roc_auc_score(all_labels_class, all_preds_class)
    except:
        auc = 0.5

    rmse = np.sqrt(mean_squared_error(all_labels_reg, all_preds_reg))

    return total_loss / len(loader), auc, rmse


# =========================
# 5. Evaluation Function
# =========================

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

            all_preds_class.extend(
                torch.sigmoid(out_class).view(-1).cpu().numpy()
            )

            all_labels_class.extend(
                batch.y_class.view(-1).cpu().numpy()
            )

            all_preds_reg.extend(
                out_reg.view(-1).cpu().numpy()
            )

            all_labels_reg.extend(
                batch.y_reg.view(-1).cpu().numpy()
            )

    try:
        auc = roc_auc_score(all_labels_class, all_preds_class)
    except:
        auc = 0.5

    rmse = np.sqrt(mean_squared_error(all_labels_reg, all_preds_reg))

    return auc, rmse


# =========================
# 6. Training Loop
# =========================

print("Starting Multi-Task Training...")

best_auc = 0
epochs = 40

for epoch in range(1, epochs + 1):

    train_loss, train_auc, train_rmse = train(train_loader)

    test_auc, test_rmse = evaluate(test_loader)

    # update scheduler
    scheduler.step(test_auc)

    print(
        f"Epoch {epoch:03d} | "
        f"Loss: {train_loss:.4f} | "
        f"Train AUC: {train_auc:.4f} RMSE: {train_rmse:.4f} | "
        f"Test AUC: {test_auc:.4f} RMSE: {test_rmse:.4f}"
    )

    # save best model
    if test_auc > best_auc:
        best_auc = test_auc
        torch.save(model.state_dict(), "bace_model.pth")
        print("  -> Best Model Saved!")

print("Training Completed!")