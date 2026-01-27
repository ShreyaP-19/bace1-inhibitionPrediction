import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
import pandas as pd
import torch
from torch_geometric.loader import DataLoader

from features.rdkit_graph import smiles_to_graph
from models.gcn import GCN

# -------------------------
# 1. Load dataset
# -------------------------
df = pd.read_csv("data/bace.csv")

graphs = []
for _, row in df.iterrows():
    graph = smiles_to_graph(row["smiles"], row["label"])
    if graph is not None:
        graphs.append(graph)

print(f"Total graphs loaded: {len(graphs)}")

# -------------------------
# 2. DataLoader
# -------------------------
train_loader = DataLoader(
    graphs,
    batch_size=32,
    shuffle=True
)

# -------------------------
# 3. Model setup
# -------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = GCN().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = torch.nn.CrossEntropyLoss()

# -------------------------
# 4. Training loop
# -------------------------
epochs = 20

for epoch in range(epochs):
    model.train()
    total_loss = 0

    for batch in train_loader:
        batch = batch.to(device)

        optimizer.zero_grad()

        out = model(batch.x, batch.edge_index, batch.batch)
        loss = criterion(out, batch.y)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch+1}/{epochs} | Loss: {total_loss:.4f}")

# -------------------------
# 5. Save model
# -------------------------
torch.save(model.state_dict(), "models/bace1_model.pth")
print("✅ GCN model trained and saved successfully")
