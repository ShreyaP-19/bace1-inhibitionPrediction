import torch
from torch_geometric.loader import DataLoader
from data.bace_dataset import BACEDataset
from models.gnn import GCN

dataset = BACEDataset("data/sample_smiles.csv")

loader = DataLoader(dataset, batch_size=4, shuffle=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = GCN(input_dim=6, hidden_dim=64).to(device)
criterion = torch.nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

for epoch in range(1, 51):
    model.train()
    total_loss = 0

    for batch in loader:
        batch = batch.to(device)

        optimizer.zero_grad()

        output = model(batch.x, batch.edge_index, batch.batch)
        loss = criterion(output.view(-1), batch.y)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(loader)
    print(f"Epoch {epoch:03d} | Avg Loss: {avg_loss:.4f}")
