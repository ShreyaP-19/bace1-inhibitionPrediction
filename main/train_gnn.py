import torch
from torch_geometric.loader import DataLoader
from main.bace_dataset import BACEDataset
from models.gnn import GCN

# Point to correct csv path if needed, assuming user has it
dataset = BACEDataset("data/bace.csv") 

loader = DataLoader(dataset, batch_size=4, shuffle=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Updated GCN init
model = GCN(input_dim=6, hidden_dim=64, transformer_name="DeepChem/ChemBERTa-77M-MTR").to(device)
criterion = torch.nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

for epoch in range(1, 51):
    model.train()
    total_loss = 0

    for batch_data in loader:
        batch_data = batch_data.to(device)

        optimizer.zero_grad()

        # Pass all required inputs
        output = model(
            x=batch_data.x, 
            edge_index=batch_data.edge_index, 
            batch=batch_data.batch,
            input_ids=batch_data.input_ids,
            attention_mask=batch_data.attention_mask,
            heavy_atom_mask=batch_data.heavy_atom_mask
        )
        loss = criterion(output.view(-1), batch_data.y)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(loader)
    print(f"Epoch {epoch:03d} | Avg Loss: {avg_loss:.4f}")
