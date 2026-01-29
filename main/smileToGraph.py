from main.bace_dataset import BACEDataset

# dataset = BACEDataset("data/sample_smiles.csv")
from features.rdkit_graph import smiles_to_graph
graph = smiles_to_graph("C") # Methane
print(f"Methane edge_index shape: {graph.edge_index.shape}")
print(f"Methane edge_index: {graph.edge_index}")


# print("Dataset size:", len(dataset))
# graph = dataset[3]
# print(graph)
print("Node features:", graph.x.shape)
print("Edge index:", graph.edge_index.shape)

