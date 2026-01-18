from main.bace_dataset import BACEDataset

dataset = BACEDataset("data/sample_smiles.csv")

print("Dataset size:", len(dataset))

graph = dataset[3]
print(graph)
print("Node features:", graph.x.shape)
print("Edge index:", graph.edge_index.shape)
print("Label:", graph.y)

"""Sample output:
Dataset size: 7
Data(x=[32, 6], edge_index=[2, 70], edge_attr=[70, 3], y=[1]) =>corresponds to G=(V,E,X,Y)
Node features: torch.Size([32, 6])
Edge index: torch.Size([2, 70])
Label: tensor([1.])

"""
