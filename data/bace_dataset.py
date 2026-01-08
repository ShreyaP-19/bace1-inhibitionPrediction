import pandas as pd
from torch.utils.data import Dataset
from features.rdkit_graph import smiles_to_graph

class BACEDataset(Dataset):
    def __init__(self, csv_path):
        """
        csv_path: path to CSV file with columns [smiles, label]
        """
        self.data = pd.read_csv(csv_path)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        smiles = row["smiles"]
        label = row["label"]

        graph = smiles_to_graph(smiles, label)

        if graph is None:
            raise ValueError(f"Invalid SMILES at index {idx}: {smiles}")

        return graph
