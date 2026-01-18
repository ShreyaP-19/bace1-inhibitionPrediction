import pandas as pd
import torch
from torch.utils.data import Dataset
from features.rdkit_graph import smiles_to_graph
from transformers import AutoTokenizer

class BACEDataset(Dataset):
    def __init__(self, csv_path, model_name="DeepChem/ChemBERTa-77M-MTR"):
        """
        csv_path: path to CSV file with columns [smiles, label]
        model_name: Hugging Face model name for tokenizer
        """
        self.data = pd.read_csv(csv_path)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.heavy_atoms = {
            'C', 'N', 'O', 'S', 'P', 'F', 'Cl', 'Br', 'I',
            'c', 'n', 'o', 's', 'p' 
        } # Common heavy atoms in drug-like molecules + aromatic forms

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        smiles = row["smiles"]
        label = row["label"]

        # 1. Graph Construction
        graph = smiles_to_graph(smiles, label)
        if graph is None:
             raise ValueError(f"Invalid SMILES at index {idx}: {smiles}")

        # 2. Tokenization & Alignment
        # Pad to max_length to allow simple batching (PyG will stack [batch, max_len])
        inputs = self.tokenizer(
            smiles, 
            return_tensors="pt", 
            add_special_tokens=True,
            padding='max_length',
            max_length=128,
            truncation=True
        )
        # Keep as (1, max_len) for PyG diagonal stacking (actually simple concat for custom attrs)
        input_ids = inputs["input_ids"] 
        attention_mask = inputs["attention_mask"]

        tokens = self.tokenizer.convert_ids_to_tokens(input_ids.squeeze(0))
        
        heavy_atom_indices = []
        for i, token in enumerate(tokens):
            if token is None: continue 
            clean_token = token.replace('Ġ', '')
            if clean_token in self.heavy_atoms:
                heavy_atom_indices.append(i)
        
        # 3. Alignment Validation
        num_graph_nodes = graph.x.shape[0] 
        if len(heavy_atom_indices) != num_graph_nodes:
            raise ValueError(
                f"Alignment mismatch for SMILES: {smiles}\n"
                f"Graph nodes: {num_graph_nodes}, "
                f"Identify tokens ({len(heavy_atom_indices)}): { [tokens[i] for i in heavy_atom_indices] }"
            )

        # 4. Create Heavy Atom Mask
        # (1, seq_len)
        heavy_atom_mask = torch.zeros_like(input_ids)
        heavy_atom_mask[0, heavy_atom_indices] = 1

        # Add to graph object
        graph.input_ids = input_ids
        graph.attention_mask = attention_mask
        graph.heavy_atom_mask = heavy_atom_mask
        
        return graph
