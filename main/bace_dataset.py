import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from features.rdkit_graph import smiles_to_graph
from transformers import AutoTokenizer

class BACEDataset(Dataset):
    def __init__(self, csv_path, model_name="DeepChem/ChemBERTa-77M-MTR", max_length=128):
        """
        csv_path: path to CSV file with columns 'mol', 'Class', 'pIC50', and scalar features.
        model_name: Hugging Face model name for tokenizer.
        """
        self.data = pd.read_csv(csv_path)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.max_length = max_length
        self.heavy_atoms = {
            'C', 'N', 'O', 'S', 'P', 'F', 'Cl', 'Br', 'I',
            'c', 'n', 'o', 's', 'p'
        }

        # Select extra features
        # We now CALCULATE these on the fly to match predict.py workflow
        self.feature_names = [
            'MW', 'AlogP', 'HBA', 'HBD', 'RB', 'HeavyAtomCount', 
            'ChiralCenterCount', 'RingCount', 'PSA', 'Estate', 'MR', 'Polar'
        ]
        
        print("Calculating features for dataset using RDKit...")
        from rdkit import Chem
        from rdkit.Chem import Descriptors, Crippen, Lipinski
        from rdkit.Chem.EState import EState

        features_list = []
        valid_indices = []
        
        for i, smiles in enumerate(self.data["mol"]):
            try:
                mol = Chem.MolFromSmiles(smiles)
                if mol is None: continue
                
                # Exact same logic as predict.py
                mw = Descriptors.MolWt(mol)
                logp = Crippen.MolLogP(mol)
                hba = Lipinski.NumHAcceptors(mol)
                hbd = Lipinski.NumHDonors(mol)
                rb = Lipinski.NumRotatableBonds(mol)
                heavy = mol.GetNumHeavyAtoms()
                # Version safe chiral check
                if 'includeStereo' in Chem.FindMolChiralCenters.__code__.co_varnames:
                     chiral = len(Chem.FindMolChiralCenters(mol, includeStereo=True))
                else:
                     chiral = len(Chem.FindMolChiralCenters(mol))
                
                ring = Lipinski.RingCount(mol)
                psa = Descriptors.TPSA(mol)
                mr = Crippen.MolMR(mol)
                estate = Descriptors.MaxAbsEStateIndex(mol) 
                polar = psa 
                
                feats = [mw, logp, hba, hbd, rb, heavy, chiral, ring, psa, estate, mr, polar]
                features_list.append(feats)
                valid_indices.append(i)
            except Exception as e:
                print(f"Error calculating features for {smiles}: {e}")

        # Convert to tensor
        self.features = torch.tensor(features_list, dtype=torch.float32)
        
        # Filter data to only valid molecules
        self.data = self.data.iloc[valid_indices].reset_index(drop=True)
        
        # Z-score normalization using THIS dataset's stats
        mean = self.features.mean(dim=0)
        std = self.features.std(dim=0)
        self.features = (self.features - mean) / (std + 1e-6)
        
        print(f"Calculated features for {len(self.features)} molecules.")
        print(f"Stats - Mean: {mean[:3]}... Std: {std[:3]}...") # Debug info
        
        self.num_extra_features = self.features.shape[1]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        smiles = row["mol"] # Main task: Renamed 'smiles' to 'mol'
        
        # Targets
        y_class = torch.tensor(row["Class"], dtype=torch.float32).unsqueeze(0) # (1,)
        y_reg = torch.tensor(row["pIC50"], dtype=torch.float32).unsqueeze(0)   # (1,)

        # 1. Graph Construction
        # Note: smiles_to_graph usually takes label, but we handle labels separately now.
        # We pass 0 as dummy label since we attach real labels to the dict output.
        graph = smiles_to_graph(smiles, 0)
        
        if graph is None:
             # Return None or handle error. For now, skipping/error is acceptable but PyTorch requires a return.
             # Ideally we filter valid graphs at init. For now, let's assume validity.
             raise ValueError(f"Invalid SMILES at index {idx}: {smiles}")

        # 2. Tokenization
        inputs = self.tokenizer(
            smiles, 
            return_tensors="pt", 
            add_special_tokens=True,
            padding='max_length',
            max_length=self.max_length,
            truncation=True
        )
        input_ids = inputs["input_ids"] 
        attention_mask = inputs["attention_mask"]

        # 3. Heavy Atom Mask (Alignment)
        # Re-using the alignment logic
        tokens = self.tokenizer.convert_ids_to_tokens(input_ids.squeeze(0))
        heavy_atom_indices = []
        for i, token in enumerate(tokens):
            if token is None: continue 
            clean_token = token.replace('Ġ', '')
            if clean_token in self.heavy_atoms:
                heavy_atom_indices.append(i)
        
        # Alignment Validation (Optional: make this soft to prevent crashes on edge cases)
        num_graph_nodes = graph.x.shape[0] 
        # If mismatch, we can mask out graph features or truncate.
        # Currently, we maintain strict check or simple truncation if list is too long.
        if len(heavy_atom_indices) > num_graph_nodes:
             heavy_atom_indices = heavy_atom_indices[:num_graph_nodes]
        
        heavy_atom_mask = torch.zeros_like(input_ids)
        heavy_atom_mask[0, heavy_atom_indices] = 1 # Mark heavy atom positions

        # 4. Extra Features
        extra_feats = self.features[idx]

        # 5. Pack everything
        # We return a dict-like object. PyG loader handles graph attributes.
        # But standard DataLoader expects dicts or tuples.
        # We attach standard tensors to the graph object to use PyG's batching for 'x', 'edge_index' etc.
        # And let PyG collate the rest.
        
        graph.input_ids = input_ids
        graph.attention_mask = attention_mask
        graph.heavy_atom_mask = heavy_atom_mask
        graph.extra_features = extra_feats.unsqueeze(0) # (1, num_features)
        
        graph.y_class = y_class
        graph.y_reg = y_reg
        
        return graph
