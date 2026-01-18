import torch
import pandas as pd
import os
import sys

# Add project root to path to allow importing 'features' and 'models'
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from main.bace_dataset import BACEDataset
from models.gnn import GCN
from torch_geometric.loader import DataLoader

def test_fusion():
    print("Testing MolPROP Fusion Implementation...")
    
    # 1. Create Dummy Data
    dummy_csv = "data/verify_dummy.csv"
    os.makedirs("data", exist_ok=True)
    
    # SMILES samples:
    # 1. Simple: C
    # 2. Ring + Heavy atoms: c1ccccc1O (Phenol)
    # 3. Explicit H (should be ignored): [H]C([H])([H])[H] -> C
    # 4. Halogen: CC(Cl)Br
    df = pd.DataFrame({
        "smiles": ["C", "c1ccccc1O", "CC(Cl)Br"],
        "label": [0, 1, 0]
    })
    df.to_csv(dummy_csv, index=False)
    
    # 2. Load Dataset
    print(f"Loading dataset from {dummy_csv}...")
    try:
        dataset = BACEDataset(dummy_csv)
    except Exception as e:
        print(f"FAILED to load dataset: {e}")
        return

    # Check length
    print(f"Dataset length: {len(dataset)}")
    
    # Check item 1 (Phenol: c1ccccc1O -> 6 C + 1 O = 7 heavy atoms)
    # Graph nodes should be 7.
    data = dataset[1]
    print(f"Sample 1 (c1ccccc1O):")
    print(f"  - Graph Nodes: {data.x.shape[0]}")
    print(f"  - Input IDs shape: {data.input_ids.shape}")
    print(f"  - Heavy Atom Mask shape: {data.heavy_atom_mask.shape}")
    print(f"  - Heavy Atom Mask sum: {data.heavy_atom_mask.sum().item()}")
    
    if data.x.shape[0] != data.heavy_atom_mask.sum().item():
        print("FAIL: Mismatch between graph nodes and heavy atom tokens!")
        # Print tokens to debug
        tokenizer = dataset.tokenizer
        tokens = tokenizer.convert_ids_to_tokens(data.input_ids.squeeze(0))
        print(f"  - Tokens: {tokens}")
        return
    else:
        print("PASS: Alignment verified.")

    # 3. Initialize Model
    print("Initializing GCN Model...")
    device = torch.device("cpu") # Test on CPU
    model = GCN(input_dim=6, hidden_dim=32).to(device) # input_dim=6 from rdkit_graph
    
    # 4. Forward Pass
    print("Running Forward Pass...")
    loader = DataLoader(dataset, batch_size=2)
    batch = next(iter(loader))
    batch = batch.to(device)
    
    try:
        output = model(
            x=batch.x,
            edge_index=batch.edge_index,
            batch=batch.batch,
            input_ids=batch.input_ids,
            attention_mask=batch.attention_mask,
            heavy_atom_mask=batch.heavy_atom_mask
        )
        print(f"Output shape: {output.shape}")
        if output.shape == (2, 1):
             print("PASS: Forward pass successful.")
        else:
             print(f"FAIL: Unexpected output shape {output.shape}")
             
    except Exception as e:
        print(f"FAIL: Forward pass error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_fusion()
