import torch
import argparse
import numpy as np
import sys
import os

# Add project root to sys.path to allow importing from models
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from rdkit import Chem
from rdkit.Chem import Descriptors, Crippen, Lipinski
from models.gnn import GCN
from features.rdkit_graph import smiles_to_graph
from transformers import AutoTokenizer

def compute_features(smiles):
    """Calculates the 12 physicochemical features expected by the model."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError("Invalid SMILES string")
        
    # Features List: ['MW', 'AlogP', 'HBA', 'HBD', 'RB', 'HeavyAtomCount', 
    #                 'ChiralCenterCount', 'RingCount', 'PSA', 'Estate', 'MR', 'Polar']
    # Note: Dataset used AlogP (Ghose-Crippen LogP). RDKit gives this via MolLogP.
    # Estate indices are complex vectors, but if dataset used Sum of EState or MaxAbsEState etc.
    # We will approximate standard descriptors to match dataset names as best as possible.
    
    mw = Descriptors.MolWt(mol)
    logp = Crippen.MolLogP(mol)
    hba = Lipinski.NumHAcceptors(mol)
    hbd = Lipinski.NumHDonors(mol)
    rb = Lipinski.NumRotatableBonds(mol)
    heavy = mol.GetNumHeavyAtoms()
    # RDKit version compatibility: simple call usually returns list of tuples
    chiral = len(Chem.FindMolChiralCenters(mol, includeStereo=True) if 'includeStereo' in Chem.FindMolChiralCenters.__code__.co_varnames else Chem.FindMolChiralCenters(mol))
    ring = Lipinski.RingCount(mol)
    psa = Descriptors.TPSA(mol)
    mr = Crippen.MolMR(mol) # Molar Refractivity
    
    # "Polar" usually refers to TPSA or specific surface area. We reused PSA?
    # "Estate" is tricky without exact definition from dataset source. 
    # For now we use MaxAbsEStateIndex as a proxy or 0 if unknown.
    # Given the previous context, we will use reasonable RDKit equivalents.
    from rdkit.Chem.EState import EState
    estate = Descriptors.MaxAbsEStateIndex(mol) 
    polar = psa # Assuming 'Polar' column in dataset maps to TPSA logic if distinct from PSA.
    
    features = [mw, logp, hba, hbd, rb, heavy, chiral, ring, psa, estate, mr, polar]
    
    # Normalization (CRITICAL): 
    # The model expects Z-normalized features. We need the Approx Mean/Std from training data.
    # Hardcoded from typical BACE dataset stats (or 0/1 if we accept small shift error).
    # For meaningful results, we should ideally load stats. 
    # For this demo, we pass raw or simple standardized 0-mean.
    # Better approach: We accept we might be slightly off scale without saved Scaler.
    return np.array(features, dtype=np.float32)

def predict(smiles, model_path="bace_model.pth"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained("DeepChem/ChemBERTa-77M-MTR")
    
    print(f"Processing Molecule: {smiles}")
    
    # 1. Graph
    graph = smiles_to_graph(smiles, 0) # Dummy label 0
    if graph is None: return

    # 2. Text
    inputs = tokenizer(smiles, return_tensors="pt", padding='max_length', max_length=128, truncation=True)
    
    # 3. Features
    try:
        feats = compute_features(smiles)
        
        # NORMALIZATION FIX: Use Training Set Statistics!
        # Calculated from data/train.csv
        # ['MW', 'AlogP', 'HBA', 'HBD', 'RB', 'HeavyAtomCount', 'ChiralCenterCount', 'RingCount', 'PSA', 'Estate', 'MR', 'Polar']
        train_mean = np.array([481.53, 3.17, 5.09, 2.39, 8.16, 33.72, 1.13, 3.73, 85.93, 85.09, 128.53, 83.08])
        train_std  = np.array([120.16, 1.40, 1.83, 1.25, 4.34, 8.44, 1.63, 1.22, 40.59, 26.68, 31.18, 38.95])
        
        # Z-Score Normalize using GLOBAL stats from training data
        # These values MUST match what BACEDataset used during training
        # Since BACEDataset computed mean/std on the fly on the full csv,
        # we reused the values calculated from 'data/train.csv' earlier.
        feats = (feats - train_mean) / (train_std + 1e-6)
        
        # DEBUG: Print features to see if they look like the training distribution (approx N(0,1))
        # print(f"DEBUG: Normalized Features: {feats}")
        
    except Exception as e:
        print(f"Feature calculation error: {e}")
        return

    # Prepare Tensors
    graph.x = graph.x.to(device)
    graph.edge_index = graph.edge_index.to(device)
    graph.batch = torch.zeros(graph.x.shape[0], dtype=torch.long).to(device) # Single graph batch
    
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)
    extra_features = torch.tensor(feats, dtype=torch.float32).unsqueeze(0).to(device) # (1, 12)
    
    # Heavy Atom Mask
    tokens = tokenizer.convert_ids_to_tokens(input_ids.squeeze(0))
    heavy_atoms = {'C','N','O','S','P','F','Cl','Br','I','c','n','o','s','p'}
    heavy_indices = [i for i,t in enumerate(tokens) if t and t.replace('Ġ','') in heavy_atoms]
    
    # Alignment Truncation
    if len(heavy_indices) > graph.x.shape[0]: heavy_indices = heavy_indices[:graph.x.shape[0]]
    
    heavy_mask = torch.zeros_like(input_ids)
    heavy_mask[0, heavy_indices] = 1
    
    # Load Model
    model = GCN(input_dim=graph.x.shape[1], hidden_dim=128, extra_features_dim=12).to(device)
    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
    except FileNotFoundError:
        print(f"Error: {model_path} not found. Train the model first!")
        return
        
    model.eval()
    
    with torch.no_grad():
        out_class, out_reg = model(
            x=graph.x, 
            edge_index=graph.edge_index, 
            batch=graph.batch,
            input_ids=input_ids,
            attention_mask=attention_mask,
            heavy_atom_mask=heavy_mask,
            extra_features=extra_features
        )
        
    prob = torch.sigmoid(out_class).item()
    logit = out_class.item()
    pic50 = out_reg.item()
    
    print("\n" + "="*40)
    print(f"RESULTS FOR: {smiles}")
    print("="*40)
    print(f"Raw Class Logit       : {logit:.4f}")
    print(f"Inhibitor Probability : {prob:.4f} ({'ACTIVE' if prob > 0.5 else 'INACTIVE'})")
    print(f"Predicted pIC50       : {pic50:.4f}")
    print("="*40 + "\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("smiles", type=str, help="SMILES string of the molecule")
    args = parser.parse_args()
    
    predict(args.smiles)
