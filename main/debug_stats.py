
import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors, Crippen, Lipinski
from rdkit.Chem.EState import EState

def calculate_stats():
    # Load Train Data
    print("Loading data/train.csv...")
    try:
        data = pd.read_csv("data/train.csv")
    except FileNotFoundError:
        print("Error: data/train.csv not found.")
        return

    features_list = []
    
    print(f"Calculating features for {len(data)} molecules...")
    for smiles in data["mol"]:
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None: continue
            
            # Exact same logic as bace_dataset.py
            mw = Descriptors.MolWt(mol)
            logp = Crippen.MolLogP(mol)
            hba = Lipinski.NumHAcceptors(mol)
            hbd = Lipinski.NumHDonors(mol)
            rb = Lipinski.NumRotatableBonds(mol)
            heavy = mol.GetNumHeavyAtoms()
            
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
        except Exception as e:
            pass

    features = np.array(features_list)
    mean = np.mean(features, axis=0)
    std = np.std(features, axis=0)
    
    import json
    with open("stats.json", "w") as f:
        json.dump({"mean": mean.tolist(), "std": std.tolist()}, f)
    
    print("Stats saved to stats.json")

if __name__ == "__main__":
    calculate_stats()
