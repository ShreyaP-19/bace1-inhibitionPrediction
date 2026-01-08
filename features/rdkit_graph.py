from rdkit import Chem
import torch
from torch_geometric.data import Data

def atom_features(atom):
    #convert an RDKit atom object into a numerical feature vector
    return [
        atom.GetAtomicNum(),            # Atomic number (C=6, O=8, etc.)
        atom.GetDegree(),               # Number of neighbors
        atom.GetFormalCharge(),         # Formal charge
        atom.GetHybridization().real,   # sp, sp2, sp3 as numbers
        atom.GetIsAromatic(),           # Aromatic or not (0/1)
        atom.IsInRing()                 # Ring membership (0/1)
    ]
def bond_features(bond):
    #convert an RDKit bond object into numerical feature vector
    return [
        bond.GetBondTypeAsDouble(),     # 1, 2, 3, aromatic=1.5
        bond.GetIsConjugated(),         # Conjugation
        bond.IsInRing()                 # Ring bond or not
    ]


def smiles_to_graph(smiles, label=None):
    """
    Convert SMILES string into a PyTorch Geometric graph
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    node_features = []
    for atom in mol.GetAtoms():
        node_features.append(atom_features(atom))
    x = torch.tensor(node_features, dtype=torch.float)
    
    edge_index = []
    edge_attr = []
    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        # Undirected graph → add both directions
        edge_index.append([i, j])
        edge_index.append([j, i]) 

        bond_feat = bond_features(bond)
        edge_attr.append(bond_feat)
        edge_attr.append(bond_feat)  # Both directions

    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    edge_attr = torch.tensor(edge_attr, dtype=torch.float)
    
    if label is not None:
        y = torch.tensor([label], dtype=torch.float)
        return Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)
    
    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)