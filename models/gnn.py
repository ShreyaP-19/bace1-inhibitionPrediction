import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool
from transformers import AutoModel

class GCN(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, transformer_name="DeepChem/ChemBERTa-77M-MTR", transformer_dim=384, extra_features_dim=12):
        super(GCN, self).__init__()

        # 1. Transformer Branch
        self.transformer = AutoModel.from_pretrained(transformer_name)
        # Project Transformer embeddings to match GNN feature size
        self.chem_proj = torch.nn.Linear(transformer_dim, hidden_dim)

        # 2. Graph Branch
        # Input dim is now original graph features + projected transformer features
        self.conv1 = GCNConv(input_dim + hidden_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim * 2) # Increase dim for capacity

        # 3. Extra Features Branch (MLP)
        self.feature_mlp = torch.nn.Sequential(
            torch.nn.Linear(extra_features_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.ReLU()
        )

        # 4. Heads
        # Fusion Dimension: (Graph Pooled) + (Features MLP)
        # Graph Pooled dim = hidden_dim * 2
        # Features dim = hidden_dim
        # Total = hidden_dim * 3
        fusion_dim = (hidden_dim * 2) + hidden_dim
        
        self.classification_head = torch.nn.Sequential(
            torch.nn.Linear(fusion_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.2),
            torch.nn.Linear(hidden_dim, 1)
        )
        
        self.regression_head = torch.nn.Sequential(
            torch.nn.Linear(fusion_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.2),
            torch.nn.Linear(hidden_dim, 1)
        )

    def forward(self, x, edge_index, batch, input_ids, attention_mask, heavy_atom_mask, extra_features):
        """
        x: Node features [N_nodes, input_dim]
        edge_index: [2, N_edges]
        batch: [N_nodes]
        input_ids: [Batch, Seq_Len]
        attention_mask: [Batch, Seq_Len]
        heavy_atom_mask: [Batch, Seq_Len]
        extra_features: [Batch, extra_features_dim]
        """
        
        # --- BRANCH 1: TRANSFORMER ---
        # First, pass input_ids and attention_mask through the transformer
        outputs = self.transformer(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden_state = outputs.last_hidden_state # [Batch, Seq_Len, 384]

        # Extract Heavy Atom Embeddings for Node Fusion
        # Flatten batch/seq dims to align with PyG stacked nodes
        flat_hidden = last_hidden_state.view(-1, last_hidden_state.size(-1)) 
        flat_mask = heavy_atom_mask.view(-1)
        selected_chem_features = flat_hidden[flat_mask.bool()]
        
        # Project
        projected_chem = self.chem_proj(selected_chem_features) # [N_nodes, hidden_dim]
        
        # Safe concatenation (Handle mismatch by PAD/TRUNCATE Features, NEVER slice Graph Nodes)
        if projected_chem.shape[0] != x.shape[0]:
             if projected_chem.shape[0] > x.shape[0]:
                 # Truncate features
                 projected_chem = projected_chem[:x.shape[0]]
             else:
                 # Pad features with zeros
                 padding = torch.zeros((x.shape[0] - projected_chem.shape[0], projected_chem.shape[1]), device=x.device)
                 projected_chem = torch.cat([projected_chem, padding], dim=0)

        # Fuse at Node Level
        x_fused_node = torch.cat([x, projected_chem], dim=-1)

        # --- BRANCH 2: GRAPH GNN ---
        x_gnn = self.conv1(x_fused_node, edge_index)
        x_gnn = F.relu(x_gnn)
        x_gnn = self.conv2(x_gnn, edge_index)
        x_gnn = F.relu(x_gnn)

        # Graph Pooling (Readout)
        x_graph_pooled = global_mean_pool(x_gnn, batch) # [Batch, hidden_dim * 2]

        # --- BRANCH 3: EXTRA FEATURES ---
        x_feats = self.feature_mlp(extra_features) # [Batch, hidden_dim]

        # --- FINAL FUSION ---
        # Concatenate Graph Representation + Scalar Features Representation
        x_final = torch.cat([x_graph_pooled, x_feats], dim=1) # [Batch, hidden*3]

        # --- OUTPUTS ---
        out_class = self.classification_head(x_final)
        out_reg = self.regression_head(x_final)

        return out_class, out_reg

