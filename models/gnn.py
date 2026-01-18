import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool
from transformers import AutoModel

class GCN(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, transformer_name="DeepChem/ChemBERTa-77M-MTR", transformer_dim=384):
        super(GCN, self).__init__()

        # Transformer model
        self.transformer = AutoModel.from_pretrained(transformer_name)
        
        # Projection for Transformer embeddings to match GNN feature size or simply reduce
        # Here we project to hidden_dim to make it compatible/manageable
        self.chem_proj = torch.nn.Linear(transformer_dim, hidden_dim)

        # Graph convolution layers
        # Input dim is now original graph features + projected transformer features
        self.conv1 = GCNConv(input_dim + hidden_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)

        # Final classifier
        self.lin = torch.nn.Linear(hidden_dim, 1)

    def forward(self, x, edge_index, batch, input_ids, attention_mask, heavy_atom_mask):
        # 1. Transformer Pass
        # We need to reshape/process inputs if they come from PyG Batch
        # PyG Batch usually stacks graphs. For text, we need [batch_size, seq_len]
        # BUT 'batch' in PyG is [num_nodes]. We have token inputs per graph.
        # We assume the DataLoader collates them correctly.
        # However, default PyG DataBatching for 'input_ids' (since it's a field in Data) 
        # might just stack them. 
        # CAUTION: 'input_ids' in Data object (from bace_dataset) was 1D tensor?
        # In bace_dataset check lines: 
        # inputs["input_ids"].squeeze(0) -> so each graph has (seq_len,)
        # When batched by PyG, distinct custom attributes are stacked.
        # So input_ids will be [batch_size * seq_len] if we are not careful?
        # NO, PyG default collate checks dimension.
        # Let's verify how we want to pass them.
        # For simplicity, if we rely on PyG batching, input_ids will be concatenated.
        # But BERT expects [batch, seq_len].
        # We might need to unbatch or ensure 'input_ids' is [batch, seq_len] in the batch object.
        # Actually, standard PyG Batch preserves shape if we treat it as a custom attribute?
        # Let's assume input_ids is [batch_size, seq_len].
        
        # To be safe with PyG batching behavior, we usually need to view/reshape if it flattens.
        # However, for this implementation let's rely on the passed arguments having correct shape 
        # or checking the shape.
        
        # Extract Graph-level BERT inputs
        # If input_ids is (num_graphs * seq_len), we need to know seq_len or batch_size.
        # But 'ptr' in Batch is useful.
        
        # Wait, PyG Batch simply concatenates attributes across the batch dimension (dim 0)
        # unless documented otherwise.
        # If each graph has input_ids [seq_len], then Batch has [num_graphs * seq_len]?
        # Or [num_graphs, seq_len]?
        # By default PyG concatenates along the first dimension.
        # So correct shape requires 'input_ids' to be [1, seq_len] in the Dataset return 
        # if we want [num_graphs, seq_len]. 
        # In bace_dataset I did: `input_ids = inputs["input_ids"].squeeze(0)` which is `(seq_len,)`.
        # So batching will yield `(sum_seq_len,)` which is WRONG for BERT input.
        
        # FIXING ASSUMPTION:
        # We need to reshape input_ids back to [batch_size, seq_len].
        # But seq_len might differ! BERT requires padding if we batch.
        # ChemBERTa on SMILES has variable length.
        # If we didn't pad in dataset, we can't easily reshape.
        
        # SOLUTION: We should Pad in Dataset.
        # Use tokenizer(..., padding='max_length', max_length=128, truncation=True)
        pass # Placeholder for thought trace
        
        # Back to code:
        outputs = self.transformer(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden_state = outputs.last_hidden_state # [batch_size, seq_len, 384]

        # 2. Extract Heavy Atom Embeddings
        # mask is [batch_size, seq_len]. 1 for heavy atoms.
        # We want to flatten these selected tokens to align with 'x' (which is all nodes stacked)
        
        # Flatten batch/seq dims
        flat_hidden = last_hidden_state.view(-1, last_hidden_state.size(-1)) # [N_tokens_total, 384]
        flat_mask = heavy_atom_mask.view(-1) # [N_tokens_total]
        
        # Bool selection
        selected_chem_features = flat_hidden[flat_mask.bool()] # [N_heavy_atoms_total, 384]
        
        # Verify alignment strictly?
        if selected_chem_features.shape[0] != x.shape[0]:
            raise RuntimeError(f"Fusion Alignment Error: selected {selected_chem_features.shape[0]} atoms from BERT, but Graph has {x.shape[0]} nodes.")

        # 3. Fusion
        projected_chem = self.chem_proj(selected_chem_features) # [N_nodes, hidden_dim]
        
        # Concat
        x = torch.cat([x, projected_chem], dim=-1)

        # 4. GNN Layers
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        x = F.relu(x)

        # Graph-level pooling
        x = global_mean_pool(x, batch)

        # Classification
        x = self.lin(x)

        return x

