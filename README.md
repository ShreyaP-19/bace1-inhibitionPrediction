
# 🧬 BACE-1 Inhibitor Prediction using Multimodal Deep Learning

# 📌 Overview
This project presents a hybrid deep learning framework for predicting BACE-1 inhibitor activity and binding strength, a crucial task in early-stage Alzheimer's drug discovery. The model integrates **sequence-based (SMILES)** and **structure-based (molecular graph)** representations to improve prediction accuracy and robustness.

The system combines a **pre-trained chemical language model (ChemBERTa-2)** with a **Graph Neural Network (GNN)** using a multimodal fusion strategy, followed by a **dual-task learning approach**.

---

## 🚀 Key Features
- 🔬 **Multimodal Representation**
  - SMILES sequences (Transformer-based encoding)
  - Molecular graphs (GNN-based encoding)

- 🧠 **Hybrid Deep Learning Model**
  - ChemBERTa-2 for sequence understanding
  - Graph Neural Networks for structural learning

- 🔗 **Feature Fusion Mechanism**
  - Combines sequential and structural embeddings

- 🎯 **Dual-Task Learning**
  - Classification: BACE-1 inhibitor (Active / Inactive)
  - Regression: Binding affinity (pIC50 prediction)

---

## 📊 Performance
| Dataset | Accuracy | ROC-AUC | RMSE |
|--------|---------|--------|------|
| Training | 91.07% | 0.9605 | 0.7352 |
| Testing  | 82.84% | 0.8943 | 0.9133 |

---

## 🗂️ Dataset
- Source: **MoleculeNet (BACE dataset)**
- Total Molecules: **1513**
- Includes:
  - SMILES strings
  - Bioactivity labels
  - IC50 / pIC50 values

---

## ⚙️ Tech Stack
- **Programming Language:** Python
- **Libraries:**
  - PyTorch
  - PyTorch Geometric
  - Hugging Face Transformers
  - RDKit
  - Pandas, NumPy, Scikit-learn
- **Model Used:**
  - ChemBERTa-2 (pretrained)
  - Graph Neural Network (GNN)

---

## 🏗️ System Architecture
1. Data preprocessing using RDKit
2. Dual molecular representation (SMILES + Graph)
3. Feature extraction:
   - Transformer (SMILES)
   - GNN (Graph)
4. Feature fusion
5. Dual-task prediction head

