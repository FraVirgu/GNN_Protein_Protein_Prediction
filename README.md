# Drug-Drug Interaction Prediction using Graph Neural Networks

A comprehensive system for predicting potential drug-drug interactions using Graph Neural Networks (GNNs). This project leverages molecular graph representations and advanced deep learning techniques to identify interactions between drug pairs.

## Overview

This project aims to predict potential interactions between different drugs using Graph Neural Networks (GNNs). Since drugs can naturally be represented as molecular graphs, GNNs are ideally suited for capturing the structural relationships between atoms and bonds that determine drug interactions. The system converts drug SMILES representations into molecular graphs and uses GCN layers to learn drug embeddings for interaction prediction.

## Key Features

- **Graph Neural Network Architecture**: Utilizes Graph Convolutional Networks (GCN) to process molecular structures
- **Molecular Graph Representation**: Converts SMILES to molecular graphs with 17 atom-level features
- **Dual-Drug Encoder**: Processes two drugs through shared GCN encoders with mean pooling
- **Scalable Architecture**: Adaptive model sizing based on training dataset size
- **GPU Optimization**: Efficient batch processing with padded adjacency matrices
- **Interaction Probability**: Output sigmoid layer provides confidence scores for interactions
- **Comprehensive Dataset**: Built on TDC DrugBank dataset with both positive and generated negative samples

## Technical Architecture

### 1. Graph Representation

Each drug is converted from its SMILES representation into a molecular graph:
- **Nodes**: Atoms
- **Edges**: Chemical bonds (undirected)
- **Adjacency Matrix (A)**: N × N matrix representing bond connections
- **Feature Matrix (X)**: N × F matrix with atomic features

### 2. Atomic Features (17 per atom)

1. Atomic number
2. Degree
3. Total number of hydrogens
4. Aromaticity
5. Formal charge
6. Total valence
7. Explicit valence
8. Implicit valence
9. Radical electrons
10. Ring membership
11. Ring size 3
12. Ring size 4
13. Ring size 5
14. Ring size 6
15. Hybridization
16. Chirality tag
17. Scaled atomic mass (mass × 0.01)

### 3. Graph Convolutional Network (GCN)

The GCN layer aggregates information from neighboring atoms using the formula:

```
X^(k+1) = Â X^(k) W^(k)
```

Where:
- **Â** = D^(-1/2) (A + I) D^(-1/2) (normalized adjacency matrix with self-loops)
- **W^(k)** = learnable weight matrix
- **D** = degree matrix

### 4. Model Architecture

**Input Layer:**
- Two drug molecular graphs (adjacency matrices + node features)

**Encoder Stage:**
- GCN Layer 1 + ReLU activation
- GCN Layer 2 + ReLU activation
- Mean pooling (readout layer) → generates drug embeddings

**Fusion & Classification:**
- Concatenate embeddings from both drugs
- Dense Layer (512 units) + ReLU
- Dense Layer (512 units) + ReLU
- Output sigmoid layer → interaction probability [0, 1]

## Training Configuration

- **Optimizer**: Stochastic Gradient Descent (mini-batch)
- **Learning Rate Schedule**: Cosine decay from 10^-1 to 10^-4
- **Encoder Dimensions**:
  - Training samples < 1000 → 128 units
  - Training samples ≥ 1000 → 512 units
- **Dense Layers**: Fixed at 512 units
- **GPU Optimization**: Adjacency matrices padded to match largest molecule

## Dataset

- **Source**: TDC DrugBank dataset
- **Positive Samples**: Known drug-drug interactions
- **Negative Samples**: Randomly generated drug pairs not in dataset
- **Limitation**: Some generated negatives may be unknown true interactions

## Results

Performance metrics for different maximum graph sizes (N = maximum atoms):

| Max Atoms (N) | Accuracy | Training Samples | Test Samples | Training Time |
|---|---|---|---|---|
| 50 | 72% | 6,235 | 1,784 | 27 minutes |
| 40 | 70% | 5,812 | 1,627 | 24 minutes |
| 30 | 60% | 3,981 | 1,125 | 15 minutes |

**Best performance achieved with N = 50 atoms**

## Project Structure

- `data_creation.py` - Dataset preparation and graph construction from SMILES
- `models.py` - GCN architecture and model definitions
- `train.py` - Training loop with learning rate scheduling
- `test.py` - Model evaluation and performance metrics
- `GNN_Protein_Protein_Prediction.pdf` - Detailed technical documentation

## Getting Started

### Prerequisites

- Python 3.8 or higher
- PyTorch (for deep learning)
- RDKit (for molecular graph conversion)
- Required dependencies (see Installation section)

### Installation

1. Clone the repository:
```bash
git clone https://github.com/FraVirgu/Drug-Drug-Interaction_final.git
cd Drug-Drug-Interaction_final
```

2. Install required packages:
```bash
pip install -r requirements.txt
```

### Usage

**Data Preparation:**
```bash
python data_creation.py
```

**Training the Model:**
```bash
python train.py
```

**Testing/Evaluation:**
```bash
python test.py
```

## Methodology Highlights

- **Molecular Representation**: SMILES → RDKit → Molecular Graph
- **Feature Engineering**: 17 comprehensive atomic features capturing chemical properties
- **Graph Convolution**: Neighborhood aggregation through normalized adjacency matrices
- **Embedding Learning**: Shared dual-encoder architecture for drug pair processing
- **Scalable Design**: Adaptive model sizing based on dataset characteristics

## Future Improvements

- Incorporate biological information (protein targets, pathway data)
- Implement Minimum Common Substructure (MCS) matching
- Include 3D molecular conformations
- Enhanced negative sampling strategies
- Cross-dataset validation

## Data Sources

This project incorporates data from:
- TDC (Therapeutics Data Commons) DrugBank dataset
- RDKit molecular toolkit
- Chemical research publications

## Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

This project is licensed under the [MIT License](LICENSE) - see the LICENSE file for details.

## Authors

- **FraVirgu** - Initial work and GNN implementation

## Acknowledgments

- TDC (Therapeutics Data Commons) for the DrugBank dataset
- RDKit community for molecular toolkit
- Thanks to all contributors and data sources
- Special thanks to the open-source community

## Contact

For questions or suggestions, please open an issue in the repository.

---

*Last updated: 2026-03-02*