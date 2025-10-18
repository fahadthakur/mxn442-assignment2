# Fake News Classifier: MXN442 Assignment 2

**Multi-Modal Fake News Detection using Machine Learning and Deep Learning**

## Overview

This repository contains the implementation of a comprehensive fake news detection system that combines traditional machine learning approaches with modern deep learning techniques. The project reproduces and extends research on fake news classification using multiple datasets and various modeling approaches.

## Project Structure

```
.
├── fake_news_classifier.ipynb    # Main implementation notebook
├── Dataset/
│   ├── Kaggle.csv                # Kaggle fake news dataset
│   ├── Mcintire_DataSet.csv      # McIntire fake news dataset
│   └── linguistic_features/      # Pre-computed linguistic features
│       ├── arrays_kaggle.xlsx
│       └── arrays_Mcintire.xlsx
├── all_models/
│   └── best_model.keras          # Saved RoBERTa+LF model weights
├── LICENSE
├── README.md
└── Spec/                         # Assignment specifications
```

## Getting Started

### Prerequisites

Before running the notebook, you need to pull the large dataset and model files using Git LFS:

```bash
git lfs pull
```

This will download:
- `Dataset/Kaggle.csv`
- `Dataset/Mcintire_DataSet.csv`
- `Dataset/linguistic_features/*.xlsx`
- `all_models/best_model.keras`

### Installation

The notebook automatically installs required packages on first run. Dependencies include:

- **Core Libraries**: NumPy, Pandas, Matplotlib, Seaborn
- **Machine Learning**: scikit-learn, XGBoost
- **NLP**: NLTK, Gensim
- **Deep Learning**: TensorFlow, Transformers (Hugging Face)
- **Utilities**: openpyxl, statsmodels

### Running the Notebook

1. Ensure Git LFS files are pulled: `git lfs pull`
2. Open `fake_news_classifier.ipynb` in Jupyter Notebook or VS Code
3. Run cells sequentially - the first cell will handle dependency installation
4. All models will be trained automatically (or loaded from saved checkpoints if available)


## Methodology

### Data Preprocessing

1. **Text Cleaning**:
   - Whitespace normalization and lowercasing
   - URL and email/mention removal
   - Special character and punctuation removal
   - Tokenization using NLTK

2. **Text Processing**:
   - Stop word removal
   - Lemmatization using WordNet
   - Feature standardization using StandardScaler

### Feature Engineering

Three feature representation approaches:

1. **Word2Vec Embeddings**:
   - Vector size: 100
   - Window size: 5
   - Mean pooling of word vectors
   - Combined with linguistic features

2. **TF-IDF Embeddings**:
   - Captures term importance
   - Sparse representation
   - Combined with linguistic features

3. **Transformer Embeddings** (GenAI Extension):
   - RoBERTa-base backbone
   - Max sequence length: 510 tokens
   - Late fusion with linguistic features

### Models Implemented

#### Traditional Machine Learning
1. **Logistic Regression**: Linear baseline classifier
2. **Support Vector Machine (SVM)**: Non-linear decision boundaries with RBF kernel
3. **XGBoost**: Gradient boosting ensemble

Each model trained on both Word2Vec and TF-IDF feature representations.

#### Deep Learning (GenAI Extension)
**RoBERTa + Linguistic Features (Late Fusion)**:
- Pre-trained RoBERTa-base transformer backbone
- Two-tower architecture:
  - Text tower: RoBERTa CLS token → Dense projection
  - Linguistic features tower: Dense layers with normalization
- FiLM-style gating mechanism for feature fusion
- Two-phase training:
  - Phase 1: Frozen transformer (2 epochs)
  - Phase 2: Full fine-tuning (5 epochs)
- Optimization: AdamW with warmup cosine schedule
- Label smoothing: 0.05
- Class weighting for imbalanced data

### Training Strategy

- **Random Seed**: 42 (for reproducibility)
- **Validation**: Holdout validation set
- **Early Stopping**: Patience of 2 epochs on validation AUC
- **Best Model Selection**: Based on validation AUC

### Evaluation Metrics

- **Accuracy**: Overall classification accuracy
- **Precision**: Positive predictive value
- **Recall**: True positive rate
- **F1 Score**: Harmonic mean of precision and recall
- **AUC-ROC**: Area under ROC curve
- **AUC-PR**: Area under precision-recall curve
- **Confusion Matrix**: Detailed classification breakdown

## Results

### Model Performance Comparison

All models were evaluated on both datasets with comprehensive metrics including accuracy, F1-score, AUC-ROC, and AUC-PR. Key findings:

1. **Best Traditional Model**: XGBoost with TF-IDF features
2. **Best Overall Model**: RoBERTa + Linguistic Features (GenAI Extension)
3. **Dataset Performance**: Models generally performed better on Kaggle dataset compared to McIntire
4. **Feature Importance**: Combination of text embeddings and linguistic features improved performance

### Out-of-Distribution (OOD) Testing

Cross-dataset evaluation was performed to assess model generalization:
- McIntire-trained models tested on Kaggle dataset
- RoBERTa model showed superior OOD performance
- Analysis provided insights into model robustness

## GenAI Extension

### Transformer-Based Architecture

The GenAI extension implements a state-of-the-art transformer-based architecture:

**Key Innovations**:
1. **Multi-Modal Fusion**: Late fusion strategy combining:
   - Contextual embeddings from RoBERTa transformer
   - Hand-crafted linguistic features
   - FiLM-style gating for adaptive feature selection

2. **Two-Phase Training**:
   - Prevents catastrophic forgetting
   - Stable convergence
   - Better utilization of pre-trained knowledge

3. **Advanced Optimization**:
   - Warmup cosine learning rate schedule
   - AdamW optimizer with weight decay
   - Label smoothing for better calibration
   - Class weighting for imbalanced datasets

## Visualization

The notebook generates comprehensive visualizations:
- Label distribution plots
- Top linguistic features analysis
- ROC curves for all models
- Model accuracy comparisons (ranked)
- Confusion matrices
- Training history plots

## Model Checkpointing

Trained models are saved in `all_models/` directory:
- **best_model.keras**: Best RoBERTa+LF model (by validation AUC)
- Models are automatically loaded if checkpoints exist
- Saves training time on subsequent runs

## Reproducibility

To ensure reproducibility:
1. Fixed random seed (42) across all libraries
2. Deterministic operations enabled
3. Saved model checkpoints provided
4. Complete preprocessing pipeline documented
5. Git LFS for large file version control


## License

See `LICENSE` file for details.

## Citation

If you use this code or find it helpful, please cite the original paper and this repository.