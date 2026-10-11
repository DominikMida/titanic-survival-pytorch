# titanic-survival-pytorch

A PyTorch-based project for binary survival prediction using a structured passenger dataset. The repository contains the complete workflow for preprocessing, model training, evaluation, and result generation in a clean, reproducible format.

## Overview

This project focuses on predicting whether a passenger survived based on available attributes such as age, family size, fare, embarkation port, and other available metadata. The solution is implemented with a modular Python pipeline and a feed-forward neural network trained with PyTorch.

The main goals are to:

- clean and prepare tabular data for machine learning
- engineer useful features from raw inputs
- train a binary classification model
- evaluate performance with standard metrics
- provide a project structure suitable for further experimentation and iteration

## Project Structure

```text
.
├── data/
│   ├── raw/
│   │   ├── wisla_train.csv
│   │   ├── wisla_test.csv
│   │   └── wisla_sample_submission.csv
│   └── processed/
├── notebooks/
├── results/
├── saved_models/
├── src/
│   ├── __init__.py
│   ├── data_preprocessing.py
│   ├── model.py
│   ├── train.py
│   └── evaluate.py
├── README.md
├── requirements.txt
├── .gitignore
└── .venv/
```

## Technical Stack

- Python
- PyTorch
- Pandas
- NumPy
- scikit-learn
- Matplotlib

## Model Architecture

The project uses a Multi-Layer Perceptron (MLP) for binary classification:

- Input layer: engineered feature vector
- Hidden layers: 16 and 8 neurons
- Activation: ReLU
- Output layer: 2 classes
- Loss: Cross Entropy Loss
- Optimizer: AdamW

## Data Processing

The preprocessing pipeline includes:

- loading raw train and test datasets
- imputing missing values using dataset statistics
- handling outliers in fare values
- removing irrelevant features
- extracting and mapping title information from passenger names
- normalizing numerical variables
- one-hot encoding categorical features
- preparing final arrays for training

## Training and Evaluation

The model is trained on the prepared training dataset. The metrics below are measured on that same dataset, so they should not be interpreted as validation or test performance.

### Current results

- Accuracy: 84.85%
- F1 Score: 0.8486
- Cross Entropy Loss: 0.3679

## Requirements

```bash
python 3.10+
pip install -r requirements.txt
```

## How to Run

### 1. Create a virtual environment

```bash
python -m venv .venv
```

Windows:

```bash
.venv\Scripts\activate
```

Linux/macOS:

```bash
source .venv/bin/activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Run preprocessing

```bash
python src/data_preprocessing.py
```

### 4. Train the model

```bash
python src/train.py
```

### 5. Evaluate the model

```bash
python src/evaluate.py
```

## Outputs

The project saves:

- processed datasets
- trained model weights
- training loss graph
- evaluation metrics
- confusion matrix

## Potential Extensions

This repository provides a solid baseline and can be extended with:

- hyperparameter tuning
- cross-validation
- alternative model comparisons
- feature importance analysis
- improved feature engineering
- submission generation for Kaggle-style prediction tasks
