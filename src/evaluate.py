import torch
import torch.nn as nn
import pickle
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, f1_score
from pathlib import Path
try:
    from .model import TitanicMLP
except ImportError:
    from model import TitanicMLP


def load_model(model_path: str, n_features: int) -> nn.Module:
    model = TitanicMLP(n_features=n_features)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    print(f"[OK] Model loaded from {model_path}")
    return model


def load_data(data_path: str):
    with open(data_path, 'rb') as f:
        X, y = pickle.load(f)
    return X, y


def get_predictions(model: nn.Module, X: np.ndarray):
    X_tensor = torch.FloatTensor(X)
    
    with torch.no_grad():
        logits = model(X_tensor)
        predictions = logits.argmax(dim=1).numpy()
    
    return predictions


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'f1': f1_score(y_true, y_pred, average='weighted'),
        'confusion_matrix': confusion_matrix(y_true, y_pred)
    }
    return metrics


def compute_cross_entropy_loss(model: nn.Module, X: np.ndarray, y_true: np.ndarray) -> float:
    X_tensor = torch.FloatTensor(X)
    y_tensor = torch.LongTensor(y_true)

    with torch.no_grad():
        logits = model(X_tensor)
        loss = nn.CrossEntropyLoss()(logits, y_tensor).item()

    return float(loss)


def plot_confusion_matrix(cm: np.ndarray, output_path: str = "results/confusion_matrix.png") -> None:
    fig, ax = plt.subplots(figsize=(8, 6))
    
    im = ax.imshow(cm, cmap='Blues', aspect='auto')
    
    ax.set_xlabel('Predicted Label', fontsize=12)
    ax.set_ylabel('True Label', fontsize=12)
    ax.set_title('Confusion Matrix - Titanic Survival MLP', fontsize=14, fontweight='bold')
    
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            text = ax.text(j, i, cm[i, j], ha='center', va='center', color='black', fontsize=12)
    
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(['Did not survive', 'Survived'])
    ax.set_yticklabels(['Did not survive', 'Survived'])
    
    plt.colorbar(im, ax=ax)
    plt.tight_layout()
    
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150)
    print(f"[OK] Confusion matrix saved to {output_path}")
    plt.close()


def evaluate_pipeline(data_path: str = "data/processed/wisla_train.pkl",
                     model_path: str = "saved_models/titanic_model.pt") -> None:
    print("\n" + "="*50)
    print("TITANIC MLP EVALUATION")
    print("="*50 + "\n")
    
    X, y = load_data(data_path)
    model = load_model(model_path, n_features=X.shape[1])
    
    y_pred = get_predictions(model, X)
    
    metrics = compute_metrics(y, y_pred)
    cross_entropy_loss = compute_cross_entropy_loss(model, X, y)
    
    print("EVALUATION RESULTS:")
    print(f"  Accuracy:  {metrics['accuracy']*100:.2f}%")
    print(f"  F1 Score:  {metrics['f1']:.4f}")
    print(f"  Cross Entropy Loss: {cross_entropy_loss:.4f}")
    print(f"\nConfusion Matrix:")
    print(f"  {metrics['confusion_matrix']}\n")
    
    print("Classification Report:")
    print(classification_report(y, y_pred, target_names=['Did not survive', 'Survived']))
    
    plot_confusion_matrix(metrics['confusion_matrix'])
    
    eval_metrics = {
        'accuracy': metrics['accuracy'],
        'f1_score': metrics['f1'],
        'cross_entropy_loss': cross_entropy_loss,
        'confusion_matrix': metrics['confusion_matrix'].tolist(),
        'predictions': y_pred.tolist()
    }
    
    eval_path = "results/evaluation_metrics.pkl"
    Path(eval_path).parent.mkdir(parents=True, exist_ok=True)
    with open(eval_path, 'wb') as f:
        pickle.dump(eval_metrics, f)
    print(f"[OK] Evaluation metrics saved to {eval_path}\n")


if __name__ == "__main__":
    evaluate_pipeline()