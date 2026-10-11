import torch
import torch.nn as nn
import torch.optim as optim
import pickle
import matplotlib.pyplot as plt
from pathlib import Path
from typing import List, Tuple
try:
    from .model import TitanicMLP
except ImportError:
    from model import TitanicMLP


def load_training_data(data_path: str) -> Tuple[torch.Tensor, torch.Tensor]:
    with open(data_path, 'rb') as f:
        X_train, y_train = pickle.load(f)
    
    X_tensor = torch.FloatTensor(X_train)
    y_tensor = torch.LongTensor(y_train)
    
    print(f"Training data loaded: {X_tensor.shape}, {y_tensor.shape}")
    return X_tensor, y_tensor


def train_model(model: nn.Module, 
                X_train: torch.Tensor, 
                y_train: torch.Tensor,
                n_epochs: int = 300,
                learning_rate: float = 0.001,
                verbose: bool = True) -> List[float]:
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
    
    loss_history = []
    
    print(f"\nTraining started:")
    print(f"  Epochs: {n_epochs}")
    print(f"  Learning rate: {learning_rate}")
    print(f"  Optimizer: AdamW")
    print(f"  Loss function: CrossEntropyLoss\n")
    
    for epoch in range(n_epochs):
        y_pred = model(X_train)
        loss = criterion(y_pred, y_train)
        
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        
        loss_history.append(loss.item())
        
        if verbose and (epoch + 1) % 50 == 0:
            print(f"Epoch [{epoch+1:3d}/{n_epochs}] Loss: {loss.item():.4f}")
    
    print("\n[OK] Training completed\n")
    return loss_history


def evaluate_model(model: nn.Module, 
                   X_train: torch.Tensor, 
                   y_train: torch.Tensor) -> Tuple[float, float]:
    model.eval()
    
    with torch.no_grad():
        logits = model(X_train)
        predictions = logits.argmax(dim=1)
        accuracy = (predictions == y_train).float().mean().item()
        
        criterion = nn.CrossEntropyLoss()
        loss = criterion(logits, y_train).item()
    
    return accuracy, loss


def plot_loss_curve(loss_history: List[float], 
                   output_path: str = "results/loss_curve.png") -> None:
    plt.figure(figsize=(10, 6))
    plt.plot(loss_history, linewidth=2, color='#2E86AB')
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Cross Entropy Loss', fontsize=12)
    plt.title('Training Loss Curve - Titanic Survival MLP', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150)
    print(f"[OK] Loss curve saved to {output_path}")
    plt.close()


def save_model(model: nn.Module, output_path: str = "saved_models/titanic_model.pt") -> None:
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), output_path)
    print(f"[OK] Model weights saved to {output_path}")


def train_pipeline(data_path: str = "data/processed/wisla_train.pkl",
                  model_output: str = "saved_models/titanic_model.pt",
                  plot_output: str = "results/loss_curve.png",
                  n_epochs: int = 300,
                  learning_rate: float = 0.001) -> None:
    print("\n" + "="*50)
    print("TITANIC MLP TRAINING PIPELINE")
    print("="*50)
    
    X_train, y_train = load_training_data(data_path)
    
    n_features = X_train.shape[1]
    model = TitanicMLP(n_features=n_features)
    
    loss_history = train_model(model, X_train, y_train, n_epochs, learning_rate)
    
    accuracy, final_loss = evaluate_model(model, X_train, y_train)
    print("RESULTS:")
    print(f"  Training Accuracy: {accuracy*100:.2f}%")
    print(f"  Cross Entropy Loss: {final_loss:.4f}\n")
    
    plot_loss_curve(loss_history, plot_output)
    save_model(model, model_output)
    
    metrics = {
        'accuracy': accuracy,
        'cross_entropy_loss': final_loss,
        'final_loss': final_loss,
        'n_epochs': n_epochs,
        'learning_rate': learning_rate,
        'loss_history': loss_history
    }
    
    metrics_path = "results/training_metrics.pkl"
    Path(metrics_path).parent.mkdir(parents=True, exist_ok=True)
    with open(metrics_path, 'wb') as f:
        pickle.dump(metrics, f)
    print(f"[OK] Metrics saved to {metrics_path}\n")


if __name__ == "__main__":
    train_pipeline()