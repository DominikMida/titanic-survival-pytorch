import torch
import torch.nn as nn


class TitanicMLP(nn.Module):
    def __init__(self, n_features: int, hidden_sizes: list = None):
        super(TitanicMLP, self).__init__()
        
        if hidden_sizes is None:
            hidden_sizes = [16, 8]
        
        layers = []
        prev_size = n_features
        
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.ReLU())
            prev_size = hidden_size
        
        layers.append(nn.Linear(prev_size, 2))
        
        self.model = nn.Sequential(*layers)
        
        print(f"Model architecture:")
        print(f"  Input features: {n_features}")
        print(f"  Hidden layers: {hidden_sizes}")
        print(f"  Output classes: 2 (binary classification)")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)
    
    def get_summary(self) -> str:
        return str(self.model)


if __name__ == "__main__":
    model = TitanicMLP(n_features=30)
    print("\n" + model.get_summary())
    
    x = torch.randn(4, 30)
    y = model(x)
    print(f"\nInput shape: {x.shape}")
    print(f"Output shape: {y.shape}")