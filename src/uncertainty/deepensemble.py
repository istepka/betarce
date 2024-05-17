import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import scipy.stats as stats

class DeepEnsemble(nn.Module):
    def __init__(self, 
        base_model: nn.Module, 
        num_models: int, 
        input_size: int, 
        class_threshold: float = 0.5, 
        beta_ci: float = 0.95
    ) -> None:
        super(DeepEnsemble, self).__init__()
        self.base_model = base_model
        self.models = nn.ModuleList([base_model(input_size) for _ in range(num_models)])
        
        # Freeze the weights of the base model
        # for model in self.models:
        #     model.eval()
                
        # Add thresholding layer to each model
        self.threshold = nn.Parameter(torch.tensor(class_threshold))
        self.beta_ci = beta_ci
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Forward pass through each model, thresholding the output
        outputs = []
        for model in self.models:
            output = model(x)
            outputs.append(output)
            
        # Now, create a and b parameters for beta distribution
        a = torch.stack(outputs).sum(dim=0) + 1
        b = len(outputs) - a + 2
        
        # # Get the lower bound of the beta distribution at self.beta_ci confidence
        # lower_bound = stats.beta.ppf((1 - self.beta_ci) / 2, a, b)
        
        # a = self.models[0](x) + 1
        # b = 1 - a + 1
        
        return a / (a + b)
    
    def predict_beta_parameters(self, x: torch.Tensor) -> tuple[int, int]:
        outputs = []
        for model in self.models:
            output = model(x)
            output = output > self.threshold
            outputs.append(output)
            
        a = torch.stack(outputs).sum(dim=0) + 1
        b = len(outputs) - a + 2
        
        return a.item(), b.item()
    
# Create own criterion
class OwnLoss(nn.Module):
    def __init__(self, lambda_proximity: float = 0.1):
        super(OwnLoss, self).__init__()
        
        self.lambda_proximity = lambda_proximity
        
    def forward(self, output: torch.Tensor, target: torch.Tensor, original_x: torch.Tensor) -> torch.Tensor:
        bce = -target * torch.log(output) - (1 - target) * torch.log(1 - output)
        
        proximity = torch.norm(original_x - output)
        
        return bce + self.lambda_proximity * proximity
        
        
        
class OptimizeUsingEnsemble:
    def __init__(self, ensemble: DeepEnsemble, optimizer: torch.optim.Optimizer):
        self.ensemble = ensemble
        self.optimizer = optimizer
        self.criterion = OwnLoss()
        
    def optimize_input(self, x: torch.Tensor, y: torch.Tensor, epochs: int, verbose: bool = False) -> torch.Tensor:
        # Freeze the ensemble weights
        for param in self.ensemble.parameters():
            param.requires_grad = False
            
        self.original_x = x.clone().detach()
            
        x.requires_grad = True
            
        # Optimize the input to maximize the ensemble output
        for epoch in range(epochs):
            self.optimizer.zero_grad()
            
            # Get the ensemble output
            output = self.ensemble(x)
            
            # Calculate the loss
            loss = self.criterion(output, y, self.original_x)
            
            # Backpropagate the loss
            loss.backward()
            
            # Update the input
            x = x - x.grad
            x.retain_grad()
            
            if verbose:
                print(f"Epoch {epoch}, input: {x}, gradient: {x.grad}, loss: {loss.item()}")
                
        return x
  
class BaseNet(nn.Module):
    def __init__(self, input_size: int):
        super(BaseNet, self).__init__()
        
        self.input_size = input_size
        
        self.fc1 = nn.Linear(input_size, 100)
        self.fc2 = nn.Linear(100, 100)
        self.fc3 = nn.Linear(100, 1)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.view(-1, self.input_size)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        x = torch.sigmoid(x).squeeze()
        return x
    
class BaseNetTrainer:
    def __init__(self, model, train_loader, test_loader, val_loader, optimizer, criterion):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.optimizer = optimizer
        self.criterion = criterion
        
    def train(self, epochs, verbose=False, early_stopping=False, patience=5):
        min_val_loss = float('inf')
        patience_counter = 0

        for epoch in range(epochs):
            self.model.train()
            for i, (data, target) in enumerate(self.train_loader):
                self.optimizer.zero_grad()
                output = self.model(data)
                loss = self.criterion(output, target)
                loss.backward()
                self.optimizer.step()
                if i % 100 == 0 and verbose:
                    print(f"Epoch {epoch}, iteration {i}, loss: {loss.item()}")

            val_loss = self.test(self.val_loader, type='val', verbose=verbose)

            if early_stopping:
                if val_loss < min_val_loss:
                    min_val_loss = val_loss
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= patience:
                        print(f"Early stopping at epoch {epoch}")
                        return

        self.test(self.test_loader, type='test', verbose=verbose)
            
    def test(self, loader: DataLoader, type: str = 'val', verbose: bool = False) -> float:
        self.model.eval()
        test_loss = 0
        correct = 0
        with torch.no_grad():
            for data, target in loader:
                output = self.model(data)
                test_loss += self.criterion(output, target).item()
                pred = output.round()
                correct += pred.eq(target.view_as(pred)).sum().item()
        test_loss /= len(loader.dataset)
        accuracy = correct / len(loader.dataset)
        
        if verbose:
            print(f"{type} set: Average loss: {test_loss}, Accuracy: {accuracy}")
            
        return test_loss
    
    def get_model(self):
        return self.model
    
    
if __name__ == '__main__':
    
    # Load some tabular data from scikit-learn
    from sklearn.datasets import load_breast_cancer
    from sklearn.model_selection import train_test_split
    import numpy as np
    import torch
    
    data = load_breast_cancer()
    X = data.data
    y = data.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_test, y_test, test_size=0.5, random_state=42)
    
    X_train = torch.tensor(X_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.float32)
    X_val = torch.tensor(X_val, dtype=torch.float32)
    y_val = torch.tensor(y_val, dtype=torch.float32)
    X_test = torch.tensor(X_test, dtype=torch.float32)
    y_test = torch.tensor(y_test, dtype=torch.float32)
    
    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=64, shuffle=True)
    val_loader = DataLoader(TensorDataset(X_val, y_val), batch_size=64, shuffle=False)
    test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=64, shuffle=False)
    
    # Create a deep ensemble
    deep_ensemble = DeepEnsemble(base_model=BaseNet, num_models=8, input_size=X_train.shape[1])
    
    # Train each model in the ensemble
    for model in deep_ensemble.models:
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
        criterion = torch.nn.BCELoss()
        trainer = BaseNetTrainer(model, train_loader, test_loader, val_loader, optimizer, criterion)
        trainer.train(epochs=25)
        
    # Evaluate each model in the ensemble
    losses = []
    for model in deep_ensemble.models:
        trainer = BaseNetTrainer(model, train_loader, test_loader, val_loader, optimizer, criterion)
        loss = trainer.test(test_loader, type='test', verbose=True)
        losses.append(loss)
    
    mean_loss = np.mean(losses)
    std_loss = np.std(losses)
    print(f"Mean loss: {mean_loss}, Std loss: {std_loss}")
     
    
    
    # Check the ensemble output on a single example
    example = X_test[0]
    label = y_test[0]
    output = deep_ensemble(example)
    print(f'Example input: {example}')
    print(f'Example label: {label}')
    print(f'Ensemble output: {output}')
    
    
    # Check the gradient of the ensemble output with respect to the input
    ensemble_optim = OptimizeUsingEnsemble(deep_ensemble, torch.optim.Adam(deep_ensemble.parameters(), lr=1e-4))
    ensemble_optim.optimize_input(example, label, epochs=100, verbose=True)
    
    
    a,b = deep_ensemble.predict_beta_parameters(example)
    print(f"Parameters of beta distribution: a={a}, b={b}")
    mean_beta = a / (a + b)
    print(f"Mean of beta distribution: {mean_beta}")
    lower_bound = stats.beta.ppf((1 - deep_ensemble.beta_ci) / 2, a, b)
    print(f"Lower bound of beta distribution: {lower_bound}")
    output = deep_ensemble(example)
    print(f'Ensemble output: {output}')