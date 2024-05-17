import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
import scipy.stats as stats


# Create own criterion
class BayesRiskLoss(nn.Module):
    def __init__(self):
        super(BayesRiskLoss, self).__init__()
        self.epsilon = 1e-6
        
    def forward(self, output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        a0 = (torch.sum(output, dim=1) + 1).unsqueeze(1) # Sum of the outputs of the models
        Si = (output * (a0 - output) / (a0.pow(2) * (a0 + 1))) # Variance of the beta distribution
    
        Si = Si + self.epsilon # Add epsilon to avoid division by zero
        output = output + self.epsilon # Add epsilon to avoid division by zero
    
        loss = target * (torch.log(Si) - torch.log(output)) 
        return loss.sum() 
    
class GibbsRiskLoss(nn.Module):
    def __init__(self):
        super(GibbsRiskLoss, self).__init__()
        self.epsilon = 1e-6
        
    def forward(self, output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        #\mathcal{L}_{Gibbs} = \sum^K_{i=0} (y_i - \hat y_i)^2 + \hat y_i \frac{(1 - \hat y_i)}{S_i + 1}
        a0 = (torch.sum(output, dim=1) + 1).unsqueeze(1) # Sum of the outputs of the models
        Si = (output * (a0 - output) / (a0.pow(2) * (a0 + 1)))
        
        # Change target and output to probabilities
        output_p = output / a0
        target_p = target / torch.sum(target, dim=1).unsqueeze(1)
        
        loss = (target_p - output_p).pow(2) + output_p * (1 - output_p) / (Si + 1)
        
        return loss.sum()
        

class EdlNet(nn.Module):
    def __init__(self, input_size: int) -> None:
        super(EdlNet, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_size, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 2),
            nn.ReLU()
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)       
        
     
  
class BaseNet(nn.Module):
    def __init__(self, input_size: int, arch_random_seed: int = 1) -> None:
        super(BaseNet, self).__init__()
        
        self.input_size = input_size
        
        # Get random nmumber generator
        generator = torch.Generator().manual_seed(arch_random_seed)
        layers_sizes = torch.randint(16, 128, (3,), generator=generator)
        
        self.fc1 = nn.Linear(input_size, layers_sizes[0])
        self.fc2 = nn.Linear(layers_sizes[0], layers_sizes[1])
        self.fc3 = nn.Linear(layers_sizes[1], layers_sizes[2])
        self.fc4 = nn.Linear(layers_sizes[2], 1)
        
        
        activations = [nn.ReLU, nn.LeakyReLU, nn.GELU, nn.SELU]
        chosen_activation = torch.randint(0, 4, (1,), generator=generator).item()
        self.activation = activations[chosen_activation]()
        
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.view(-1, self.input_size)
        x = self.fc1(x)
        x = self.activation(x)
        x = self.fc2(x)
        x = self.activation(x)
        x = self.fc3(x)
        x = self.activation(x)
        x = self.fc4(x)
        x = torch.sigmoid(x).squeeze()
        return x
    
class Trainer:
    def __init__(self, model, train_loader, test_loader, val_loader, optimizer, criterion):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.optimizer = optimizer
        self.criterion = criterion
        
        self.loss_history = []
        self.gradient_history = []
        
    def train(self, epochs, verbose=False, early_stopping=False, patience=5, grad_clip=100):
        min_val_loss = float('inf')
        patience_counter = 0

        for epoch in range(epochs):
            self.model.train()
            for i, (data, target) in enumerate(self.train_loader):
                self.optimizer.zero_grad()
                output = self.model(data)
                loss = self.criterion(output, target)
                
                self.loss_history.append(loss.item())
                
                loss.backward()
                
                # Calculate gradient magnitude
                total_norm = 0
                for p in self.model.parameters():
                    param_norm = p.grad.data.norm(2)
                    total_norm += param_norm.item() ** 2
                total_norm = total_norm ** (1. / 2)
                self.gradient_history.append(total_norm)
                
                # # Add gradient clipping
                # if grad_clip is not None:
                #     torch.nn.utils.clip_grad_norm_(self.model.parameters(), grad_clip)
                
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
        with torch.no_grad():
            for data, target in loader:
                output = self.model(data)
                test_loss += nn.functional.mse_loss(output, target).item()
        test_loss /= len(loader.dataset)
        
        if verbose:
            print(f"{type} set: Average loss: {test_loss}")
            
        return test_loss
    
    def get_model(self):
        return self.model
    
    
if __name__ == '__main__':
    
    # Load some tabular data from scikit-learn
    from sklearn.datasets import load_breast_cancer, load_diabetes
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
    
    # Standardize the data
    mean = X_train.mean(dim=0)
    std = X_train.std(dim=0)
    
    X_train = (X_train - mean) / std
    X_val = (X_val - mean) / std
    X_test = (X_test - mean) / std
    
    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=64, shuffle=True)
    val_loader = DataLoader(TensorDataset(X_val, y_val), batch_size=64, shuffle=False)
    test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=64, shuffle=False)
    
    # Create a deep edlnn
    models = [BaseNet(X_train.shape[1], arch_random_seed=s) for s in range(15)]
    
    # Train each model in the edlnn
    for model in models:
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
        criterion = torch.nn.BCELoss()
        trainer = Trainer(model, train_loader, test_loader, val_loader, optimizer, criterion)
        trainer.train(epochs=50)
        
    # Evaluate each model in the edlnn
    losses = []
    for model in models:
        trainer = Trainer(model, train_loader, test_loader, val_loader, optimizer, criterion)
        loss = trainer.test(test_loader, type='test', verbose=True)
        losses.append(loss)
    
    mean_loss = np.mean(losses)
    std_loss = np.std(losses)
    print(f"Mean loss: {mean_loss}, Std loss: {std_loss}")
    
    # Create new train, val, test data by querying the models about their predictions
    y_train_new = torch.ones((y_train.shape[0], 2))
    y_val_new = torch.ones((y_val.shape[0], 2))
    y_test_new = torch.ones((y_test.shape[0], 2))
    
    for i, model in enumerate(models):
        preds = model(X_train).detach().numpy()
        preds = preds > 0.5
        y_train_new[:, 0] += preds
        y_train_new[:, 1] += 1 - preds
        
        preds = model(X_val).detach().numpy()
        preds = preds > 0.5
        y_val_new[:, 0] += preds
        y_val_new[:, 1] += 1 - preds
        
        preds = model(X_test).detach().numpy()
        preds = preds > 0.5
        y_test_new[:, 0] += preds
        y_test_new[:, 1] += 1 - preds
    
    
    # Create a new model to predict the output of the ensemble
    edl_model = EdlNet(X_train.shape[1])
    
    # optimizer = torch.optim.SGD(edl_model.parameters(), lr=1e-4)
    optimizer = torch.optim.Adam(edl_model.parameters(), lr=1e-4, weight_decay=1e-5)
    criterion = torch.nn.MSELoss()
    # criterion = GibbsRiskLoss()
    # optimizer = torch.optim.Adam(edl_model.parameters(), lr=1e-5)
    # criterion = BayesRiskLoss()
    
    batch_size = 64
    tran_new_loader = DataLoader(TensorDataset(X_train, y_train_new), batch_size=batch_size, shuffle=True)
    val_new_loader = DataLoader(TensorDataset(X_val, y_val_new), batch_size=batch_size, shuffle=False)
    test_new_loader = DataLoader(TensorDataset(X_test, y_test_new), batch_size=batch_size, shuffle=False)
    
    trainer = Trainer(edl_model, tran_new_loader, test_new_loader, val_new_loader, optimizer, criterion)
    trainer.train(epochs=100, early_stopping=True, patience=20)
    
    # Plot the loss and gradient history
    fig, ax = plt.subplots(2, 1, figsize=(10, 10))
    ax[0].plot(trainer.loss_history)
    ax[0].set_title('Loss history')
    ax[1].plot(trainer.gradient_history)
    ax[1].set_title('Gradient history')
    plt.show()
    
    # Evaluate the new model
    trainer.test(test_new_loader, type='test', verbose=True)
    
    # Check the EDLNN output on a single example
    for i in np.random.randint(0, X_test.shape[0], 10):
        example = X_test[i]
        label = y_test_new[i]
        output = edl_model(example).detach()
        models_output = [model(example).detach().float() for model in models]
        
        # print(f'Example input: {example}')
        print(f'Example label: {label}')
        print(f'EDLNN output: {output}')
        # print(f'Models output: {models_output}')
    
    