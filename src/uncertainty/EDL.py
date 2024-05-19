import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
import scipy.stats as stats
import numpy as np

from .edl_repo.losses import edl_digamma_loss, edl_log_loss, edl_mse_loss, relu_evidence


class EDLModel(nn.Module):
    def __init__(self, input_size: int, num_of_classes: int = 2) -> None:
        super(EDLModel, self).__init__()
        self.num_of_classes = num_of_classes
        self.input_size = input_size
        
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
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, num_of_classes)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)  
    
    def full_predict(self, x):
        
        if isinstance(x, np.ndarray):
            x = torch.tensor(x, dtype=torch.float32)
            
            # reshape if necessary
            if len(x.shape) == 1:
                x = x.unsqueeze(0)
        
        output =  self.forward(x)
        evidence = relu_evidence(output)
        alpha = evidence + 1
        _, preds = torch.max(output, 1)
        prob = alpha / torch.sum(alpha, dim=1, keepdim=True)
        prob, _ = torch.max(prob, dim=1)
        uncertainty = (self.num_of_classes / torch.sum(alpha, dim=1, keepdim=True))
            
        return preds, prob, uncertainty
    
    def crisp_predict(self, x):
        output = self.forward(x)
        _, preds = torch.max(output, 1)
        return preds
    
    def proba_predict(self, x):
        output = self.forward(x)
        evidence = relu_evidence(output)
        alpha = evidence + 1
        prob = alpha / torch.sum(alpha, dim=1, keepdim=True)
        return prob     
        
     
class EDL:
    def __init__(self, 
            model: nn.Module,
            X_train: torch.Tensor,
            y_train: torch.Tensor,
            X_val: torch.Tensor,
            y_val: torch.Tensor, 
            criterion = 'log', 
            num_of_classes=2, 
            device='cpu',
            annealing=10,
            lr=1e-3,
            batch_size=64,
            seed=None,
        ) -> None:
        
        if seed is not None:
            torch.manual_seed(seed)
        
        self.model = model
        
        self.train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=batch_size, shuffle=True)
        self.val_loader = DataLoader(TensorDataset(X_val, y_val), batch_size=batch_size, shuffle=False)
        
        self.optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=lr)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=3, gamma=0.1)
        
        match criterion:
            case 'log':
                self.criterion = edl_log_loss
            case 'mse':
                self.criterion = edl_mse_loss
            case 'digamma':
                self.criterion = edl_digamma_loss
            case _:
                raise ValueError("Invalid criterion")
        
        self.num_of_classes = num_of_classes
        self.device = device
        self.loss_annealing = annealing

        self.loss_history = []
        self.gradient_history = []
        
    def train(self, epochs, verbose=False, early_stopping=False, patience=5, grad_clip=100):
        self.min_val_loss = float('inf')
        patience_counter = 0
        
        self.best_model = None
        

        for epoch in range(epochs):
            self.model.train()
            for i, (data, target) in enumerate(self.train_loader):
                self.optimizer.zero_grad()
                outputs = self.model(data)
                loss = self.criterion(
                    outputs, target, epoch, self.num_of_classes, self.loss_annealing, self.device
                )
                
                self.loss_history.append(loss.item())
                
                loss.backward()
                
                # Calculate gradient magnitude
                total_norm = self.calc_grad_norm()
                self.gradient_history.append(total_norm)
                
                
                self.optimizer.step()
                
                
                if i % 100 == 0 and verbose:
                    print(f"Epoch {epoch}, iteration {i}, loss: {loss.item()}")
                    
            val_loss, val_acc = self.evaluate(self.val_loader)
            
            if val_loss < self.min_val_loss:
                self.best_model = self.model.state_dict()
            
            if early_stopping:
                if val_loss < self.min_val_loss:
                    self.min_val_loss = val_loss
                    patience_counter = 0
                else:
                    patience_counter += 1
                    
                if patience_counter == patience:
                    print("Early stopping")
                    self.model.load_state_dict(self.best_model)
                    break
                
            if self.scheduler is not None:      
                self.scheduler.step()
                
    def evaluate(self, data_loader):
        # Calculate validation loss
        self.model.eval()
        val_loss = 0
        correct = 0
        with torch.no_grad():
            for data, target in data_loader:
                output = self.model(data)
                val_loss += self.criterion(
                    output, target, -1, self.num_of_classes, self.loss_annealing, self.device
                ).item()
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.argmax(dim=1, keepdim=True)).sum().item()
                
        val_loss /= len(data_loader.dataset)
        return val_loss, correct / len(data_loader.dataset)

    def calc_grad_norm(self):
        total_norm = 0
        for p in self.model.parameters():
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
        total_norm = total_norm ** (1. / 2)
        return total_norm
    
    def get_model(self):
        return self.model
    
def interpolate_samples(model: EDL, X_test, y_test, examples: int = 5, num_classes: int = 2, n_interp: int = 100):
    model.eval()
    
    # Get one sample from each class
    class_0 = X_test[torch.argmax(y_test, dim=1) == 0][:examples]
    class_1 = X_test[torch.argmax(y_test, dim=1) == 1][:examples]
    
    
    interps = []
    interp_agg_uncertainty = []
    interp_agg_prob = []
    interp_agg_pred = []
    
    
    for iex in range(examples):
        # Interpolate between the two samples
        interp = torch.zeros(n_interp, X_test.shape[1])
        
        for i in range(n_interp):
            interp[i] = class_0[iex] + (class_1[iex] - class_0[iex]) * i / n_interp
            
        interps.append(interp)
        

        agg_pred = []
        agg_uncertainty = []
        agg_prob = []
        
        for i, data in enumerate(interp):
            data = data.unsqueeze(0)
            preds, prob, uncertainty = model.full_predict(data)
            
            agg_pred.append(preds)
            agg_uncertainty.append(uncertainty)
            agg_prob.append(prob[0].unsqueeze(0))
            
        interp_agg_uncertainty.append(torch.cat(agg_uncertainty))
        interp_agg_prob.append(torch.cat(agg_prob))
        interp_agg_pred.append(torch.cat(agg_pred))
        
    # Plot the uncertainty as a function of the interpolation
    agg_uncertainty = torch.cat(interp_agg_uncertainty).detach().numpy()
    interp = torch.cat(interps).detach().numpy()
    agg_prob = torch.cat(interp_agg_prob).detach().numpy()
    
    fig, ax = plt.subplots(ncols=2, figsize=(12, 5))
    
    for i in range(examples):
        ax[0].plot(np.linspace(0, 1, n_interp), agg_uncertainty[i*n_interp:(i+1)*n_interp])
        ax[1].plot(np.linspace(0, 1, n_interp), agg_prob[i*n_interp:(i+1)*n_interp])
    
    # Unceertainty plot
    ax[0].set_title("Uncertainty")
    ax[0].set_xlabel("Interpolation")
    ax[0].set_ylabel("Uncertainty")
    
    # Probability plot
    ax[1].set_title("Probability")
    ax[1].set_xlabel("Interpolation")
    ax[1].set_ylabel("Probability")
    
    # Add legend
    ax[1].legend([f"Example {i+1}" for i in range(examples)])
    
    plt.suptitle("Interpolation between two samples from different classes")
    
    plt.tight_layout()
    plt.show()
    
if __name__ == '__main__':
    
    # Load some tabular data from scikit-learn
    from sklearn.datasets import load_breast_cancer, load_diabetes, load_wine
    from sklearn.model_selection import train_test_split
    import numpy as np
    import torch
    
    torch.random.manual_seed(32)
    
    data = load_breast_cancer()
    X = data.data
    y = data.target
    # y = (y > 0).astype(int)
    
    # Do one hot encoding of the target
    y = np.eye(len(np.unique(y)))[y]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)
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
    
    test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=64, shuffle=False)

    model = EDLModel(input_size=X_train.shape[1])
    EDLTrainer = EDL(model, X_train, y_train, X_val, y_val, criterion='log')
    EDLTrainer.train(epochs=15, verbose=True, early_stopping=True)
     
    # Plot the loss
    plt.plot(EDLTrainer.loss_history)
    plt.title("Loss history")
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.show()
    
    
    # Evaluate the model
    model.eval()
    
    num_classes = 2
    device = 'cpu'
    epoch = -1
    
    agg_pred = []
    agg_uncertainty = []
    agg_prob = []
    
    for i, (data, target) in enumerate(test_loader):
        data = data.to(device)
        target = target.to(device)
        
        preds, prob, uncertainty = model.full_predict(data)
        
        agg_pred.append(preds)
        agg_uncertainty.append(uncertainty)
        agg_prob.append(prob)
        
    
    # Evaulate using class method
    test_loss, test_acc = EDLTrainer.evaluate(test_loader)
    print(f"Test Accuracy: {test_acc}")
    print(f"Test Loss: {test_loss}")
    
    # Plot the uncertainty
    agg_uncertainty = torch.cat(agg_uncertainty).detach().numpy()
    agg_pred = torch.cat(agg_pred).detach().numpy()

    plt.hist(agg_uncertainty, bins=50)
    plt.title("Uncertainty histogram")
    plt.xlabel("Uncertainty")
    plt.ylabel("Frequency")
    plt.show()
    
    # Plot the probability
    agg_prob = torch.cat(agg_prob).detach().numpy()
    plt.hist(agg_prob, bins=50)
    plt.title("Probability histogram")
    plt.xlabel("Probability")
    plt.ylabel("Frequency")
    
    plt.show()
    
    
    
        

    interpolate_samples(model, X_test, y_test, examples=5, num_classes=2, n_interp=100)
    
    
    
    
    
    