import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset, random_split
import os
import torch.nn.functional as F
from torch.nn import functional as nn_F
import torch.nn as nn
from torchvision import datasets, transforms
import random
import itertools
from collections import defaultdict
import copy
import pickle
from torch.optim.lr_scheduler import StepLR


seed = 131
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
os.environ['PYTHONHASHSEED'] = str(seed)

# Move to GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class SimpleNN(nn.Module):
   def __init__(self, input_dim, hidden_dim):
      super(SimpleNN, self).__init__()
      self.output_layer = nn.Linear(hidden_dim, 10)
      # Initialization
      self.w_1 = torch.nn.parameter.Parameter(nn.init.normal_(torch.zeros(input_dim,hidden_dim)))
      self.w_2 = torch.nn.parameter.Parameter(nn.init.normal_(torch.zeros(hidden_dim,10)))

   def forward(self, x):
      x = x.view(x.size(0), -1) # Flatten the input
      hidden_output = torch.relu(torch.mm(x, self.w_1))
      final_output = torch.mm(hidden_output, self.w_2) # No transposition of w_2 
      return final_output
  
# # Hyperparameters
N = 15  # Number of source domains


# Learning rate schedule
def learning_rate_schedule(t, initial_lr, T):
    return initial_lr / (1 + t / (0.1 * T))

output_directory = './Neurips_mnist_non_iid/g2/results_g2'
os.makedirs(output_directory, exist_ok=True)
output_directory_data = './Neurips_mnist_non_iid/g2/datasets'
os.makedirs(output_directory_data, exist_ok=True)
with open(os.path.join(output_directory_data, 'data_loaders_mnist_non_iid.pkl'), 'rb') as f:
    data_loaders = pickle.load(f)
# Load the array from the file
alpha_values = np.loadtxt(os.path.join(output_directory, 'Alpha_values.txt'))

# Access the last row
alpha = torch.tensor(alpha_values[-1])
model_alpha = SimpleNN(input_dim=784, hidden_dim=28).to(device)
criterion = nn.CrossEntropyLoss()

num_iterations = 50
accuracies = []
lr_init = 0.01
# optimizer = torch.optim.Adam(model_alpha.parameters(), lr=lr_init, weight_decay=0.01)
optimizer = torch.optim.SGD(model_alpha.parameters(), lr=lr_init, weight_decay=0.01) 

# Define StepLR scheduler for learning rate decay
scheduler = StepLR(optimizer, step_size=1, gamma=0.95)

losses_alpha = []
validation_losses_alpha = []
model_alpha = model_alpha.to(device)

num_iterations = 2000
accuracies = []
lr_init = 0.001
optimizer = torch.optim.SGD(model_alpha.parameters(), lr=lr_init) 
losses_alpha = []
validation_losses_alpha = []

# Set a patience value
patience = 10
patience_counter = 0 

best_val_loss = float('inf')  # Initialize the best validation loss to infinity


for iteration in range(num_iterations):
    total_loss = torch.tensor(0.0, device=device)
    model_alpha.train()
    for j in range(N):
        for X_j, y_j in data_loaders[j+1]:
            model_alpha.zero_grad()
            X_j, y_j = X_j.to(device), y_j.to(device)
            y_pred_j = model_alpha(X_j.view(-1, 784))

            # Compute loss
            loss = criterion(y_pred_j, y_j)
            # Weighted loss using learned alpha
            weighted_loss = alpha[j] * loss
            total_loss += weighted_loss.item()
            
            # Zero out gradients, backward pass, optimizer step
            optimizer.zero_grad()
            weighted_loss.backward()
            optimizer.step()
    losses_alpha.append(total_loss)
                
   
    model_alpha.eval()  
    correct_predictions = 0
    total_samples_test = 0
    val_loss = 0.0
    with torch.no_grad():  
        for X_test, y_test in data_loaders['target_test']:
            X_test, y_test = X_test.to(device), y_test.to(device)
            y_pred_test = model_alpha(X_test.view(-1, 784))
            predicted_labels = torch.argmax(y_pred_test, dim=1) 
            val_loss += criterion(y_pred_test.float(), y_test.long()).item()
            correct_predictions += (predicted_labels == y_test).sum().item()
            total_samples_test += y_test.size(0)

    val_loss /= len(data_loaders['target_test'])

    accuracy_test = correct_predictions / total_samples_test
    validation_losses_alpha.append(val_loss)
    accuracies.append(accuracy_test)

    # Check for early stopping
    if validation_losses_alpha[-1] < best_val_loss:
        best_val_loss = validation_losses_alpha[-1]
        patience_counter = 0  # Reset the counter if validation loss improves
    else:
        patience_counter += 1

    if patience_counter >= patience:
        print("Early stopping triggered at iteration", iteration)
        break

    # Update learning rate
    scheduler.step()
    print(f"Iteration: {iteration}, LR: {scheduler.get_last_lr()[0]}, Test Accuracy: {accuracy_test:.4f}, Validation Loss: {val_loss:.4f}")

    model_alpha.train()

np.savetxt(os.path.join(output_directory, 'alpha_acc.txt'), np.array(accuracies))
np.savetxt(os.path.join(output_directory, 'alpha_val_loss.txt'), np.array(validation_losses_alpha))

