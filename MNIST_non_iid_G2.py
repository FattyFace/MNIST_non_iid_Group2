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

seed = 54
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
os.environ['PYTHONHASHSEED'] = str(seed)
# Move to GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# Hyperparameters
N = 15  # Number of source domains
B = 32 # Batch size
T = 300 # Number of iterations
C = 20
gamma_init = 0.001
eta_init = 0.01
c = 1e-06 # Parameter for g function
beta_t = 0.9 # Beta


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
    
    
# Learning rate schedule
def learning_rate_schedule(t, initial_lr, T):
    return initial_lr / (1 + t / (0.1 * T))



# Simplex projection for alpha
def project_simplex(alpha):
    # Sort alpha in descending order
    sorted_alpha, _ = torch.sort(alpha, descending=True)
    
    # Calculate cumulative sum
    cumulative_sum = torch.cumsum(sorted_alpha, dim=0)
    
    # Determine rho
    arange_vals = torch.arange(1, alpha.size(0) + 1, dtype=torch.float32, device=device)
    rho_values = torch.where(sorted_alpha + (1 - cumulative_sum) / arange_vals > 0)[0]
    
    if rho_values.numel() == 0:
        return torch.full_like(alpha, 1.0 / alpha.size(0))
    
    rho = rho_values[-1].item()
    
    # Calculate theta
    theta = (cumulative_sum[rho] - 1) / (rho + 1)
    
    # Project onto the simplex
    return torch.clamp(alpha - theta, min=0)




# Projection for w to have norm 1
def project_w(w):
    return w / torch.norm(w)

# g function
def g(x, c):
    return torch.sqrt(x**2 + c)



output_directory = './Neurips_mnist_non_iid/g2/results_g2'

os.makedirs(output_directory, exist_ok=True)

# data_loaders = generate_random_non_iid_data(B, N)
output_directory_data = './Neurips_mnist_non_iid/g2/datasets'

with open(os.path.join(output_directory_data, 'data_loaders_mnist_non_iid.pkl'), 'rb') as f:
    data_loaders = pickle.load(f)




# Extract the number of samples for each domain
domain_counts = []
for i in range(1, N + 1):  # We start from 1 since 0 is reserved for the target domain
    domain_counts.append(len(data_loaders[i].dataset))

# Convert to a PyTorch tensor
domain_counts = torch.tensor(domain_counts)



# Assuming domain_counts is a tensor of size N with each m_i value
inv_domain_counts = 1.0 / domain_counts
M = torch.diag(inv_domain_counts).to(device) # Move to the same device



# Alpha initialization
alpha = torch.rand(N, device=device, requires_grad=True)
alpha = project_simplex(alpha)


model = SimpleNN(input_dim=784, hidden_dim=28).to(device)
previous_model_state_dict = model.state_dict()


model_prev = SimpleNN(input_dim=784, hidden_dim=28).to(device)
model_prev.load_state_dict(previous_model_state_dict)

criterion = nn.CrossEntropyLoss()
# Keep track of w^{t-1}
w_1_t_minus_1 = model.w_1.clone().detach()
w_2_t_minus_1 = model.w_2.clone().detach()

# Alpha values history for plotting
alpha_history = []

weight_decay = 0.001
# Training loop
F_values = []
w_1_gradient_norms = []
w_2_gradient_norms = []
alpha_gradient_norms = []
gradients_norms = []
z_t = torch.zeros(N, device=device) # Initialize z_t for all source domains


f_T_values = []
f_j_values = {j: [] for j in range(N)}  # Initialize a list for each source domain




for t in range(T):
    print("=============>>>>>>>>>>>>>  t: ", t, "<<<<<<<<<<<<<<=============")
    # Update learning rate based on schedule
    gamma = learning_rate_schedule(t, gamma_init, T)
    eta = learning_rate_schedule(t, eta_init, T)

    for X_T, y_T in data_loaders['target_train']:
        # Sample minibatch from target domain
        model.zero_grad() # Clear gradients
        X_T, y_T = X_T.to(device), y_T.to(device)

        # Compute target domain loss and gradients using the model
        y_pred_T = model(X_T.view(-1, 784))
        f_T_w_t = criterion(y_pred_T, y_T.squeeze().long())
        f_T_w_t.backward()  # backward pass
        grad_w_1_T_t = model.w_1.grad.clone()
        grad_w_2_T_t = model.w_2.grad.clone()


        # Compute gradients for previous time step using the model
        model_prev.load_state_dict(previous_model_state_dict)
        model_prev.zero_grad()
        y_pred_T_minus_1 = model_prev(X_T.view(-1, 784))
        f_T_w_t_minus_1 = criterion(y_pred_T_minus_1, y_T.squeeze().long())
        f_T_w_t_minus_1.backward()

    
        # Append target loss
        f_T_values.append(f_T_w_t.item())


        # Initialize a tensor to store the gradient values for each source domain
        v = torch.zeros(N, device=device)

        # Initialize z_t_plus_1 and g_w_t
        z_t_plus_1 = torch.zeros_like(z_t)
        g_w_1_t = torch.zeros_like(model.w_1, device=device)
        g_w_2_t = torch.zeros_like(model.w_2, device=device)
        # Iterate through source domains
        for j in range(N):
            # print(">>>>>>>>>>>>>  j: ", j, "<<<<<<<<<<<<<<")
            model.zero_grad()
            X_j, y_j = next(iter(data_loaders[j+1]))
            X_j, y_j = X_j.to(device), y_j.to(device)


            # Compute source domain loss and gradients
            y_pred_j = model(X_j.view(-1, 784))
            f_j_w_t = criterion(y_pred_j, y_j.squeeze().long())
            f_j_w_t.backward()
            grad_w_1_j_t = model.w_1.grad.clone()
            non_zero_count = torch.sum(grad_w_1_j_t != 0).item()
            grad_w_2_j_t = model.w_2.grad.clone()

            # Compute loss for previous time step
            model_prev.load_state_dict(previous_model_state_dict)
            model_prev.zero_grad()
            y_pred_j_minus_1 = model_prev(X_j.view(-1, 784))
            f_j_w_t_minus_1 = criterion(y_pred_j_minus_1, y_j.squeeze().long())
            f_j_w_t_minus_1.backward()
            
            f_j_values[j].append(f_j_w_t)

            # Compute z_t_plus_1 for source j
            z_t_plus_1[j] = (1 - beta_t) * (z_t[j] + f_T_w_t - f_j_w_t - (f_T_w_t_minus_1 - f_j_w_t_minus_1)) \
                            + beta_t * (f_T_w_t - f_j_w_t)

            # Compute the j-th coordinate of vector v using g(z)
            v[j] = torch.sqrt(z_t[j]**2 + c)

            # Compute gradient using the chain rule
            grad_w_1_j = (z_t_plus_1[j] / torch.sqrt(z_t_plus_1[j]**2 + c)) * ((grad_w_1_T_t - grad_w_1_j_t)**2)
            grad_w_2_j = (z_t_plus_1[j] / torch.sqrt(z_t_plus_1[j]**2 + c)) * ((grad_w_2_T_t - grad_w_2_j_t)**2)
            
            g_w_1_t += alpha[j] * grad_w_1_j
            g_w_2_t += alpha[j] * grad_w_2_j




        # Update z_t
        z_t = z_t_plus_1

        # Compute the norm of the gradient with respect to w
        w_1_gradient_norm = torch.norm(g_w_1_t).item()
        w_1_gradient_norms.append(w_1_gradient_norm)

        w_2_gradient_norm = torch.norm(g_w_2_t).item()
        w_2_gradient_norms.append(w_2_gradient_norm)

        # Update w_t_minus_1
        w_1_t_minus_1 = model.w_1.clone().detach()
        w_2_t_minus_1 = model.w_2.clone().detach()

        previous_model_state_dict = copy.deepcopy(model.state_dict())

        with torch.no_grad():
            model.w_1 += gamma * g_w_1_t
            scale_factor = min(1, 1 / torch.norm(model.w_1, p='fro'))
            model.w_1 *= scale_factor


        with torch.no_grad():
            model.w_2 += gamma * g_w_2_t
            scale_factor = min(1, 1 / torch.norm(model.w_2, p='fro'))
            model.w_2 *= scale_factor



        v = v.to(device) # Move to the same device

        # Compute gradient for alpha
        g_alpha_t = v + 2 * C * torch.matmul(M, alpha)

        # Compute the norm of the gradient with respect to alpha
        alpha_gradient_norms.append(g_alpha_t.detach().cpu().numpy())

        with torch.no_grad():
            alpha -= eta * g_alpha_t
            alpha = project_simplex(alpha)
            # Record alpha values for plotting
    
            assert torch.all(alpha >= 0), "Negative values found in alpha!"
            assert torch.isclose(alpha.sum(), torch.tensor(1.0), atol=1e-6), "Alpha does not sum to 1!"
            alpha_history.append(copy.deepcopy(alpha.detach().cpu().numpy()))

        # Record gradients
        gradients_norms.append(w_1_gradient_norm + w_2_gradient_norm + g_alpha_t.detach().cpu().numpy())

    





filename = "F_values.txt"
np.savetxt(os.path.join(output_directory, filename), np.array(F_values))


filename = "Alpha_values.txt"
np.savetxt(os.path.join(output_directory, filename), np.array(alpha_history))



alpha_history = np.array(alpha_history)
plt.figure(figsize=(10, 6))
for i in range(N):
    plt.plot(alpha_history[:, i], label=f'Alpha {i + 1}')
plt.axhline(y=1/N, color='r', linestyle='--', label='1/N')
plt.xlabel('Iteration')
plt.ylabel('Alpha Values')
plt.legend()
filename = "Alpha_values.png"
plt.savefig(os.path.join(output_directory, filename))
plt.close()



alpha_history_array = np.array(alpha_history)

final_alpha = alpha_history_array[-1].reshape(1, -1)

plt.figure(figsize=(12, 2))

plt.imshow(final_alpha, cmap='viridis', aspect='auto', vmin=0, vmax=5/N)

for i in range(1, N):
    plt.axvline(x=i - 0.5, color='black', linewidth=1)
plt.colorbar(label='Alpha Value')

plt.xticks(range(final_alpha.shape[1]), [f'Source {i+1}' for i in range(final_alpha.shape[1])], rotation=45, ha='right')

plt.yticks([])
plt.tight_layout(pad=2) 
filename = "Alpha_Values_Heatmap.png"
plt.savefig(os.path.join(output_directory, filename))
