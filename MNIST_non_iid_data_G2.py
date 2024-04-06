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



# Move to GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyperparameters
N = 15  # Number of domains
B = 32 # Batch size
C = 0.1 # Regularization parameter
T = 6000 # Number of iterations
gamma_init = 0.1
eta_init = 0.005
c = 0.1 # Parameter for g function
beta_t = 0.8 # Beta
# Momentum hyperparameters
momentum_w = 0.9
momentum_alpha = 0.9
# Learning rate decay
decay_rate = 0.95
decay_step = 100
# Learning rate decay for alpha
alpha_decay_rate = 0.95
alpha_decay_step = 200

# Learning rate schedule
def learning_rate_schedule(t, initial_lr, T):
    return initial_lr / (1 + t / (0.1 * T))

def generate_random_non_iid_data(batch_size, num_source_domains, train=True, test_split=0.2):
    # Transformations applied to the MNIST data
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])

    # Load the MNIST dataset
    mnist_data = datasets.MNIST('./Neurips_mnist_non_iid/g2/data', train=train, download=True, transform=transform)
    
    # Sort mnist_data by target
    mnist_data = sorted(list(mnist_data), key=lambda x: x[1])

    # Create a dictionary to hold the data loaders
    data_loaders = {}
    
     # Define classes for each group of source domains
    group_1_classes = [0, 1, 2]
    group_2_classes = [3, 4, 5]
    group_3_classes = [6, 7, 8, 9]

    # Calculate the number of domains in each group
    num_domains_group_1 = num_source_domains // 3
    print(num_domains_group_1)
    num_domains_group_2 = num_source_domains // 3
    num_domains_group_3 = num_source_domains - num_domains_group_1 - num_domains_group_2
  

    # Extract data for each domain based on the class groups
    group_1_data = [(data, target) for data, target in mnist_data if target in group_1_classes]
    group_2_data = [(data, target) for data, target in mnist_data if target in group_2_classes]
    group_3_data = [(data, target) for data, target in mnist_data if target in group_3_classes]

    random.shuffle(group_1_data)
    random.shuffle(group_2_data)
    random.shuffle(group_3_data)
    # Calculate samples per domain for each group
    samples_per_domain_group_1 = 100
    samples_per_domain_group_2 = 100
    samples_per_domain_group_3 = 100
    samples_target_train = 100
    samples_target_test = 20
  
    g1_num = 0
    g2_num = 0
    g3_num = 0

    for i in range(1, num_source_domains + 1):
        if i <= num_domains_group_1:

            start_idx = i* samples_per_domain_group_1
           
            end_idx = (i+1) * samples_per_domain_group_1
     
            domain_data = group_1_data[start_idx:end_idx]
            g1_num += 1
        elif i <= num_domains_group_1 + num_domains_group_2:

            start_idx = (i - num_domains_group_1 + 1) * samples_per_domain_group_2
            
            end_idx = (i - num_domains_group_1 + 2) * samples_per_domain_group_2
           
            domain_data = group_2_data[start_idx:end_idx]
            g2_num += 1
        else:
            start_idx = (i - num_domains_group_1 - num_domains_group_2 + 1) * samples_per_domain_group_3
          
            end_idx = (i - num_domains_group_1 - num_domains_group_2 + 2) * samples_per_domain_group_3
            
            domain_data = group_3_data[start_idx:end_idx]
            g3_num += 1


        # Create DataLoader for the domain
        data_loaders[i] = DataLoader(domain_data, batch_size=batch_size, shuffle=True)

    random.shuffle(group_2_data)
    print("g1_num: ",g1_num)
    print("g2_num: ",g2_num)
    print("g3_num: ",g3_num)

    target_train_data = group_2_data[:samples_target_train]
    random.shuffle(group_2_data)
    target_test_data = group_2_data[samples_target_train:samples_target_train+samples_target_test]
    data_loaders['target_train'] = DataLoader(target_train_data, batch_size=batch_size, shuffle=True)
    random.shuffle(group_2_data)
    data_loaders['target_test'] = DataLoader(target_test_data, batch_size=batch_size, shuffle=True)

    return data_loaders

output_directory = './Neurips_mnist_non_iid/g2'

os.makedirs(output_directory, exist_ok=True)

data_loaders = generate_random_non_iid_data(B, N)
output_directory_data = './Neurips_mnist_non_iid/g2/datasets'

os.makedirs(output_directory_data, exist_ok=True)
with open(os.path.join(output_directory_data, 'data_loaders_mnist_non_iid.pkl'), 'wb') as f:
    pickle.dump(data_loaders, f)

with open(os.path.join(output_directory_data, 'data_loaders_mnist_non_iid.pkl'), 'rb') as f:
    data_loaders = pickle.load(f)



def count_classes(data):
    if hasattr(data, '__iter__') and not isinstance(data, (list, tuple)):
        labels = []
        for _, batch_labels in data:
            labels.extend(batch_labels)
    else:
        labels = data
    
    # Count occurrences of each class
    counts = {i: labels.count(i) for i in [3, 4, 5]}
    return counts

# Example dictionary for demonstration
# data_loaders_example = {
#     'target_train': [0, 1, 2, 0, 2, 2, 1, 0, 0, 1, 1, 2],
#     1: [0, 1, 2, 2, 1, 0],
#     2: [0, 0, 1, 1, 1, 2, 2, 2]
# }

class_counts = {key: count_classes(data) for key, data in data_loaders.items()}
print(class_counts)
print(len(data_loaders['target_train'].dataset), len(data_loaders['target_test'].dataset))








