import torch
import torch.nn as nn
from torch.nn import functional as F, init

import importlib
# Import custom linear layer
custom_linear = importlib.import_module("deep-codegen.pytorch_apis").custom_linear_with_bias

# LeNet-300-100 model
class LeNet(torch.nn.Module):
    # Initialize weights and biases
    def __init__(self, kernel='Custom'):
        super(LeNet, self).__init__()
        self.kernel = kernel
        if (self.kernel == 'Custom'):
            self.W1 = nn.Parameter(init.xavier_uniform_(torch.empty(300, 28*28, requires_grad=True)))
            self.b1 = nn.Parameter(torch.zeros(300, requires_grad=True))
            self.W2 = nn.Parameter(init.xavier_uniform_(torch.empty(100, 300, requires_grad=True)))
            self.b2 = nn.Parameter(torch.zeros(100, requires_grad=True))
            self.W3 = nn.Parameter(init.xavier_uniform_(torch.empty(10, 100, requires_grad=True)))
            self.b3 = nn.Parameter(torch.zeros(10, requires_grad=True))
        else:
            self.W1 = nn.Parameter(init.xavier_uniform_(torch.empty(28*28, 300, requires_grad=True)))
            self.b1 = nn.Parameter(torch.zeros(300, requires_grad=True))
            self.W2 = nn.Parameter(init.xavier_uniform_(torch.empty(300, 100, requires_grad=True)))
            self.b2 = nn.Parameter(torch.zeros(100, requires_grad=True))
            self.W3 = nn.Parameter(init.xavier_uniform_(torch.empty(100, 10, requires_grad=True)))
            self.b3 = nn.Parameter(torch.zeros(10, requires_grad=True))
    # Define layers
    def forward(self, x):
        x = x.view(x.size(0), -1)  # Flatten input
        if (self.kernel == 'Custom'):
            # Model layers with Custom Cuda linear function
            x = F.relu(custom_linear(x, self.W1, self.b1)) # First custom layer
            x = F.relu(custom_linear(x, self.W2, self.b2)) # Second custom layer
            x = F.relu(custom_linear(x, self.W3, self.b3)) # Third custom layer
        else:
            # Model layers with PyTorch linear layer
            x = F.relu(x.mm(self.W1) + self.b1) # First PyTorch layer
            x = F.relu(x.mm(self.W2) + self.b2) # Second PyTorch layer
            x = F.relu(x.mm(self.W3) + self.b3) # Third PyTorch layer
        return F.softmax(x, dim=1) # Output layer
