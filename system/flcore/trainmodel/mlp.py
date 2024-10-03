import torch
import torch.nn as nn


class Cifar10MLP(nn.Module):
    def __init__(self, in_features, hidden_size, output_size):
        super(Cifar10MLP, self).__init__()
        self.in_features = in_features
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.fc1 = nn.Linear(in_features, hidden_size) 
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = x.view(-1, self.fc1.in_features) 
        x = torch.relu(self.fc1(x))  
        x = self.fc2(x)
        return x
    
class MnistMLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(MnistMLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size) 
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = x.view(-1, self.fc1.in_features) 
        x = torch.relu(self.fc1(x))  
        x = self.fc2(x)
        return x
    