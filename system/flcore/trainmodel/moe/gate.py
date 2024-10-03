import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.init as init
# Define the gating model
class Gating(nn.Module):
    def __init__(self, input_dim,
                num_experts, dropout_rate=0.1):
        super(Gating, self).__init__()

        # Layers
        self.layer1=nn.Linear(input_dim, 128)
        self.dropout1=nn.Dropout(dropout_rate)

        self.layer2=nn.Linear(128, 256)
        
        self.activation1=nn.LeakyReLU()
        # nn.LeakyReLU()
        # nn.ReLU()
        # nn.PReLU()
        # nn.ELU()
        # nn.SELU()
        # nn.SiLU()
        # nn.Mish()
        self.dropout2=nn.Dropout(dropout_rate)

        self.layer3=nn.Linear(256, 128)
        self.activation2=nn.LeakyReLU()
        self.dropout3=nn.Dropout(dropout_rate)

        self.layer4=nn.Linear(128, num_experts)
        
        init.orthogonal_(self.layer1.weight)
        init.orthogonal_(self.layer2.weight)
        init.orthogonal_(self.layer3.weight)
        init.orthogonal_(self.layer4.weight)
        
        

        

    def forward(self, x):
        x=torch.relu(self.layer1(x))
        x=self.dropout1(x)

        x=self.layer2(x)
        x=self.activation1(x)
        x=self.dropout2(x)

        x=self.layer3(x)
        x=self.activation2(x)
        x=self.dropout3(x)

        return torch.softmax(self.layer4(x), dim=1)

# Define the gating model
class CNNGating(nn.Module):
    def __init__(self, in_channels,
                num_experts, dropout_rate=0.1):
        super(CNNGating, self).__init__()

        # Layers
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=5, padding=0, stride=1, bias=True),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(3, 3))
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, num_experts, kernel_size=5, padding=0, stride=1, bias=True),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 2))
        )
        self.layer=nn.Linear(16, 4)

    def forward(self, x): # 10,3,32,32
        x = self.conv1(x) # 10,32,9,9
        x = self.conv2(x) # 10,4,2,2
        x = torch.flatten(x, 1)
        x = self.layer(x)
        return torch.softmax(x, dim=1) # [10, 4, 5, 5]


class Gating2Layer(nn.Module):
    def __init__(self, input_dim,
                num_experts, dropout_rate=0.1):
        super(Gating2Layer, self).__init__()

        # Layers
        self.layer1=nn.Linear(input_dim, 128)
        self.layer2=nn.Linear(128, num_experts)

    def forward(self, x):
        x=torch.relu(self.layer1(x))
        # x=self.dropout1(x)

        # x=self.layer2(x)
        # x=self.leaky_relu1(x)
        # x=self.dropout2(x)

        # x=self.layer3(x)
        # x=self.leaky_relu2(x)
        # x=self.dropout3(x)

        return torch.softmax(self.layer2(x), dim=1)
    