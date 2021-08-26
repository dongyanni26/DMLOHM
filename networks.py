import torch
import torch.nn as nn 
import torch.nn.functional as F
import math
class EmbeddingNet(nn.Module):
    @staticmethod
    def weight_init(m):
        # Initialize the trainable parameters
        if isinstance(m, nn.Linear) or isinstance(m, nn.Conv1d):
            torch.nn.init.uniform_(m.weight, -0.05, 0.05)
            torch.nn.init.zeros_(m.bias)

    def _get_final_flattened_size(self):
        with torch.no_grad():
            x = torch.zeros(1, 1, self.input_channels)
            x = self.pool(self.conv(x))
        return x.numel()

    def __init__(self, input_channels, featuredim, kernel_size=None, pool_size=None):
        super(EmbeddingNet, self).__init__()
        if kernel_size is None:
           kernel_size = math.ceil(input_channels / 9)
        if pool_size is None:
           pool_size = math.ceil(kernel_size / 5)
        self.input_channels = input_channels

        self.conv = nn.Conv1d(1, 20, kernel_size)
        self.pool = nn.MaxPool1d(pool_size)
        self.features_size = self._get_final_flattened_size()

        # [n4 is set to be 100]
        self.fc1 = nn.Linear(self.features_size, featuredim)
        self.apply(self.weight_init)

    def forward(self, x):
        x = x.squeeze(dim=-1).squeeze(dim=-1)
        x = x.unsqueeze(1)
        x = self.conv(x)
        x = self.pool(x)
        x = x.view(-1, self.features_size)
        x = self.fc1(x)
        return x
    
    def get_embedding(self, x):
        return self.forward(x)

class Classifier(nn.Module):
    def _get_final_flattened_size(self):
        with torch.no_grad():
            x = torch.zeros(1, 1, self.input_channels)
            x = self.pool(self.conv(x))
        return x.numel()

    def __init__(self, input_channels, featuredim, n_classes, kernel_size=None, pool_size=None):
        super(Classifier, self).__init__()
        if kernel_size is None:
           kernel_size = math.ceil(input_channels / 9)
        if pool_size is None:
           pool_size = math.ceil(kernel_size / 5)
        self.input_channels = input_channels

        self.conv = nn.Conv1d(1, 20, kernel_size)
        self.pool = nn.MaxPool1d(pool_size)
        self.features_size = self._get_final_flattened_size()

        # [n4 is set to be 100]
        self.fc1 = nn.Linear(self.features_size, featuredim)

        self.output = nn.Linear(featuredim, n_classes)

    def forward(self, x):
        x = x.squeeze(dim=-1).squeeze(dim=-1)
        x = x.unsqueeze(1)
        x = self.conv(x)
        x = self.pool(x)
        x = x.view(-1, self.features_size)
        x = torch.tanh(self.fc1(x))
        x = self.output(x)
        return x

class TripletNet(nn.Module):
    def __init__(self, embedding_net):
        super(TripletNet, self).__init__()
        self.embedding_net = embedding_net

    def forward(self, x1, x2, x3):
        output1 = self.embedding_net(x1)
        output2 = self.embedding_net(x2)
        output3 = self.embedding_net(x3)
        return output1, output2, output3

    def get_embedding(self, x):
        return self.embedding_net(x)