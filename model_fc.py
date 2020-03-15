# Diehl, P.U. and Neil, D. and Binas, J. and Cook, M. and Liu, S.C. and Pfeiffer, M. Fast-Classifying, High-Accuracy Spiking Deep Networks Through Weight and Threshold Balancing, IEEE International Joint Conference on Neural Networks (IJCNN), 2015

import torchvision
import torchvision.transforms as transforms
import os
import time
import torch
import torch.nn as nn
import torch.nn.functional as F

class Network_ANN(nn.Module):
    def __init__(self):
        super(Network_ANN, self).__init__()
        self.fc1 = nn.Linear(784, 1200, bias=False)
        self.HalfRect1 = nn.ReLU()
        self.dropout1 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(1200, 1200, bias=False)
        self.HalfRect2 = nn.ReLU()
        self.dropout2 = nn.Dropout(0.5)
        self.fc3 = nn.Linear(1200, 10, bias=False)
        self.HalfRect3 = nn.ReLU()

    def to(self, device):
        self.device = device
        super().to(device)
        return self

    def forward(self, input):
        x = input.view(-1,784)
        x = self.fc1(x)
        x = self.HalfRect1(x)
        x = self.dropout1(x)
        x = self.fc2(x)
        x = self.HalfRect2(x)
        x = self.dropout2(x)
        x = self.fc3(x)
        x = self.HalfRect3(x)
        return x

    def normalize_nn(self, train_loader):
        fc1_weight_max = torch.max(F.relu(self.fc1.weight))
        fc2_weight_max = torch.max(F.relu(self.fc2.weight))
        fc3_weight_max = torch.max(F.relu(self.fc3.weight))
        fc1_activation_max = 0.0
        fc2_activation_max = 0.0
        fc3_activation_max = 0.0

        self.eval()
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            x = inputs.float().to(self.device).view(-1,784)
            x = self.dropout1(self.HalfRect1(self.fc1(x)))
            fc1_activation_max = max(fc1_activation_max, torch.max(x))
            x = self.dropout2(self.HalfRect2(self.fc2(x)))
            fc2_activation_max = max(fc2_activation_max, torch.max(x))
            x = self.HalfRect3(self.fc3(x))
            fc3_activation_max = max(fc3_activation_max, torch.max(x))
        self.train()

        self.factor_log = []
        previous_factor = 1

        scale_factor = max(fc1_weight_max, fc1_activation_max)
        applied_inv_factor = (scale_factor / previous_factor).item()
        self.fc1.weight.data = self.fc1.weight.data / applied_inv_factor
        self.factor_log.append(1/ applied_inv_factor)
        previous_factor = applied_inv_factor

        scale_factor = max(fc2_weight_max, fc2_activation_max)
        applied_inv_factor = (scale_factor / previous_factor).item()
        self.fc2.weight.data = self.fc2.weight.data / applied_inv_factor
        self.factor_log.append(1 / applied_inv_factor)
        previous_factor = applied_inv_factor

        scale_factor = max(fc3_weight_max, fc3_activation_max)
        applied_inv_factor = (scale_factor / previous_factor).item()
        self.fc3.weight.data = self.fc3.weight.data / applied_inv_factor
        self.factor_log.append(1 / applied_inv_factor)
        previous_factor = applied_inv_factor





class Network_SNN(nn.Module):
    def __init__(self, time_window=35, threshold=1.0, max_rate=200):
        super(Network_SNN, self).__init__()
        self.fc1 = nn.Linear(784, 1200, bias=False)
        self.HalfRect1 = nn.ReLU()
        self.dropout1 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(1200, 1200, bias=False)
        self.HalfRect2 = nn.ReLU()
        self.dropout2 = nn.Dropout(0.5)
        self.fc3 = nn.Linear(1200, 10, bias=False)
        self.HalfRect3 = nn.ReLU()

        self.threshold = threshold
        self.time_window = time_window
        self.dt = 0.001 # second
        self.refractory_t = 0
        self.max_rate = max_rate
        self.rescale_factor = 1.0/(self.dt*self.max_rate)

    def to(self, device):
        self.device = device
        super().to(device)
        return self

    def mem_update(self, t,  operator, mem, input_spk, leak=0, refrac_end=None):
        # Get input impulse from incoming spikes
        impulse = operator(input_spk)
        if refrac_end is not None:
            # Only allow non-refractory neurons to get input
            impulse = impulse * ((refrac_end <= t).float())
        # Add input to membrane potential
        mem = mem + impulse + leak
        # Check for spiking
        output_spk = (mem >= self.threshold).float()
        # Reset
        mem = mem * (1. - output_spk)
        # Ban updates until....
        if refrac_end is not None:
            refrac_end = output_spk * (t + self.refractory_t)
        return mem, output_spk

    def forward(self, input):
        batch_size = input.size(0)
        input = input.view(batch_size, 784)

        spk_input = spksum_input = torch.zeros(batch_size, 784, device=self.device)
        mem_post_fc1 = spk_post_fc1 = spksum_post_fc1 = torch.zeros(batch_size, 1200, device=self.device)
        mem_post_fc2 = spk_post_fc2 = spksum_post_fc2 = torch.zeros(batch_size, 1200, device=self.device)
        mem_post_fc3 = spk_post_fc3 = spksum_post_fc3 = torch.zeros(batch_size, 10, device=self.device)

        for t in range(self.time_window):
            spk_input = (torch.rand(input.size(), device=self.device) * self.rescale_factor <= input).float()
            spksum_input = spksum_input + spk_input

            mem_post_fc1 ,spk_post_fc1 = self.mem_update(t, self.fc1, mem_post_fc1, spk_input)
            spksum_post_fc1 = spksum_post_fc1 + spk_post_fc1

            mem_post_fc2, spk_post_fc2 = self.mem_update(t, self.fc2, mem_post_fc2, spksum_post_fc1)
            spksum_post_fc2 = spksum_post_fc2 + spk_post_fc2

            mem_post_fc3, spk_post_fc3 = self.mem_update(t, self.fc3, mem_post_fc3, spksum_post_fc2)
            spksum_post_fc3 = spksum_post_fc3 + spk_post_fc3
        outputs = spksum_post_fc3 / self.time_window
        return outputs