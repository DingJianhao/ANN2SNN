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
        self.conv1 = nn.Conv2d(in_channels=1,out_channels=16,kernel_size=5,stride=1,padding=0,bias=False)
        self.HalfRect1 = nn.ReLU()
        self.dropout1 = nn.Dropout(0.5)
        self.subsample1 = nn.AvgPool2d(2,2,0)
        self.conv2 = nn.Conv2d(in_channels=16,out_channels=16,kernel_size=5,stride=1,padding=0,bias=False)
        self.HalfRect2 = nn.ReLU()
        self.dropout2 = nn.Dropout(0.5)
        self.subsample2 = nn.AvgPool2d(2, 2, 0)
        self.fc1 = nn.Linear(256, 10, bias=False)
        self.HalfRect3 = nn.ReLU()

    def to(self, device):
        self.device = device
        super().to(device)
        return self

    def forward(self, input):
        x = self.conv1(input)
        x = self.HalfRect1(x)
        x = self.dropout1(x)
        x = self.subsample1(x)
        x = self.conv2(x)
        x = self.HalfRect2(x)
        x = self.dropout2(x)
        x = self.subsample2(x)
        x = x.view(-1,256)
        x = self.fc1(x)
        x = self.HalfRect3(x)
        return x

    def normalize_nn(self, train_loader):
        conv1_weight_max = torch.max(F.relu(self.conv1.weight))
        conv2_weight_max = torch.max(F.relu(self.conv2.weight))
        fc1_weight_max = torch.max(F.relu(self.fc1.weight))
        conv1_activation_max = 0.0
        conv2_activation_max = 0.0
        fc1_activation_max = 0.0

        self.eval()
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            x = inputs.float().to(self.device)
            x = self.dropout1(self.HalfRect2(self.conv1(x)))
            conv1_activation_max = max(conv1_activation_max, torch.max(x))
            x = self.subsample1(x)
            x = self.dropout2(self.HalfRect2(self.conv2(x)))
            conv2_activation_max = max(conv2_activation_max, torch.max(x))
            x = self.subsample2(x)
            x = x.view(-1, 256)
            x = self.HalfRect3(self.fc1(x))
            fc1_activation_max = max(fc1_activation_max, torch.max(x))
        self.train()

        self.factor_log = []
        previous_factor = 1

        scale_factor = max(conv1_weight_max, conv1_activation_max)
        applied_inv_factor = (scale_factor / previous_factor).item()
        self.conv1.weight.data = self.conv1.weight.data / applied_inv_factor
        self.factor_log.append(1/ applied_inv_factor)
        previous_factor = applied_inv_factor

        scale_factor = max(conv2_weight_max, conv2_activation_max)
        applied_inv_factor = (scale_factor / previous_factor).item()
        self.conv2.weight.data = self.conv2.weight.data / applied_inv_factor
        self.factor_log.append(1 / applied_inv_factor)
        previous_factor = applied_inv_factor

        scale_factor = max(fc1_weight_max, fc1_activation_max)
        applied_inv_factor = (scale_factor / previous_factor).item()
        self.fc1.weight.data = self.fc1.weight.data / applied_inv_factor
        self.factor_log.append(1 / applied_inv_factor)
        previous_factor = applied_inv_factor

class Network_SNN(nn.Module):
    def __init__(self, time_window=35, threshold=1.0, max_rate=200):
        super(Network_SNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=5, stride=1, padding=0, bias=False)
        self.HalfRect1 = nn.ReLU()
        self.dropout1 = nn.Dropout(0.5)
        self.subsample1 = nn.AvgPool2d(2, 2, 0)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=5, stride=1, padding=0, bias=False)
        self.HalfRect2 = nn.ReLU()
        self.dropout2 = nn.Dropout(0.5)
        self.subsample2 = nn.AvgPool2d(2, 2, 0)
        self.fc1 = nn.Linear(256, 10, bias=False)
        self.HalfRect3 = nn.ReLU()

        self.threshold = threshold
        self.time_window = time_window
        self.dt = 0.001  # second
        self.refractory_t = 0
        self.max_rate = max_rate
        self.rescale_factor = 1.0 / (self.dt * self.max_rate)

    def to(self, device):
        self.device = device
        super().to(device)
        return self

    def mem_update(self, t,  operator, mem, input_spk, leak=0, refrac_end=None):
        # Get input impulse from incoming spikes
        impulse = operator(input_spk)
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

        spk_input = spksum_input = torch.zeros(batch_size, 1,28,28, device=self.device)
        mem_post_conv1 = spk_post_conv1 = spksum_post_conv1 = torch.zeros(batch_size, 16,24,24, device=self.device)
        mem_post_subsample1 = spk_post_subsample1 = spksum_post_subsample1 = torch.zeros(batch_size, 16,12,12, device=self.device)
        mem_post_conv2 = spk_post_conv2 = spksum_post_conv2 = torch.zeros(batch_size, 16,8,8, device=self.device)
        mem_post_subsample2 = spk_post_subsample2 = spksum_post_subsample2 = torch.zeros(batch_size, 16, 4, 4, device=self.device)
        mem_post_fc1 = spk_post_fc1 = spksum_post_fc1 = torch.zeros(batch_size, 10, device=self.device)

        for t in range(self.time_window):
            spk_input = (torch.rand(input.size(), device=self.device) * self.rescale_factor <= input).float()
            spksum_input = spksum_input + spk_input

            mem_post_conv1, spk_post_conv1 = self.mem_update(t, self.conv1, mem_post_conv1, spk_input)
            spksum_post_conv1 = spksum_post_conv1 + spk_post_conv1

            mem_post_subsample1, spk_post_subsample1 = self.mem_update(t, self.subsample1, mem_post_subsample1, spk_post_conv1)
            spksum_post_subsample1 = spksum_post_subsample1 + spk_post_subsample1

            mem_post_conv2, spk_post_conv2 = self.mem_update(t, self.conv2, mem_post_conv2, spk_post_subsample1)
            spksum_post_conv2 = spksum_post_conv2 + spk_post_conv2

            mem_post_subsample2, spk_post_subsample2 = self.mem_update(t, self.subsample2, mem_post_subsample2,
                                                                       spk_post_conv2)
            spksum_post_subsample2 = spksum_post_subsample2 + spk_post_subsample2

            spk_post_subsample2_ = spk_post_subsample2.view(batch_size,256)
            mem_post_fc1, spk_post_fc1 = self.mem_update(t, self.fc1, mem_post_fc1, spk_post_subsample2_)
            spksum_post_fc1 = spksum_post_fc1 + spk_post_fc1
        outputs = spksum_post_fc1 / self.time_window
        return outputs


if __name__ == "__main__":
    from torch.autograd import Variable
    features = []
    def hook(self, input, output):
        print(output.data.cpu().numpy().shape)
        features.append(output.data.cpu().numpy())
    net = Network_ANN()
    net.to(torch.device("cpu"))
    net.train()

    for w in net.named_parameters():
        print(w[0])

    for m in net.modules():
        m.register_forward_hook(hook)

    y = net(Variable(torch.randn(11, 1, 28, 28)))
    print(y.size())