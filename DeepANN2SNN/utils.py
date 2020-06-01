import torch
from DeepANN2SNN.modules import *
import matplotlib.pyplot as plt
import numpy as np
import os

forward_list = ['LinearwithBN', 'Conv2dwithBN', 'SpkGenerator', 'Dropout', 'Resize1d','MaxPool2d','AvgPool2d','Softmax','Linear','Conv2d']

def show(x,name):
    if not os.path.exists('fig'):
        os.mkdir('fig')
    print(name)
    #print(x)
    print('range:[',np.min(x),',',np.max(x),']')
    plt.imshow(x)
    plt.savefig("fig/"+name)
    #plt.show()

def manual_scale(m,scale):
    if hasattr(m, 'weight_nobn') and m.weight_nobn is not None:
        m.weight_nobn.data = m.weight_nobn.data * scale
    else:
        m.weight.data = m.weight.data * scale
    if hasattr(m, 'bias_nobn') and m.bias_nobn is not None:
        m.bias_nobn.data = m.bias_nobn.data * scale
    else:
        m.bias.data = m.bias.data * scale

def manual_scale_v2(m,scale):
    if hasattr(m,'forward_scale'):
        m.forward_scale = scale * m.forward_scale
    else:
        if hasattr(m, 'weight_nobn') and m.weight_nobn is not None:
            m.weight_nobn.data = m.weight_nobn.data * scale
        else:
            m.weight.data = m.weight.data * scale
        if hasattr(m, 'bias_nobn') and m.bias_nobn is not None:
            m.bias_nobn.data = m.bias_nobn.data * scale
        else:
            m.bias.data = m.bias.data * scale

def getpercentile(hist,axis,percentile):
    sum = np.sum(hist)
    tgt = percentile * sum
    s = 0
    for i in range(hist.shape[0]):
        s += hist[i]
        if s>=tgt:
            break
    n = i - 1
    r = (1 - (s - tgt) / sum)
    return axis[n] + (axis[n+1] - axis[n])*r


def normalize_with_bias(net,trainloader,device,robust=True):
    if robust:
        print('Using robust normalization...')
        amin = 0
        amax = 20
        bins = 100
        axis = np.linspace(amin,amax,bins)
        hist_dict = dict()
    else:
        print('Using weight normalization...')

    print('normalize with bias...')
    for n,m in net.named_modules():
        if hasattr(m,'transBatchNorm'):
            m.transBatchNorm()

    activation_max = dict()
    for n,m in net.named_modules():
        if m.__class__.__name__ in forward_list:
            activation_max[n] = -1e5
    net.eval()
    correct = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        x = inputs.to(device)
        for n,m in net.named_modules():
            if m.__class__.__name__ in forward_list: # forward
                # x = m(x)
                if hasattr(m, 'ann_forward_nobn'):
                    x = F.relu(m.ann_forward_nobn(x))
                elif hasattr(m, 'ann_forward_scale'):
                    x = m.ann_forward_scale(x)
                elif hasattr(m, 'ann_forward'):
                    x = F.relu(m.ann_forward(x))
                else:
                    x = m(x)
                #print(m.__class__.__name__,x.size())
                if m.__class__.__name__ in forward_list:
                    activation_max[n] = max(activation_max[n],torch.max(x))
                    if robust:
                        hist,axis = np.histogram(x.detach().cpu().numpy(), bins=bins, range=(amin, amax))
                        if n not in hist_dict.keys():
                            hist_dict[n] = hist
                        else:
                            hist_dict[n] = hist_dict[n] + hist

        pred = x.data.max(1, keepdim=True)[1]
        correct += pred.cpu().eq(targets.data.view_as(pred)).sum()

    if robust:
        for n in hist_dict.keys():
            print('before:',activation_max[n])
            activation_max[n]=getpercentile(hist_dict[n], axis, 0.99)
            print(activation_max[n])
    #print(getpercentile(hist,axis,1.0))
    print('before: ',activation_max)
    print(correct)
    last_lambda = 0
    for n,m in net.named_modules():
        if m.__class__.__name__ == 'SpkGenerator':
            last_lambda = activation_max[n]
        if m.__class__.__name__ in ['LinearwithBN', 'Conv2dwithBN']:
            if hasattr(m,'weight_nobn') and m.weight_nobn is not None:
                print(n,m.__class__.__name__,'before:',torch.max(m.weight_nobn))
                m.weight_nobn.data = m.weight_nobn.data * last_lambda / activation_max[n]
                print(n, m.__class__.__name__, 'after:', torch.max(m.weight_nobn))
            if hasattr(m,'bias_nobn') and m.bias_nobn is not None:
                m.bias_nobn.data = m.bias_nobn.data / activation_max[n]
            last_lambda = activation_max[n]
        if m.__class__.__name__ in ['Linear','Conv2d']:
            if hasattr(m,'weight') and m.weight is not None:
                m.weight.data = m.weight.data * last_lambda / activation_max[n]
            if hasattr(m,'bias') and m.bias is not None:
                m.bias.data = m.bias.data / activation_max[n]
            last_lambda = activation_max[n]
        # if m.__class__.__name__ in ['MaxPool2d','AvgPool2d']:
        #     if hasattr(m, 'forward_scale'):
        #         m.forward_scale = last_lambda / activation_max[n]
        #         last_lambda = activation_max[n]

    # validate:
    act_max = dict()
    correct = 0
    for n,m in net.named_modules():
        if m.__class__.__name__ in ['LinearwithBN', 'Conv2dwithBN', 'SpkGenerator', 'Dropout', 'Resize1d','MaxPool2d','AvgPool2d']:
            act_max[n] = -1e5
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        x = inputs.to(device)
        for n, m in net.named_modules():
            if m.__class__.__name__ in ['LinearwithBN', 'Conv2dwithBN', 'SpkGenerator', 'Dropout', 'Resize1d',
                                        'MaxPool2d', 'AvgPool2d']:  # forward
                if hasattr(m,'ann_forward_nobn'):
                    x = F.relu(m.ann_forward_nobn(x))
                elif hasattr(m,'ann_forward_scale'):
                    x = m.ann_forward_scale(x)
                elif hasattr(m,'ann_forward'):
                    x = F.relu(m.ann_forward(x))
                else:
                    x = m(x)
                if m.__class__.__name__ in ['LinearwithBN', 'Conv2dwithBN', 'SpkGenerator', 'Dropout', 'Resize1d',
                                            'MaxPool2d', 'AvgPool2d']:
                    act_max[n] = max(act_max[n], torch.max(x))
        pred = x.data.max(1, keepdim=True)[1]
        correct += pred.cpu().eq(targets.data.view_as(pred)).sum()
    print(correct)
    print('after: ',act_max)

    #exit(-1)

def check(net,trainloader,device):
    idx = 45
    input = trainloader.dataset[idx][0].to(device)
    output = torch.LongTensor(trainloader.dataset[idx][1]).to(device)
    with torch.no_grad():
        net.visualize_forward(input, trainloader.dataset[idx][1])

    # input = torch.rand(1,3,32,32).to(device)
    # s = SpkGenerator(1000)
    # s.to(device)
    # s.device = device
    # s.mode = 'SNN'
    # m = MaxPool2d(2,2)
    # m.to(device)
    # m.device = device
    # m.mode = 'SNN'
    # m.snn_reset()
    # s.snn_reset()
    # for t in range(100):
    #     o = s(input.view(1,3,32,32))
    #     o = m(o)
    # #print(m.spk_cnt.size())
    # o1 = s.spk_cnt.view(3,32,32).detach().cpu().numpy()
    # o2 = m.spk_cnt.view(3,16,16).detach().cpu().numpy()
    # plt.imshow(o1[1,:,:])
    # plt.show()
    # plt.imshow(o2[1,:,:])
    # plt.show()
    #
    # m2 = torch.nn.MaxPool2d(2,2)
    # p = m2(input.view(1,3,32,32))
    # p1 = input.view(3,32,32).detach().cpu().numpy()
    # p2 = p.view(3,16,16).detach().cpu().numpy()
    # plt.imshow(p1[1, :, :])
    # plt.show()
    # plt.imshow(p2[1, :, :])
    # plt.show()


def snn_val(net, epoch, testloader, device, criterion):
    net.eval()
    tmp_mode = net.mode
    net.set_working_mode('SNN')
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            print('Epoch', epoch, 'SNNVal', batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                  % (test_loss / (batch_idx + 1), 100. * correct / total, correct, total))
            if batch_idx ==10:
                break
    net.set_working_mode(tmp_mode)
    return correct,total

def ann_val(net, epoch, testloader, device, criterion):
    net.eval()
    tmp_mode = net.mode
    net.set_working_mode('ANN')
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

        print('Epoch', epoch, 'ANNVal', batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
              % (test_loss / (batch_idx + 1), 100. * correct / total, correct, total))
    net.set_working_mode(tmp_mode)
    return correct,total
