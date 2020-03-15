import torchvision
import torchvision.transforms as transforms
import time
import torch
import torch.nn as nn
import ANN2SNN.Diehl2015.model_fc as models


def mnist_transform():
    return transforms.Compose([transforms.ToTensor()])

train_batch_size = 100
train_dataset = torchvision.datasets.MNIST(root="D:\Dataset\mnist", train=True, download=True, transform=mnist_transform())
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True)

val_dataset = torchvision.datasets.MNIST(root="D:\Dataset\mnist", train=False, download=False, transform=mnist_transform())
val_loader = torch.utils.data.DataLoader(train_dataset, batch_size=128, shuffle=False)

device = torch.device("cuda:0")
learning_rate = 1
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch
num_epochs = 1
acc_record = list([])
loss_train_record = list([])
loss_test_record = list([])

ANN = models.Network_ANN()
SNN = models.Network_SNN(time_window=35, threshold=1.0, max_rate=200)
ANN.to(device)
SNN.to(device)
optimizer = torch.optim.SGD(ANN.parameters(), lr=learning_rate)
criterion = nn.MSELoss().to(device)

for epoch in range(num_epochs): #TODO
    running_loss = 0
    start_time = time.time()
    for i, (inputs, targets) in enumerate(train_loader):
        batch_sz = inputs.size(0)
        ANN.zero_grad()
        optimizer.zero_grad()
        inputs = inputs.float().to(device)
        labels_ = torch.zeros(batch_sz, 10).scatter_(1, targets.view(-1, 1), 1).to(device)
        outputs = ANN(inputs)
        loss = criterion(outputs, labels_)
        running_loss += loss.cpu().item()
        loss.backward()
        optimizer.step()
        if (i+1) % 100 == 0:
             print ('Epoch [%d/%d], Step [%d/%d], Training Loss: %.5f Time elasped:%.2f s'
                    %(epoch+1, num_epochs, i+1, len(train_dataset)//train_batch_size,running_loss,time.time()-start_time))
             running_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(val_loader):
            batch_sz = inputs.size(0)
            inputs = inputs.float().to(device)
            labels_ = torch.zeros(batch_sz, 10).scatter_(1, targets.view(-1, 1), 1).to(device)
            outputs = ANN(inputs)
            targets = targets.to(device)
            loss = criterion(outputs, labels_)
            _, predicted = outputs.max(1)
            total += float(targets.size(0))
            correct += float(predicted.eq(targets).sum().cpu().item())
    print("ANN Test Accuracy: %.3f" % (100 * correct / total))
    acc = 100. * float(correct) / float(total)
    acc_record.append(acc)

    correct = 0
    total = 0
    SNN.load_state_dict(ANN.state_dict())
    for batch_idx, (inputs, targets) in enumerate(val_loader):
        batch_sz = inputs.size(0)
        targets = targets.to(device)
        inputs = inputs.float().to(device)
        outputs = SNN(inputs)
        _, predicted = outputs.max(1)
        total += float(targets.size(0))
        correct += float(predicted.eq(targets).sum().cpu().item())
    print("SNN Test Accuracy: %.3f" % (100 * correct / total))

ANN.normalize_nn(train_loader)
for i,value in enumerate(ANN.factor_log):
    print('Normalization Factor for Layer %d: %3.5f'%(i,value))
correct = 0
total = 0
SNN_normalized = models.Network_SNN(time_window=35, threshold=1.0, max_rate=1000)
SNN_normalized.to(device)
SNN_normalized.load_state_dict(ANN.state_dict())
for batch_idx, (inputs, targets) in enumerate(val_loader):
    batch_sz = inputs.size(0)
    targets = targets.to(device)
    inputs = inputs.float().to(device)
    outputs = SNN_normalized(inputs)
    _, predicted = outputs.max(1)
    total += float(targets.size(0))
    correct += float(predicted.eq(targets).sum().cpu().item())
print("Normalized SNN Test Accuracy: %.3f" % (100 * correct / total))