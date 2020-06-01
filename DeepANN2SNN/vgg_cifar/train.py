import torch.optim as optim

import torchvision
import torchvision.transforms as transforms

import os

from DeepANN2SNN.vgg_cifar.model import *
from DeepANN2SNN.utils import *

lr = 0.001
Epochs = 10
resume = True
nosave = True
notrain = True

# resume = False
# nosave = False
# notrain = False

robust = False

device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

# Data
print('==> Preparing data..')
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    #transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    #transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

trainset = torchvision.datasets.CIFAR10(
    root='D:/Dataset/cifar10', train=True, download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=128, shuffle=True)

testset = torchvision.datasets.CIFAR10(
    root='D:/Dataset/cifar10', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(
    testset, batch_size=100, shuffle=False)

classes = ('plane', 'car', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck')

# Model
print('==> Building model..')
net = VGG16(time_step=200)
net = net.to(device)
# for name,module in net.named_modules():
#     print(name,module.__class__.__name__)
# exit(-2)

# if device == 'cuda':
#     net = torch.nn.DataParallel(net)
#     cudnn.benchmark = True

if resume:
    # Load checkpoint.
    print('==> Resuming from checkpoint..')
    assert os.path.isdir('../checkpoint'), 'Error: no checkpoint directory found!'
    #checkpoint = torch.load('../checkpoint/vggcifar-ckpt.pth')
    checkpoint = torch.load('../checkpoint/vggcifar-acc90.pth')
    net.load_state_dict(checkpoint['net'])
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch']

criterion = nn.CrossEntropyLoss()
# optimizer = optim.SGD(net.parameters(), lr=lr,
#                       momentum=0.9, weight_decay=5e-4)
optimizer = optim.Adam(net.parameters(), lr=lr,weight_decay=5e-4)

# Training


def train(epoch):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        if (batch_idx+1) % 100==1:
            print('Epoch',epoch,'Train',batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                     % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))


def val(epoch):
    global best_acc
    correct,total = ann_val(net, epoch, testloader, device, criterion)
    #normalize_with_bias(net, trainloader, device)
    #snn_val(net, epoch, testloader, device, criterion)
    # Save checkpoint.
    if not nosave:
        acc = 100.*correct/total
        if acc > best_acc:
            print('Saving..')
            state = {
                'net': net.state_dict(),
                'acc': acc,
                'epoch': epoch,
            }
            if not os.path.isdir('../checkpoint'):
                os.mkdir('../checkpoint')
            torch.save(state, '../checkpoint/vggcifar-ckpt.pth')
            best_acc = acc

if not notrain:
    for epoch in range(start_epoch, start_epoch+Epochs):
        train(epoch)
        val(epoch)
#ann_val(net, 0, testloader, device, criterion)

with torch.no_grad():
    normalize_with_bias(net,testloader,device,robust=robust)
    snn_val(net, 0, testloader, device, criterion)
    check(net,testloader,device)

    # idx = 27
    # input = testloader.dataset[idx][0].to(device)
    # output = torch.LongTensor(testloader.dataset[idx][1]).to(device)
    # with torch.no_grad():
    #     net.finegrained_tune(input, output)
