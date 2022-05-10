# https://github.com/pytorch/examples/blob/master/mnist/main.py
from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
from PIL import Image
# Training settings
batch_size = 20

transform = transforms.Compose([
                                transforms.Scale(256),
                                transforms.CenterCrop(224),
                                transforms.ToTensor()])

img_data = datasets.ImageFolder('./flower_photos/flower_photos',
                                transform=transform)
                                

train_loader = torch.utils.data.DataLoader(img_data, batch_size=batch_size,shuffle=True)

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.mp = nn.MaxPool2d(2)
        self.fc1 = nn.Linear(56180, 5)

    def forward(self, x):
        in_size = x.size(0)
        x = F.relu(self.mp(self.conv1(x)))
        x = F.relu(self.mp(self.conv2(x)))
        x = x.view(in_size, -1)  # flatten the tensor
        x = self.fc(x)
        return x

model = Net()
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.5)

def train(epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = Variable(data), Variable(target)
        optimizer.zero_grad()
        output = model(data)
        loss = F.cross_entropy(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % 10 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))


# def test():
#     model.eval()
#     test_loss = 0
#     correct = 0
#     for data, target in test_loader:
#         data, target = Variable(data, volatile=True), Variable(target)
#         output = model(data)
#         # sum up batch loss
#         test_loss += F.nll_loss(output, target, size_average=False).data
#         # get the index of the max log-probability
#         pred = output.data.max(1, keepdim=True)[1]
#         correct += pred.eq(target.data.view_as(pred)).cpu().sum()

#     test_loss /= len(test_loader.dataset)
#     print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
#         test_loss, correct, len(test_loader.dataset),
#         100. * correct / len(test_loader.dataset)))

# for epoch in range(1, 10):
#     train(epoch)
#     #test()

# torch.save(model.state_dict(), 'model.pth')

print(img_data.class_to_idx)

model.load_state_dict(torch.load('model.pth'))
img = Image.open("./flower_photos/test/dandelion.jpg")
img = transform(img)
data = img.unsqueeze(0)
output = model(data)
print(output[0])