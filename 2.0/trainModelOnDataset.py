import numpy as np
import matplotlib.pyplot as plt
import torch
import torchvision
import torchvision.transforms as transforms
from fishDataset import fishDataset
from torch.utils.data import Dataset, DataLoader
from CNNmodels import Net, vgg16
import torch.optim as optim

PATH = './2.0/savedModels/net_10_2.pth'
epochs = 10
batch_size = 200


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((155, 227)),
    transforms.ToTensor(),
    transforms.Normalize((0.4042, 0.4353, 0.3998), (0.2251, 0.2185, 0.2127))
])


dataset = fishDataset("FishBoxes/Fishes/",
                      "FishBoxes/labels.csv", trans=transform)


train_size = int(0.7 * len(dataset))
test_size = len(dataset) - train_size
trainset, testset = torch.utils.data.random_split(
    dataset, [train_size, test_size])

trainloader = DataLoader(
    trainset, batch_size=batch_size, shuffle=True)
testloader = DataLoader(
    testset, batch_size=batch_size, shuffle=False)

classes = ("ALB", "BET", "DOL", "LAG", "NoF", "OTHER", "SHARK", "YFT")


net = Net()
# net = vgg16()
net.to(device)

criterion = torch.nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

for epoch in range(epochs):
    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        # get the inputs
        inputs, labels = data[0].to(device), data[1].to(device)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += float(loss.item())
        if i % 5 == 0:    # print every 10 mini-batches
            print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}')
            running_loss = 0.0
    print("---End of epoch " + str(epoch+1) + "---")

print('Finished Training')

torch.save(net.state_dict(), PATH)

print("Saved model to: " + PATH)
