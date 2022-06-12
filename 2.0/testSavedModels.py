import numpy as np
import matplotlib.pyplot as plt
import torch
import torchvision
import torchvision.transforms as transforms
from fishDataset import fishDataset
from torch.utils.data import Dataset, DataLoader
from CNNmodels import Net, vgg16
import torch.optim as optim

classes = ("ALB", "BET", "DOL", "LAG", "NoF", "OTHER", "SHARK", "YFT")


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((155, 227)),
    transforms.ToTensor(),
    transforms.Normalize((0.4042, 0.4353, 0.3998), (0.2251, 0.2185, 0.2127))
])

batch_size = 4

dataset = fishDataset("FishBoxes/Fishes/",
                      "FishBoxes/labels.csv", trans=transform)

# il dataset che uso per il test è diverso dal dataset di test usato durante il training sul quale il modello ha ottenuto una accuracy
# quindi l'accuracy sarà probabilmente maggiore (devo salvarmi il dataset di test?)
# train_size = int(0.8 * len(dataset))
# test_size = len(dataset) - train_size
# trainset, testset = torch.utils.data.random_split(
#     dataset, [train_size, test_size])

testloader = DataLoader(
    dataset, batch_size=batch_size, shuffle=True)

dataiter = iter(testloader)
images, labels = dataiter.next()

images = images.to(device)

PATH = './2.0/savedModels/net_20e_88%.pth'
# net = vgg16()
net = Net()
net.load_state_dict(torch.load(PATH))
net.to(device)


# def imshow(img):
#     invTrans = transforms.Compose([transforms.Normalize(mean=[0., 0., 0.],
#                                                         std=[1/0.2251, 1/0.2185, 1/0.2127]),
#                                    transforms.Normalize(mean=[-0.4042, -0.4353, -0.3998],
#                                                         std=[1., 1., 1.]),
#                                    ])
#     img = img.cpu()
#     img = invTrans(img)
#     npimg = img.numpy()
#     plt.imshow(np.transpose(npimg, (1, 2, 0)))
#     plt.show()


# print('GroundTruth: ', ' '.join(f'{classes[labels[j]]:5s}' for j in range(4)))
# imshow(torchvision.utils.make_grid(images))

# outputs = net(images)

# outputs.to(device)

# _, predicted = torch.max(outputs, 1)

# print('Predicted: ', ' '.join(f'{classes[predicted[j]]:5s}'
#                               for j in range(4)))


correct = 0
total = 0
# since we're not training, we don't need to calculate the gradients for our outputs
with torch.no_grad():
    for data in testloader:
        images, labels = data
        images = images.to(device)
        labels = labels.to(device)
        # calculate outputs by running images through the network
        outputs = net(images)
        # the class with the highest energy is what we choose as prediction
        _, predicted = torch.max(outputs.data, 1)
        print(outputs.data)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(
    f'Accuracy of the network on the whole dataset: {100 * correct // total} %')


# prepare to count predictions for each class
correct_pred = {classname: 0 for classname in classes}
total_pred = {classname: 0 for classname in classes}

# again no gradients needed
with torch.no_grad():
    for data in testloader:
        images, labels = data
        images = images.to(device)
        labels = labels.to(device)
        outputs = net(images)
        _, predictions = torch.max(outputs, 1)
        # collect the correct predictions for each class
        for label, prediction in zip(labels, predictions):
            if label == prediction:
                correct_pred[classes[label]] += 1
            total_pred[classes[label]] += 1

accPerClass = {}
# print accuracy for each class
for classname, correct_count in correct_pred.items():
    accuracy = 100 * float(correct_count) / total_pred[classname]
    accPerClass[classname] = accuracy

    # print(f'Accuracy for class: {classname:5s} is {accuracy:.1f} %')

d_view = [(v, k) for k, v in accPerClass.items()]
d_view.sort(reverse=True)  # natively sort tuples by first element
for v, k in d_view:
    # print("%s: %d" % (k, v))
    print(f'Accuracy for class: {k:6s} is {v:.1f} %')
