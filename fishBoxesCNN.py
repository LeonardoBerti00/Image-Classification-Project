import torch
import torchvision.transforms as T
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets
import matplotlib.pyplot as plt
from torchvision.io import read_image
import pandas as pd
import torch.nn.functional as F

torch.cuda.empty_cache()

# accuracy: 60.6% without NoF
# accuracy: 54.1% with NoF

batchSize = 100
epochs = 100
learningRate = 1e-3
momentum = 0.9
width = 227     # 227
height = 155    # 155

# Aggiungere più immagini per le classi con pochi sample (shark, DOL)


class fishDataset(Dataset):
    def __init__(self, img_dir, labels_dir, trans=None):

        # labels stored in a csv file, each line has the form namefile,label
        # img1.png,dog
        # img2.png,cat
        self.labels = pd.read_csv(labels_dir)
        self.img_dir = img_dir
        self.transforms = trans

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):

        # getting the label
        label = self.labels.iloc[idx, 0]

        image = self.labels.iloc[idx, 1]

        # reading the image
        img_path = self.img_dir + str(label) + "/" + str(image)
        img = read_image(img_path)
        # print("{}\t{}".format(img_path, img.shape), end="\n")

        if self.transforms:
            img = self.transforms(img)

        # plt.imshow(img.permute(1, 2, 0))
        # print(label)
        # plt.show()

        # return the image with the corresponding label
        if label == "ALB":
            label = 0
        elif label == "BET":
            label = 1
        elif label == "DOL":
            label = 2
        elif label == "LAG":
            label = 3
        elif label == "NoF":
            label = 4
        elif label == "OTHER":
            label = 5
        elif label == "SHARK":
            label = 6
        elif label == "YFT":
            label = 7

        return img, label


transformation = T.Compose([
    T.ToPILImage(),
    # T.TrivialAugmentWide(),
    # T.RandomCrop(size=(width, height)),
    # T.RandomInvert(p=0.5),
    # T.RandomPosterize(bits=2, p=0.5),
    # T.RandomAdjustSharpness(sharpness_factor=2, p=0.5),
    # T.RandomAutocontrast(p=0.5),
    # T.RandomEqualize(p=0.5),
    # T.AutoAugment(),
    T.Resize((height, width)),
    # T.RandomPerspective(p=0.5),
    # T.RandomRotation(degrees=(0, 360)),
    # T.RandomHorizontalFlip(p=0.5),
    # T.RandomVerticalFlip(p=0.5),
    # T.RandomAffine(degrees=0.0, translate=(0.1, 0.3)),
    # T.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5),
    # T.Grayscale(),
    T.ToTensor(),
    T.Normalize((0.4042, 0.4353, 0.3998), (0.2251, 0.2185, 0.2127))
])

dataset = fishDataset("FishBoxes/Fishes/",
                      "FishBoxes/labels.csv", trans=transformation)

train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
training_data, test_data = torch.utils.data.random_split(
    dataset, [train_size, test_size])

train_dataloader = DataLoader(
    training_data, batch_size=batchSize, shuffle=True)
test_dataloader = DataLoader(test_data, batch_size=batchSize, shuffle=False)

# define the device for the computation
device = "cuda" if torch.cuda.is_available() else "cpu"

# define our CNN


# class myCNN(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.vgg = nn.Sequential(
#             nn.Conv2d(3, 64, 3, padding=1),
#             nn.ReLU(),
#             nn.Conv2d(64, 64, 3, padding=1),
#             nn.ReLU(),

#             nn.MaxPool2d(2, stride=2),

#             nn.Conv2d(64, 128, 3, padding=1),
#             nn.ReLU(),
#             nn.Conv2d(128, 128, 3, padding=1),
#             nn.ReLU(),

#             nn.MaxPool2d(2, stride=2),

#             nn.Conv2d(128, 256, 3, padding=1),
#             nn.ReLU(),
#             nn.Conv2d(256, 256, 3, padding=1),
#             nn.ReLU(),
#             nn.Conv2d(256, 256, 1),
#             nn.ReLU(),

#             nn.MaxPool2d(2, stride=2),

#             nn.Conv2d(256, 512, 3, padding=1),
#             nn.ReLU(),
#             nn.Conv2d(512, 512, 3, padding=1),
#             nn.ReLU(),
#             nn.Conv2d(512, 512, 1),
#             nn.ReLU(),

#             nn.MaxPool2d(2, stride=2),

#             nn.Conv2d(512, 512, 3, padding=1),
#             nn.ReLU(),
#             nn.Conv2d(512, 512, 3, padding=1),
#             nn.ReLU(),
#             nn.Conv2d(512, 512, 1),
#             nn.ReLU(),

#             nn.MaxPool2d(2, stride=2))

#         self.dense = nn.Sequential(
#             nn.Linear(14336, 4096),
#             nn.ReLU(),
#             nn.Linear(4096, 4096),
#             nn.ReLU(),
#             nn.Linear(4096, 8),
#             # nn.ReLU()
#         )

#     def forward(self, x):
#         out = self.vgg(x)
#         out = torch.flatten(out, 1)
#         out = self.dense(out)
#         return out

class myCNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(3, 20, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(20, 16, 5)
        self.fc1 = nn.Linear(29680, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 8)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


# instance of our model
model = myCNN()
model.to(device)

# hyperparameter settings
# learning_rate = 0.001
# epochs = 5

# loss function definition
loss_fn = nn.CrossEntropyLoss()


# optimizer definition
optimizer = torch.optim.SGD(
    model.parameters(), lr=learningRate, momentum=momentum)
# optimizer = torch.optim.AdamW(model.parameters(), learningRate)

# defining the training loop


def trainingLoop(train_dataloader, model, loss_fn, optimizer):

    for batch, (X, y) in enumerate(train_dataloader):
        # move data on gpu
        X = X.float()
        optimizer.zero_grad()
        loss = loss_fn(model(X.to(device)), y.to(device))
        # backpropagation

        loss.backward()
        optimizer.step()

        if batch % 5 == 0:
            print(f"The loss is {loss.item()}")


acc = 0


def testLoop(test_dataloader, model, loss_fn):
    print_size = len(test_dataloader.dataset)
    num_batches = len(test_dataloader)
    test_loss = 0
    correct = 0

    with torch.no_grad():
        for X, y in test_dataloader:
            X, y = X.float().to(device), y.to(device)

            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    test_loss = test_loss/num_batches
    correct = correct / print_size

    print(f"Accuracy: {correct * 100}, Average loss: {test_loss}")
    acc = correct * 100


for e in range(epochs):
    print("------------Start of Epoch {}/{}------------".format(e, epochs-1))
    trainingLoop(train_dataloader, model, loss_fn, optimizer)
    testLoop(test_dataloader, model, loss_fn)
    print("------------End of Epoch {}/{}------------".format(e, epochs-1))

if acc > 54:
    modelName = "model_NoF_" + str(acc) + ".pth"
    torch.save(model.state_dict(), modelName)
    print("Model saved as " + modelName)
