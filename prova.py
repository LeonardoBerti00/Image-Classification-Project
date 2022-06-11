from torchvision.transforms.transforms import Normalize
import torch
import torchvision.transforms as T
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets
import matplotlib.pyplot as plt
from torchvision.io import read_image
import pandas as pd
import torch.nn.functional as F
import torchvision.models as models


#vgg16 = models.vgg16(pretrained=True)

#define hyperparameters
batchSize = 8
epochs = 5
learningRate = 1e-4
momentum = 0.1
width = 400     # 800
height = 237    # 474
num_classes = 7

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
        #print(type(img))
        imgFinal = img.float() / 255
        #print("daje")
        # print("{}\t{}".format(img_path, img.shape), end="\n")

        if self.transforms:
            imgFinal = self.transforms(imgFinal)

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

        return imgFinal, label


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
    #T.Normalize(),
    # T.RandomPerspective(p=0.5),
    #T.RandomRotation(degrees=(0, 360)),
    #T.RandomHorizontalFlip(p=0.5),
    #T.RandomVerticalFlip(p=0.5),
    #T.RandomAffine(degrees=0.0, translate=(0.1, 0.3)),
    # T.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5),
    T.Grayscale(),
    T.ToTensor(),
    T.Normalize((0.4219), (0.2113))
    ])

dataset = fishDataset("FishBoxes/Fishes/",
                      "FishBoxes/labels.csv", trans=transformation)

train_size = int(1.0 * len(dataset))
test_size = len(dataset) - train_size
training_data, test_data = torch.utils.data.random_split(dataset, [train_size, test_size])

train_dataloader = DataLoader(training_data, batch_size=batchSize, shuffle=True)
test_dataloader = DataLoader(test_data, batch_size=batchSize, shuffle=False)

# define the device for the computation
device = "cuda" if torch.cuda.is_available() else "cpu"

def ComputeMeanandSTD(train_dataloader):
  #to Normalize = x - mean / std
  channels_sum, channels_squared_sum, num_batches = 0, 0, 0
  for data, _ in train_dataloader:
      # Mean over batch, height and width, but not over the channels
      channels_sum += torch.mean(data, dim=[0,2,3])
      channels_squared_sum += torch.mean(data**2, dim=[0,2,3])
      num_batches += 1
      
  mean = channels_sum / num_batches

  # std = sqrt(E[X^2] - (E[X])^2)
  std = (channels_squared_sum / num_batches - mean ** 2) ** 0.5
  # output
  print('mean: '  + str(mean))
  print('std:  '  + str(std))
  return mean, std

ComputeMeanandSTD(train_dataloader)