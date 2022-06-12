import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.io import read_image
import pandas as pd
import torchvision.transforms as T


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
    T.Resize((227, 155)),
    T.ToTensor(),
])

dataset = fishDataset("FishBoxes/Fishes/",
                      "FishBoxes/labels.csv", trans=transformation)


train_dataloader = DataLoader(
    dataset, batch_size=64)


def get_mean_and_std(dataloader):
    channels_sum, channels_squared_sum, num_batches = 0, 0, 0
    for data, _ in dataloader:
        data = torch.Tensor.float(data)
        # Mean over batch, height and width, but not over the channels
        channels_sum += torch.mean(data, dim=[0, 2, 3])
        channels_squared_sum += torch.mean(data**2, dim=[0, 2, 3])
        num_batches += 1

    mean = channels_sum / num_batches

    # std = sqrt(E[X^2] - (E[X])^2)
    std = (channels_squared_sum / num_batches - mean ** 2) ** 0.5

    return mean, std


print(get_mean_and_std(train_dataloader))

T.Normalize((0.4044, 0.4356, 0.3999), (0.2244, 0.2178, 0.2120))