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
        imgg = read_image(img_path)
        img = imgg.float() / 255
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
