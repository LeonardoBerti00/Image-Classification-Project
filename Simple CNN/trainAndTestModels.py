import numpy as np
import matplotlib.pyplot as plt
import torch
import torchvision
import torchvision.transforms as T
from fishDataset import fishDataset
from torch.utils.data import Dataset, DataLoader
from CNNmodels import *
import torch.optim as optim
from math import floor
import time


def trainModelOnData(trainLoader, model, epochs=10, batch_size=32, lrate=0.001, momen=0.9, device="cpu", testLoader=False):
    """
    Train a model on data
    :param trainLoader: data to train on
    :param model: model to train
    :param epochs: number of epochs to train
    :param batch_size: batch size to use
    :param lrate: learning rate to use
    :param momen: momentum to use for SGD
    :device: device to use
    :return: trained model
    """

    print(f'Training model for {epochs} epochs on {batch_size} batch size:')

    start_time = time.time()

    model.to(device)

    criterion = torch.nn.CrossEntropyLoss()

    optimizer = optim.SGD(model.parameters(), lr=lrate, momentum=momen)

    for epoch in range(epochs):
        running_loss = 0.0
        print(f'Epoch {epoch+1}:')
        for i, data in enumerate(trainloader, 0):
            # get the inputs
            inputs, labels = data[0].to(device), data[1].to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += float(loss.item())
            if i % 20 == 0:    # print every 10 mini-batches
                print(
                    f'   loss: {running_loss / 2000:.4f}')
                running_loss = 0.0

        if(testLoader != False and epoch != epochs-1):
            print(
                f'Accuracy of the network on epoch {epoch+1}: {testModelOnData(testloader, model, device):.1f} %\n')

    print(f'Finished training in {time.time() - start_time:.1f} seconds')
    return model


def testModelOnData(data, model, device="cpu"):
    """
    Test a model on data
    :param data: data to test on
    :param model: model to test
    :param device: device to use
    :return: accuracy of model on data
    """

    correct = 0
    total = 0
    # since we're not training, we don't need to calculate the gradients for our outputs
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            images = images.to(device)
            labels = labels.to(device)
            # calculate outputs by running images through the network
            outputs = model(images)
            # the class with the highest energy is what we choose as prediction
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total

    return accuracy


def loadFishDataset(imagesPath, labelsPath, split=0.8, transformations=[], numWork=0):
    """
    Load a dataset
    :param path: path to dataset
    :param split: split to load
    :return: train and test dataLoaders
    """
    normalization = [
        T.ToPILImage(),
        T.Resize((imgHeight, imgWidth)),
        T.ToTensor(),
        T.Normalize((0.4042, 0.4353, 0.3998), (0.2251, 0.2185, 0.2127))
    ]
    if transformations:
        for t in transformations:
            normalization.insert(-2, t)

    normalization = T.Compose(normalization)

    dataset = fishDataset(imagesPath, labelsPath, trans=normalization)

    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    trainset, testset = torch.utils.data.random_split(
        dataset, [train_size, test_size])

    trainloader = DataLoader(
        trainset, batch_size=batch_size, shuffle=True, num_workers=numWork)
    testloader = DataLoader(
        testset, batch_size=batch_size, shuffle=False, num_workers=numWork)

    return trainloader, testloader


if __name__ == "__main__":
    # standardizzare i dati di test e di train (in csv separati così i risultati dovrebbero essere più consistenti)
    model = Net()   # Net() or vgg16()
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    dataSplit = 0.8
    numWorkers = 0  # [0..6]
    epochs = 20
    batch_size = 64
    imgWidth = 227
    imgHeight = 155
    lrate = 0.001
    momentum = 0.9
    transformations = []
    testWhileTraining = True
    saveModel = True
    modelName = "net_20e"

    trainloader, testloader = loadFishDataset(
        "FishBoxes/Fishes/", "FishBoxes/labels.csv", dataSplit, transformations, numWorkers)

    if testWhileTraining:
        model = trainModelOnData(
            trainloader, model, epochs, batch_size, lrate, momentum, device, testloader)
    else:
        model = trainModelOnData(
            trainloader, model, epochs, batch_size, lrate, momentum, device)

    accuracy = testModelOnData(testloader, model, device)

    print(
        f'Accuracy of the final network on the test images: {accuracy:.1f} %')

    if saveModel:
        path = "./2.0/savedModels/" + modelName + \
            "_" + str(floor(accuracy)) + "%.pth"
        torch.save(model.state_dict(), path)
        print("Saved model to: " + path)
