from fishDataset import fishDataset
import torch
import torchvision.transforms as T
from torch.utils.data import DataLoader
import torch.optim as optim
from math import floor
import time
from fishDataset import fishDataset
from CNNmodel import SimpleModel
import os


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
    # optimizer = optim.AdamW(model.parameters(), lr=lrate)

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


def testModelOnClasses(data, model, device="cpu"):
    classes = ("ALB", "BET", "DOL", "LAG", "NoF", "OTHER", "SHARK", "YFT")

    correct_pred = {classname: 0 for classname in classes}

    total_pred = {classname: 0 for classname in classes}

    # again no gradients needed
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
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

    return accPerClass


if __name__ == "__main__":
    # standardizzare i dati di test e di train (in csv separati così i risultati dovrebbero essere più consistenti)
    model = SimpleModel()
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    dataSplit = 0.8
    numWorkers = 0  # [0..6]
    epochs = 30
    batch_size = 64
    imgWidth = 227
    imgHeight = 155
    lrate = 0.001
    momentum = 0.9
    transformations = []
    testWhileTraining = True
    saveModel = True
    modelName = "simpleModelSGD"

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

    testModelOnClasses(testloader, model, device)

    if saveModel:
        modelName = f'{modelName}_{accuracy:.1f}%'
        savePath = "./SimpleCNN/SavedModels/" + modelName
        if not os.path.exists(savePath):
            os.makedirs(savePath)

        with open(savePath + "/" + modelName + ".txt", "w") as file:
            file.write(
                f'Accuracy of the final network on the test images: {accuracy:.1f} %\n\n')
            file.write(f'Train and test data split: {dataSplit*100:.0f}%\n')
            file.write(
                f'Size of the images during the training is : {imgWidth}x{imgHeight}\n')
            file.write(f'Epochs: {epochs}\n')
            file.write(f'Batch size: {batch_size}\n')
            file.write(f'Learning rate: {lrate}\n')
            file.write(f'Momentum: {momentum}\n')
            file.write(f'Transformations: {transformations.__str__()}\n')

        torch.save(model.state_dict(), savePath + "/" + modelName + ".pth")
        print("Saved model to: " + savePath + "/" + modelName + ".pth")
