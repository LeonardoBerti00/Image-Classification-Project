# Image-Classification-Project
For our project we chose to try to solve a problem given as a competition on the site kaggle.com.
This is the competition -> https://www.kaggle.com/competitions/the-nature-conservancy-fisheries-monitoring/overview

In this competition the goal was to automatically detect and classify species of tunas, sharks and more that fishing vessels catch. The images are collected by analyzing frame by frame the footage captured by the on-board cameras. The ultimate goal is to ease the monitoring of fishing activities to fill science and compliance monitoring gaps in order to have a positive impact on the oceans.

In FishBoxes there is the dataset that we created, there are only the fishes extracted from the photos thanks to a public json file (FishBoxes/BoxesJsons) in which someone had created bounding boxes annotations on the entire training set using a software called Sloth. 

In CreateDataset there are the scripts that we wrote to create the dataset:
  - CreateDataset/fishDataset.py   this is the class we used to create the dataset to be given as input to the pytorch model 
  - CreateDataset/extractFish.py   this is the script we used to extract the fishes from the original dataset
  - CreateDataset/BuildDataset.py   this is the script we used to create the file with the labels -> FishBoxes/labels.csv

In Pre Trained Models there are the scrpts to load and utilize some pre trained pytorch models.

In VGG there is the our implementation of the famous CNN [VGG](https://arxiv.org/pdf/1409.1556.pdf).

In SimpleCNN there is an implementation of a simple CNN with 2 conv layers e 1 pool.

For more information, we wrote a short report with the methods used and the results:  https://docs.google.com/document/d/13B4Z0PlE7_ozwVeMN_uXxhOzC2ui6S9_dmj-0-3gPtk/edit
