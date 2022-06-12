import os

# Costruisco il file labels.csv

labelsName = ["ALB", "BET", "DOL", "LAG", "NoF", "OTHER", "SHARK", "YFT"]
with open("FishBoxes/labels.csv", "w") as f:
    f.write("label,image\n")
    for label in labelsName:
        for img in os.listdir("FishBoxes/Fishes/" + label):
            f.write(label + "," + img + "\n")
