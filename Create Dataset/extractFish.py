import os
import json
import numpy as np
from PIL import Image
import PIL
from numpy import asarray
from math import floor
import cv2

# Estraggo il pesce dall'immagine originale e lo salvo in FishBoxes/Fishes/
# Il database originale non lo ho caricato su github

ALB = json.load(open("FishBoxes/ALB.json"))
BET = json.load(open("FishBoxes/BET.json"))
DOL = json.load(open("FishBoxes/DOL.json"))
LAG = json.load(open("FishBoxes/LAG.json"))
OTHER = json.load(open("FishBoxes/OTHER.json"))
SHARK = json.load(open("FishBoxes/SHARK.json"))
YFT = json.load(open("FishBoxes/YFT.json"))

fishes = {"ALB": ALB, "BET": BET, "DOL": DOL, "LAG": LAG,
          "OTHER": OTHER, "SHARK": SHARK, "YFT": YFT}

for fishType in fishes.keys():
    imgCount = 0
    for img in fishes[fishType]:
        img_path = "input/train/train/" + img["filename"]
        image = Image.open(img_path)
        image = asarray(image)
        for fishBox in img["annotations"]:
            imgCount += 1
            print("{}".format(img_path), end="\n")
            x1 = floor(fishBox["x"])
            y1 = floor(fishBox["y"])
            x2 = floor(x1 + fishBox["width"])
            y2 = floor(y1 + fishBox["height"])
            width = floor(fishBox["width"])
            height = floor(fishBox["height"])
            fish = image[y1:y2, x1:x2]
            if not fish.data:
                print("Empty fish")
            fish = cv2.resize(fish, dsize=(227, 155),
                              interpolation=cv2.INTER_CUBIC)
            # 400x237
            # print("1\t{}\t{}".format(width, height))
            # if height > 237:
            #     diff = height - 237
            #     height = 237
            #     width = floor(width - (diff * (width / (height+diff))))
            #     print("{}\t{}\t{}\t{}".format(width, height, x2 - x1, y2 - y1))

            # # print("2\t{}\t{}".format(width, height))
            # if width > 400:
            #     diff = width - 400
            #     width = 400
            #     height = floor(height - (diff * (height / (width+diff))))

            # print("{}\t{}\t{}\t{}".format(width, height, x2 - x1, y2 - y1))

            # fish = cv2.resize(fish, dsize=(width, height),
            #                   interpolation=cv2.INTER_CUBIC)

            # print("{}\t{}".format(237-height, 400-width), end="\n")
            # border = cv2.copyMakeBorder(
            #     fish,
            #     top=(237 - height) // 2,
            #     bottom=(237 - height) // 2,
            #     left=(400 - width) // 2,
            #     right=(400 - width) // 2,
            #     borderType=cv2.BORDER_CONSTANT,
            #     value=[0, 0, 0]
            # )
            # plt.savefig(
            #     "FishBoxes/fishes/{}/{}".format(fishType, "img_{}".format(imgCount)))
            Image.fromarray(fish).save(
                "FishBoxes/fishes/{}/{}.jpg".format(fishType, "img_{}".format(imgCount)))
            # plt.show()
        # plt.imshow(img)
        # plt.show()


# for img in ALB:
    print(img)
