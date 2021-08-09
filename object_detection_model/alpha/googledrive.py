import json
import os
import cv2

if __name__ == "__main__":

   path = os.getcwd()

   file = open(f"{path}/data/gt_dataset.txt")
   gt_dataset = json.load(file)

   for name in list(gt_dataset.keys()):

      img = cv2.imread(f"{os.getcwd()}/images/{name}")

      cv2.imwrite(f"{os.getcwd()}/data/{name}",img)
