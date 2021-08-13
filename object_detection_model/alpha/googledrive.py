import json
import os
import cv2
import preprocess_data

if __name__ == "__main__":

   path = os.getcwd()

   input_path = f"{os.getcwd()}/annotations/test_annotations.csv"
   save_path = f"{os.getcwd()}/data"
   
   gt_dataset = preprocess_data.preprocessing_label(input_path,save_path)

   """
   file = open(f"{path}/data/gt_dataset.txt")
   gt_dataset = json.load(file)

   file.close()
   """
   for name in list(gt_dataset.keys()):

      img = cv2.imread(f"{os.getcwd()}/images/{name}")

      cv2.imwrite(f"{os.getcwd()}/googledrive/{name}",img)
