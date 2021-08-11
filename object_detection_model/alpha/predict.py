import tensorflow as tf
import tensorflow.keras.backend as K
import numpy as np
import json
import cv2
import os


@tf.function
def step_predict(img,model):

   predictions = model(img,training=False)

   return predictions

def predict(model_path,image_path):


   #define model
   model = tf.keras.models.load_model(model_path)


   #read image
   img = cv2.imread(f"{image_path}/test.jpg")

   img = img / np.float64(255)

   img = np.pad(img,((108,108),(0,0),(0,0)),mode="constant",constant_values=(0,0))
   
   y = step_predict(img[np.newaxis,:,:,:],model)

   print(type(y))

   print(type(y[0]))

   return y


if __name__ == "__main__":

   path = os.getcwd()

   model_path = f"{path}/model"
   image_path = f"{path}/data"

   y = predict(model_path,image_path)
