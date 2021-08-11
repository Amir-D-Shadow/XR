import cv2
import json
import os
from matplotlib import pyplot as plt
import numpy as np

def draw_anchor_box(img,feat,class_color_map):

  """
  img -- numpy_array
  feat -- list [class,left_x,left_y,center_x,center_y]
  """

  left_x = feat[1]
  left_y = feat[2]

  width = (feat[3]-feat[1])*2
  height = (feat[4]-feat[2])*2

  right_x = feat[1] + width
  right_y = feat[2] + height

  updated_img = cv2.rectangle(img,(left_x,left_y),(right_x,right_y),class_color_map[feat[0]],2)

  return updated_img


def preprocess_class_color_map(class_info,save_path,name="class_color_map.txt"):

  num_class = len(list(class_info.keys()))

  step = int(256*3/num_class)

  class_color_map = {}
  x,y,z = 0,0,0

  for k in class_info.values():

    if x < 255:

      x = x + step

    elif y < 255:

      y = y + step

    elif z < 255:

      z = z + step

    if x > 255:

      x = 255

    elif y > 255:

      y = 255

    elif z > 255:

      z = 255

    class_color_map[k] = (x,y,z)

  with open(f"{save_path}/{name}","w") as file:

    file.write(json.dumps(class_color_map))

    file.close()

  return class_color_map


def reverse_class_info(class_info,save_path,name="reversed_class_map.txt"):

  reversed_class_map = {}

  for k in class_info.keys():

    val = class_info[k]

    reversed_class_map[val] = k

  with open(f"{save_path}/{name}","w") as file:

    file.write(json.dumps(reversed_class_map))

    file.close()

  return reversed_class_map



   
