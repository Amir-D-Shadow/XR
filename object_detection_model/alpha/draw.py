import cv2
import json
import os
from matplotlib import pyplot as plt
import numpy as np

def draw_anchor_box(img,feat,reversed_class_map,class_color_map):

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

  updated_img = cv2.putText(updated_img,reversed_class_map[feat[0]],(left_x,left_y),cv2.FONT_HERSHEY_DUPLEX,0.8,class_color_map[feat[0]],2,cv2.LINE_AA)

  return updated_img






   
