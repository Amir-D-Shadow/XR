import pandas as pd
import numpy as np
import json
from Kmean import Kmean_IOU
import os
from numba import jit

def preprocess_class(input_path,save_path ,name="class_map.txt"):

   """
   MS COCO 2017 Dataset

   return dict
   """
   dataset = pd.read_csv(input_path)

   #get class map
   class_array = dataset.iloc[:,3]
   class_map = {}
   idx = 0
   
   for i in range(class_array.shape[0]):

      if not (class_array[i] in class_map.keys()):

         class_map[class_array[i]] = idx
         idx = idx + 1

   #save class map 
   with open(f"{save_path}/{name}","w") as file:
      
     file.write(json.dumps(class_map))

   return class_map

def preprocess_bbox_info(path,path_pos,path_hw,name_pos="bbox_pos.txt",name_hw="bbox_hw.txt"):

   """
   MS COCO 2017 Dataset

   return numpy.ndarray,bbox_pos,bbox_hw
   """

   dataset = pd.read_csv(path)

   """
   #find center
   dataset["center_x"] = (dataset["xmax"] + dataset["xmin"])/2
   dataset["center_y"] = (dataset["ymax"] + dataset["ymin"])/2
   """

   #get positional data
   bbox_pos = dataset.iloc[:,4:].to_numpy()

   #save bbox_pos
   with open(f"{path_pos}/{name_pos}","w") as file:

      file.write(json.dumps(bbox_pos.tolist()))

   #construct bbox_hw (m,2) : h -> 0 , w -> 1
   m = bbox_pos.shape[0]
   bbox_hw = np.zeros((m,2))

   for i in range(m):

      bbox_hw[i,0] = bbox_pos[i,3] - bbox_pos[i,1]
      bbox_hw[i,1] = bbox_pos[i,2] - bbox_pos[i,0]


   #save bbox_hw 
   with open(f"{path_hw}/{name_hw}","w") as file:

      file.write(json.dumps(bbox_hw.tolist()))


   return bbox_pos,bbox_hw


def preprocess_pre_define_anchor_box(bbox_hw,save_path,K=9,name="anchors.txt"):

   """
   bbox_hw -- numpy.ndarray (m,2)
   """

   anchors = Kmean(bbox_hw,K)

   with open(f"{save_path}/{name}","w") as file:

      file.write(json.dumps(anchors.tolist()))

   
   return anchors

def preprocessing_label(input_path,save_path,name="gt_dataset.txt"):

   """
   save dict as {obj1:[[class,xmin,ymin,xcenter,ycenter],[class,xmin,ymin,xcenter,ycenter],...],obj2:...} (for each key)
   """

   #read csv file
   dataset = pd.read_csv(input_path)
   m = dataset.shape[0]

   #calibrate bbox pos
   dataset["xmin"] = dataset["xmin"] + (640 - dataset["width"] )//2
   dataset["xmax"] = dataset["xmax"] + (640 - dataset["width"] )//2

   dataset["ymin"] = dataset["ymin"] + (640 - dataset["height"] )//2
   dataset["ymax"] = dataset["ymax"] + (640 - dataset["height"] )//2

   #calculate center
   dataset["xcenter"] = (dataset["xmin"] + dataset["xmax"])/2
   dataset["ycenter"] = (dataset["ymin"] + dataset["ymax"])/2

   #Group the label
   gt_dataset = {}
   
   for i in range(m):

      filename = dataset.iloc[i,0]
      tmp = []

      #class
      tmp.append(dataset.iloc[i,3])

      #xmin xmax ymin ymax
      tmp.append(dataset.iloc[i,4].item())
      tmp.append(dataset.iloc[i,5].item())
      tmp.append(dataset.iloc[i,8].item())
      tmp.append(dataset.iloc[i,9].item())

      if not ( filename in gt_dataset.keys()  ):

         gt_dataset[filename] = []
         
      gt_dataset[filename].append(tmp)


   #save dataset
   with open(f"{save_path}/{name}","w") as file:

      file.write(json.dumps(gt_dataset))

      file.close()

   return gt_dataset
   
def preprocess_image(img,standard_shape=(640,640)):

   """
   img -- numpy.ndarray
   standard_shape -- (height,width)
   """
   #get h,w
   height,width = standard_shape

   #get pad size
   padH = (height - img.shape[0])//2
   padW = (width - img.shape[1])//2

   #pad img
   diff_H = height - img.shape[0]
   diff_W = width - img.shape[1]
   
   if (diff_H % 2 ) == 0 and (diff_W % 2) == 0:

      img_pad = np.pad(img,((padH,padH),(padW,padW),(0,0)),mode="constant",constant_values=(0,0))

   elif (diff_H % 2 ) != 0 and (diff_W % 2) == 0:

      img_pad = np.pad(img,((padH,padH+1),(padW,padW),(0,0)),mode="constant",constant_values=(0,0))

   elif (diff_H % 2 ) == 0 and (diff_W % 2) != 0:

      img_pad = np.pad(img,((padH,padH),(padW,padW+1),(0,0)),mode="constant",constant_values=(0,0))

   elif (diff_H % 2 ) != 0 and (diff_W % 2) != 0:

      img_pad = np.pad(img,((padH,padH+1),(padW,padW+1),(0,0)),mode="constant",constant_values=(0,0))

   return img_pad
   


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
   

if __name__ == "__main__":

   path = os.getcwd()
   
   
   #class map
   #class_map = preprocess_class(f"{path}/test_set.csv",f"{os.getcwd()}/data")

   """
   file = open(f"{path}/data/class_map.txt")
   a = json.load(file)

   file.close()

   #bbox
   bbox_pos,bbox_hw = preprocess_bbox_info(f"{path}/valid_set.csv",f"{os.getcwd()}/data",f"{os.getcwd()}/data")

   file = open(f"{path}/data/bbox_pos.txt")
   b = json.load(file)

   file = open(f"{path}/data/bbox_hw.txt")
   c = json.load(file)

   file.close()

   #pre-define anchor box
   anchors = preprocess_pre_define_anchor_box(bbox_hw,f"{os.getcwd()}/data")
   """
   """
   file = open(f"{path}/data/anchors.txt")
   anchors = json.load(file)
   file.close()
   
   preprocess_y_true(f"{path}/valid_set.csv",f"{path}/y_true",np.array(anchors),class_map)
   """

   #gt_dataset = preprocessing_label(f"{path}/test_set.csv",f"{path}/data")
   
