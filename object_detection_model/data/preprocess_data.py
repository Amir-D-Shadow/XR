import pandas as pd
import numpy as np
import json
from Kmean import Kmean
import os
from numba import jit

def preprocess_class(path,path_class_map ,name="class_map.txt"):

   """
   MS COCO 2017 Dataset

   return dict
   """
   dataset = pd.read_csv(path)

   #get class map
   class_array = dataset.iloc[:,3]
   class_map = {}
   idx = 0
   
   for i in range(class_array.shape[0]):

      if not (class_array[i] in class_map.keys()):

         class_map[class_array[i]] = idx
         idx = idx + 1

   #save class map 
   with open(f"{path_class_map}/{name}","w") as file:
      
     file.write(json.dumps(class_map))

   return class_map

def preprocess_bbox_info(path,path_pos,path_hw,name_pos="bbox_pos.txt",name_hw="bbox_hw.txt"):

   """
   MS COCO 2017 Dataset

   return numpy.ndarray,bbox_hw
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

def preprocess_y_true(input_path,save_path,anchors,class_map,input_shape = (640,640),pos_info_format = [(76,76,255),(38,38,255),(19,19,255)],bbox_type=3):

   """
   anchors -- numpy.ndarray (K,2)
   input_shape -- (x,y)
   """

   #read csv file
   dataset = pd.read_csv(input_path)
   m = dataset.shape[0]

   #get center info
   dataset["x_center"] = (dataset["xmin"] + dataset["xmax"])/2
   dataset["y_center"] = (dataset["ymin"] + dataset["ymax"])/2

   #calibrate bbox pos
   for i in range(m):

      #image shape
      img_x = dataset.iloc[i,1]
      img_y = dataset.iloc[i,2]

      #calibrate
      padding_x = int((input_shape[0]-img_x)/2)
      padding_y = int((input_shape[1]-img_y)/2)

      dataset.iloc[i,4] = dataset.iloc[i,4] + padding_x
      dataset.iloc[i,5] = dataset.iloc[i,5] + padding_y
      dataset.iloc[i,6] = dataset.iloc[i,6] + padding_x
      dataset.iloc[i,7] = dataset.iloc[i,7] + padding_y

   #set up
   prev_name = "#"
   pos_format_size = len(pos_info_format)

   #loop through dataset
   for i in range(m):

      curr_name = dataset.iloc[i,0]
      class_name = dataset.iloc[i,3]

      if curr_name == prev_name:

         #update pos info
         update_pos_info(pos_info,(dataset.iloc[i,8],dataset.iloc[i,9]),dataset.iloc[i,4:8].to_numpy().reshape(1,4),class_map[dataset.iloc[i,3]],anchors)
      
      elif curr_name != prev_name and prev_name != "#":

         #save prev pos info
         for q in range(pos_format_size):
            
            h_size = pos_info_format[q][0]
            w_size = pos_info_format[q][1]
            
            with open(f"{save_path}/{curr_name}_{h_size}x{w_size}.txt","w") as file:

               file.write(json.dumps((pos_info[q]).tolist()))

               file.close()

         #crreate new pos info
         pos_info = [np.zeros(pos_info_format[0]),np.zeros(pos_info_format[1]),np.zeros(pos_info_format[2])]

         #update pos info
         update_pos_info(pos_info,(dataset.iloc[i,8],dataset.iloc[i,9]),dataset.iloc[i,4:8].to_numpy().reshape(1,4),class_map[dataset.iloc[i,3]],anchors)

      elif prev_name == "#":

         #crreate new pos info
         pos_info = [np.zeros(pos_info_format[0]),np.zeros(pos_info_format[1]),np.zeros(pos_info_format[2])]

         #update pos info
         update_pos_info(pos_info,(dataset.iloc[i,8],dataset.iloc[i,9]),dataset.iloc[i,4:8].to_numpy().reshape(1,4),class_map[dataset.iloc[i,3]],anchors)

      #update prev_name
      prev_name = curr_name

@jit(nopython=True)
def update_pos_info(pos_info,center_info,obj_pos,class_index,anchors,image_shape = (640,640),bbox_type = 3,feature_size = 85):

   """
   center_info -- (x,y)
   pos_info -- list containing numpy.ndarray
   obj_pos -- numpy.ndarray (1,4)
   anchors -- numpy.ndarray (K,2)
   image_shape -- (x,y)
   """
   #center_x center y
   center_x = center_info[0]
   center_y = center_info[1]
   
   #object bbox h w
   bbox_h = obj_pos[0,3] - obj_pos[0,1]
   bbox_w = obj_pos[0,2] - obj_pos[0,0]
   
   #find bext anchor index
   max_index = 0
   max_iou = 0
   
   for i in range(anchors.shape[0]):

      min_h = np.minimum(anchors[i,0],bbox_h).item()
      min_w = np.minimum(anchors[i,1],bbox_w).item()

      #intersection
      intersection_area = min_w * min_h

      #union
      union_area = bbox_h * bbox_w  + anchors[i,0] * anchors[i,1] - intersection_area

      #iou
      cur_iou = intersection_area / union_area

      if cur_iou > max_iou:

         max_iou = cur_iou

         max_index = i

   #size of particular type
   type_size = np.int64(anchors.shape[0]/bbox_type).item()
   
   #best box index -- dim 1 (determine which type of box)
   best_anchor_index = np.int64(max_index/type_size).item()
      
   #best box index -- dim 2 (in a particular type of box , determine sub class of particular type )
   sub_class_index = max_index % type_size

   #update info (prob,xmin,ymin,xmax,ymax,class)
   feature_vector = np.zeros(1,255)

   #prob
   feature_vector[0,0] = 1

   #xmin,ymin,xmax,ymax
   feature_vactor[0,1] = obj_pos[0,0]
   feature_vactor[0,2] = obj_pos[0,1]
   feature_vactor[0,3] = obj_pos[0,2]
   feature_vactor[0,4] = obj_pos[0,3]

   #class
   feature_vactor[0,(class_index+5)] = 1

   #feature mapping
   feature_per_image_pixel_x = pos_info[best_anchor_index].shape[0] / image_shape[0]
   feature_per_image_pixel_y = pos_info[best_anchor_index].shape[1] / image_shape[1]

   target_x = np.int64(center_x * feature_per_image_pixel_x).item()
   target_y = np.int64(center_y * feature_per_image_pixel_y).item()

   (pos_info[best_anchor_index])[target_y,target_x] = feature_vector
   

if __name__ == "__main__":

   path = os.getcwd()
   
   
   #class map
   class_map = preprocess_class(f"{path}/valid_set.csv",f"{os.getcwd()}/data")

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
   file = open(f"{path}/data/anchors.txt")
   anchors = json.load(file)
   
   preprocess_y_true(f"{path}/valid_set.csv",f"{path}/y_true",np.array(anchors),class_map)
