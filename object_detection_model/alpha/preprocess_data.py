import pandas as pd
import numpy as np
import json
from Kmean import Kmean
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
   dataset["xmin"] = dataset["xmin"] + (640 - dataset["width"] )//2
   dataset["xmax"] = dataset["xmax"] + (640 - dataset["width"] )//2

   dataset["ymin"] = dataset["ymin"] + (640 - dataset["height"] )//2
   dataset["ymax"] = dataset["ymax"] + (640 - dataset["height"] )//2

   #set up
   prev_name = "#"
   pos_format_size = len(pos_info_format)

   #loop through dataset
   for i in range(m):

      curr_name = dataset.iloc[i,0]
      class_name = dataset.iloc[i,3]

      if curr_name == prev_name:

         #update pos info
         update_pos_info(pos_info,dataset.iloc[i,8].item(),dataset.iloc[i,9].item(),dataset.iloc[i,4].item(),dataset.iloc[i,5].item(),dataset.iloc[i,6].item(),dataset.iloc[i,7].item(),class_map[dataset.iloc[i,3]],anchors,image_shape=input_shape,bbox_type=bbox_type,feature_size=85)
      
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
         update_pos_info(pos_info,dataset.iloc[i,8].item(),dataset.iloc[i,9].item(),dataset.iloc[i,4].item(),dataset.iloc[i,5].item(),dataset.iloc[i,6].item(),dataset.iloc[i,7].item(),class_map[dataset.iloc[i,3]],anchors,image_shape=input_shape,bbox_type=bbox_type,feature_size=85)

      elif prev_name == "#":

         #crreate new pos info
         pos_info = [np.zeros(pos_info_format[0]),np.zeros(pos_info_format[1]),np.zeros(pos_info_format[2])]

         #update pos info
         update_pos_info(pos_info,dataset.iloc[i,8].item(),dataset.iloc[i,9].item(),dataset.iloc[i,4].item(),dataset.iloc[i,5].item(),dataset.iloc[i,6].item(),dataset.iloc[i,7].item(),class_map[dataset.iloc[i,3]],anchors,image_shape=input_shape,bbox_type=bbox_type,feature_size=85)

      #update prev_name
      prev_name = curr_name

   #save last obj
   for q in range(pos_format_size):
   
      h_size = pos_info_format[q][0]
      w_size = pos_info_format[q][1]
      
      with open(f"{save_path}/{curr_name}_{h_size}x{w_size}.txt","w") as file:

         file.write(json.dumps((pos_info[q]).tolist()))

         file.close()


#@jit(nopython=True)
def update_pos_info(pos_info,center_x,center_y,xmin,ymin,xmax,ymax,class_index,anchors,image_shape = (640,640),bbox_type = 3,feature_size = 85):

   """
   center_info -- (x,y)
   pos_info -- list containing numpy.ndarray
   obj_pos -- numpy.ndarray (1,4)
   anchors -- numpy.ndarray (K,2)
   image_shape -- (x,y)
   """
   
   #object bbox h w
   bbox_h = xmax - xmin
   bbox_w = ymax - ymin
   
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
   #feature mapping
   feature_per_image_pixel_x = pos_info[best_anchor_index].shape[0] / image_shape[0]
   feature_per_image_pixel_y = pos_info[best_anchor_index].shape[1] / image_shape[1]

   target_x = np.int64(center_x * feature_per_image_pixel_x).item()
   target_y = np.int64(center_y * feature_per_image_pixel_y).item()

   #prob
   (pos_info[best_anchor_index])[target_y,target_x,0] = 1

   #xmin,ymin,xmax,ymax
   (pos_info[best_anchor_index])[target_y,target_x,1] = xmin
   (pos_info[best_anchor_index])[target_y,target_x,2] = ymin
   (pos_info[best_anchor_index])[target_y,target_x,3] = xmax
   (pos_info[best_anchor_index])[target_y,target_x,4] = ymax

   #class
   (pos_info[best_anchor_index])[target_y,target_x,(class_index+5)] = 1

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
   
