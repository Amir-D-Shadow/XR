import pandas as pd
import numpy as np
import json
import os
from numba import jit
import cv2
import random
import preprocess_data
import time
import data_augment
from Kmean import Kmean_IOU

#generator
def get_gt_data(batch_size,img_info,class_info,img_path,img_shape = (640,640),aug_flag=False):

   """
   #img_shape -- (height,width)
   """
   
   img_list_shuffled = list(img_info.keys())
   
   random.shuffle(img_list_shuffled)

   m = len(img_list_shuffled)

   idx = 0

   while m >= batch_size:

      """
      #check remaining sample
      #if m < batch_size:

         #break
      """
      
      #get name list
      name_list = []

      for i in range(idx*batch_size,(idx+1)*batch_size):

         name_list.append(img_list_shuffled[i])


      #get image data -- np.array
      img_data = get_image_data(name_list,img_path,img_shape,aug_flag)

      #get y_true data -- tuple (np.array,np.array,np.array)
      label = get_y_true(name_list,img_info,class_info,img_shape)

      #update remaining sample
      m = m - batch_size
      idx = idx + 1

      yield img_data,label


      
def get_image_data(name_list,img_path,img_shape=(640,640),aug_flag=False):

   """
   return numpy.ndarray
   """

   img_data = []

   for name in name_list:

      #img -- numpy.ndarray
      img = cv2.imread(f"{img_path}/{name}")

      #calibrate image
      img = preprocess_data.preprocess_image(img,img_shape)

      #img = cv2.resize(img,(128,128))
      #data augmentation
      if aug_flag:

         img = data_augment.data_aug(img)

      #save img
      img_data.append(img)

   img_data = np.array(img_data)

   return img_data


def get_y_true(name_list,img_info,class_info,img_shape = (640,640)):

   """
   name_list -- list
   img_info -- dict -- {obj1:[[class,xmin,ymin,xcenter,ycenter],[class,xmin,ymin,xcenter,ycenter],...],obj2:...} (for each key)
   class_info -- dict
   img_shape -- (height,width)
   """
   #initialize y_true
   small_true = []
   medium_true = []
   large_true = []
   
   
   for name in name_list:


      #initialize y_true extra dim will be removed when it is saved (it is used for overlap region checking)
      obj_small_true = np.zeros((80,80,85))
      obj_medium_true = np.zeros((40,40,85))
      obj_large_true = np.zeros((20,20,85))

      #obj_small_true = np.zeros((16,16,91))
      #obj_medium_true = np.zeros((8,8,91))
      #obj_large_true = np.zeros((4,4,91))

      #get (obj_info -- list)
      obj_info = img_info[name]

      n = len(obj_info)

      #initial bbox hw
      bbox_hw = np.zeros((n,2))

      #loop via all object in the image (obj -- list) to fill bbox_hw
      for i in range(n):

         bbox_hw[i,0] = (obj_info[i][4] - obj_info[i][2])*2
         bbox_hw[i,1] = (obj_info[i][3] - obj_info[i][1])*2

      #get cluster index
      cluster_idx = Kmean_IOU(bbox_hw)
      
      #update y_true
      obj_small_true,obj_medium_true,obj_large_true = update_y_true(obj_info,class_info,cluster_idx,obj_small_true,obj_medium_true,obj_large_true,img_shape = (640,640))
      
      #save image info
      small_true.append(obj_small_true[:,:,:])
      medium_true.append(obj_medium_true[:,:,:])
      large_true.append(obj_large_true[:,:,:])

   #convert y_true to numpy array
   small_true = np.array(small_true)
   medium_true = np.array(medium_true)
   large_true = np.array(large_true)

   return (large_true,medium_true,small_true)

   
def update_y_true(obj_info,class_info,cluster_idx,obj_small_true,obj_medium_true,obj_large_true,img_shape = (640,640)):

   """
   obj -- list [class,xmin,ymin,xcenter,ycenter]
   img_shape -- (height,width)
   """
   """
   _,xmin,ymin,xcenter,ycenter = obj

   #avoid x,y division by zeros for ratio calculation
   xmin = xmin + 1e-18
   ymin = ymin + 1e-18
   xcenter = xcenter + 1e-18
   ycenter = ycenter + 1e-18

   width = (xcenter - xmin) * 2
   height = (ycenter - ymin) * 2

   xmax = xmin + width
   ymax = ymin + height

   area  = width * height
   """
   #update large obj
   for i in cluster_idx[0]:

      #get obj info
      class_name,xmin,ymin,xcenter,ycenter = obj_info[i]

      class_id = class_info[class_name]

      xmax = (xcenter - xmin) * 2 + xmin
      ymax = (ycenter - ymin) * 2 + ymin

      #set up
      step_h = obj_large_true.shape[0] / img_shape[0]
      step_w = obj_large_true.shape[1] / img_shape[1]

      h_pos = int(step_h * ycenter)
      w_pos = int(step_w * xcenter)
      
      #prob
      obj_large_true[h_pos,w_pos,0] = 1

      #xmin,ymin
      obj_large_true[h_pos,w_pos,1] = xmin 
      obj_large_true[h_pos,w_pos,2] = ymin 

      #xcenter,ycenter
      obj_large_true[h_pos,w_pos,3] = xcenter 
      obj_large_true[h_pos,w_pos,4] = ycenter 

      #class
      obj_large_true[h_pos,w_pos,5+class_id] = 1

      #multiple positive
      obj_large_true = multiple_positive_labeling(obj_large_true,class_id,xmin,ymin,xmax,ymax,xcenter,ycenter,step_w,step_h)


   #update medium obj
   for i in cluster_idx[1]:

      #get obj info
      class_name,xmin,ymin,xcenter,ycenter = obj_info[i]

      class_id = class_info[class_name]

      xmax = (xcenter - xmin) * 2 + xmin
      ymax = (ycenter - ymin) * 2 + ymin

      #set up
      step_h = obj_medium_true.shape[0] / img_shape[0]
      step_w = obj_medium_true.shape[1] / img_shape[1]

      h_pos = int(step_h * ycenter)
      w_pos = int(step_w * xcenter) 
         
      #prob
      obj_medium_true[h_pos,w_pos,0] = 1

      #xmin,ymin
      obj_medium_true[h_pos,w_pos,1] = xmin 
      obj_medium_true[h_pos,w_pos,2] = ymin 

      #xcenter,ycenter
      obj_medium_true[h_pos,w_pos,3] = xcenter 
      obj_medium_true[h_pos,w_pos,4] = ycenter 

      #class
      obj_medium_true[h_pos,w_pos,5+class_id] = 1

      #multiple positive
      obj_medium_true = multiple_positive_labeling(obj_medium_true,class_id,xmin,ymin,xmax,ymax,xcenter,ycenter,step_w,step_h)


   #update small obj
   for i in cluster_idx[2]:

      #get obj info
      class_name,xmin,ymin,xcenter,ycenter = obj_info[i]

      class_id = class_info[class_name]

      xmax = (xcenter - xmin) * 2 + xmin
      ymax = (ycenter - ymin) * 2 + ymin

      #set up
      step_h = obj_small_true.shape[0] / img_shape[0]
      step_w = obj_small_true.shape[1] / img_shape[1]

      h_pos = int(step_h * ycenter)
      w_pos = int(step_w * xcenter)
      
      #prob
      obj_small_true[h_pos,w_pos,0] = 1

      #xmin,ymin
      obj_small_true[h_pos,w_pos,1] = xmin 
      obj_small_true[h_pos,w_pos,2] = ymin 

      #xcenter,ycenter
      obj_small_true[h_pos,w_pos,3] = xcenter 
      obj_small_true[h_pos,w_pos,4] = ycenter 

      #class
      obj_small_true[h_pos,w_pos,5+class_id] = 1


      #multiple positive
      obj_small_true = multiple_positive_labeling(obj_small_true,class_id,xmin,ymin,xmax,ymax,xcenter,ycenter,step_w,step_h)
                 
   return obj_small_true,obj_medium_true,obj_large_true

            
@jit(nopython=True)  
def multiple_positive_labeling(y_true,class_id,xmin,ymin,xmax,ymax,xcenter,ycenter,step_w,step_h):

   """
   y_true -- numpy array
   """

   xlow = int(xmin*step_w)
   ylow = int(ymin*step_h)

   xhigh = int(xmax*step_w)
   yhigh = int(ymax*step_h)
   
   w_pos_init = int(xcenter*step_w - 32*step_w)
   h_pos_init = int(ycenter*step_h - 32*step_h)

   w_max = int(w_pos_init + 3*32*step_w)
   h_max = int(h_pos_init + 3*32*step_h)

   n = 0

   for w_pos in range(w_pos_init,w_max):

      for h_pos in range(h_pos_init,h_max):

         if (n >= 9):

            break

         if (y_true[h_pos,w_pos,0] == 0) and (w_pos > xlow) and (w_pos < xhigh) and (h_pos > ylow) and (h_pos < yhigh):

            #prob
            y_true[h_pos,w_pos,0] = 1

            #xmin,ymin
            y_true[h_pos,w_pos,1] = xmin + y_true[h_pos,w_pos,1] 
            y_true[h_pos,w_pos,2] = ymin + y_true[h_pos,w_pos,2]

            #xcenter,ycenter
            y_true[h_pos,w_pos,3] = xcenter + y_true[h_pos,w_pos,3]
            y_true[h_pos,w_pos,4] = ycenter + y_true[h_pos,w_pos,4]

            #class
            y_true[h_pos,w_pos,5+class_id] = 1

            #update n
            n = n + 1


   return y_true


if __name__ == "__main__":

   cur_path = os.getcwd()
   """
   gt_dataset = preprocess_data.preprocessing_label(f"{cur_path}/annotations/train_annotations.csv",f"{cur_path}/data")

   class_info =  preprocess_data.preprocess_class(f"{cur_path}/annotations/train_annotations.csv",f"{cur_path}/data")
   """
   file = open(f"{cur_path}/data/class_map.txt")
   class_info = json.load(file)

   file.close()

   file = open(f"{cur_path}/data/gt_dataset.txt")
   gt_dataset = json.load(file)

   file.close()

   i = 0
   ctime = time.time()
   for img_data,label in get_gt_data(8,gt_dataset,class_info,f"{os.getcwd()}/images",aug_flag=True):

      print(img_data.shape)

      i = i + 1

      if i > 5:

         break

   print(time.time()-ctime)
   

   with open(f"{os.getcwd()}/data/label_small.txt","w") as file:

      file.write(json.dumps((label[2]).tolist()))

      file.close()

   with open(f"{os.getcwd()}/data/label_medium.txt","w") as file:

      file.write(json.dumps((label[1]).tolist()))

      file.close()

   with open(f"{os.getcwd()}/data/label_large.txt","w") as file:

      file.write(json.dumps((label[0]).tolist()))

      file.close()      

