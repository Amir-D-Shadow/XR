import pandas as pd
import numpy as np
import json
import os
from numba import jit
import cv2
import random
import preprocess_data
import time

#generator
def get_gt_data(batch_size,img_info,class_info,img_path,img_shape = (640,640),standard_scale=(19360,66930)):

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
      img_data = get_image_data(name_list,img_path,img_shape)

      #get y_true data -- tuple (np.array,np.array,np.array)
      label = get_y_true(name_list,img_info,class_info,img_shape,standard_scale)

      #update remaining sample
      m = m - batch_size
      idx = idx + 1

      yield img_data,label

"""
def get_gt_data(batch_size,img_list_shuffled,img_info,class_info,img_path,img_shape = (640,640),standard_scale=(8000,50000)):
   
   random.shuffle(img_list_shuffled)
      
   #get name list
   name_list = []

   for i in range(batch_size):

      name_list.append(img_list_shuffled[i])

      img_list_shuffled.remove(img_list_shuffled[i])


   #get image data -- np.array
   img_data = get_image_data(name_list,img_path,img_shape)

   #get y_true data -- tuple (np.array,np.array,np.array)
   label = get_y_true(name_list,img_info,class_info)

   return img_data,label
"""
      
def get_image_data(name_list,img_path,img_shape=(640,640)):

   """
   return numpy.ndarray
   """

   img_data = []

   for name in name_list:

      #img -- numpy.ndarray
      img = cv2.imread(f"{img_path}/{name}")

      #calibrate image
      img = preprocess_data.preprocess_image(img,img_shape)

      #save img
      img_data.append(img)

   img_data = np.array(img_data)

   return img_data


def get_y_true(name_list,img_info,class_info,img_shape = (640,640),standard_scale=(19360,66930)):

   """
   name_list -- list
   img_info -- dict -- {obj1:[[class,xmin,ymin,xcenter,ycenter],[class,xmin,ymin,xcenter,ycenter],...],obj2:...} (for each key)
   class_info -- dict
   standard_scale -- dict (small , medium , large)
   img_shape -- (height,width)
   """
   #initialize y_true
   small_true = []
   medium_true = []
   large_true = []
   
   
   for name in name_list:


      #initialize y_true extra dim will be removed when it is saved (it is used for overlap region checking)
      obj_small_true = np.zeros((80,80,86))
      obj_medium_true = np.zeros((40,40,86))
      obj_large_true = np.zeros((20,20,86))

      #get (obj_info -- list)
      obj_info = img_info[name]

      #loop via all object in the image (obj -- list)
      for obj in obj_info:

         #update y_true
         obj_small_true,obj_medium_true,obj_large_true = update_y_true(obj,class_info[obj[0]],obj_small_true,obj_medium_true,obj_large_true,img_shape,standard_scale)
         
      #save image info
      small_true.append(obj_small_true[:,:,:-1])
      medium_true.append(obj_medium_true[:,:,:-1])
      large_true.append(obj_large_true[:,:,:-1])

   #convert y_true to numpy array
   small_true = np.array(small_true)
   medium_true = np.array(medium_true)
   large_true = np.array(large_true)

   return (small_true,medium_true,large_true)

   
def update_y_true(obj,class_id,obj_small_true,obj_medium_true,obj_large_true,img_shape = (640,640),standard_scale=(19360,66930)):

   """
   obj -- list [class,xmin,ymin,xcenter,ycenter]
   img_shape -- (height,width)
   """

   _,xmin,ymin,xcenter,ycenter = obj

   width = (xcenter - xmin) * 2
   height = (ycenter - ymin) * 2

   xmax = xmin + width
   ymax = ymin + height

   area  = width * height

   if area < standard_scale[0]:

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

      #label occupied cells
      obj_small_true[h_pos,w_pos,-1] == 1

      #multiple positive
      obj_small_true = multiple_positive_labeling(obj_small_true,class_id,xmin,ymin,xmax,ymax,xcenter,ycenter,step_w,step_h)

   elif (area > standard_scale[0]) and (area < standard_scale[1]):

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

      #label occupied cells
      obj_medium_true[h_pos,w_pos,-1] == 1

      #multiple positive
      obj_medium_true = multiple_positive_labeling(obj_medium_true,class_id,xmin,ymin,xmax,ymax,xcenter,ycenter,step_w,step_h)

   elif (area > standard_scale[1]) :

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

      #label occupied cells
      obj_large_true[h_pos,w_pos,-1] == 1

      #multiple positive
      obj_large_true = multiple_positive_labeling(obj_large_true,class_id,xmin,ymin,xmax,ymax,xcenter,ycenter,step_w,step_h)

   return obj_small_true,obj_medium_true,obj_large_true

            
@jit(nopython=True)  
def multiple_positive_labeling(y_true,class_id,xmin,ymin,xmax,ymax,xcenter,ycenter,step_w,step_h):

   """
   y_true -- numpy array
   """

   w_pos_init = int(xmin*step_w)
   h_pos_init = int(ymin*step_h)

   w_max = int(xmax*step_w)
   h_max = int(ymax*step_h)

   for w_pos in range(w_pos_init,w_max):

      for h_pos in range(h_pos_init,h_max):

         if y_true[h_pos,w_pos,-1] == 0:

            #prob
            y_true[h_pos,w_pos,0] = 1

            #xmin,ymin
            y_true[h_pos,w_pos,1] = xmin
            y_true[h_pos,w_pos,2] = ymin

            #xcenter,ycenter
            y_true[h_pos,w_pos,3] = xcenter
            y_true[h_pos,w_pos,4] = ycenter

            #class
            y_true[h_pos,w_pos,5+class_id] = 1

            #label occupied cells
            y_true[h_pos,w_pos,-1] == 1


   return y_true

if __name__ == "__main__":

   cur_path = os.getcwd()

   file = open(f"{cur_path}/data/gt_dataset.txt")
   gt_dataset = json.load(file)

   file.close()

   file = open(f"{cur_path}/data/class_map.txt")
   class_info = json.load(file)

   file.close()
   """
   name_list = ["000000001000_jpg.rf.a3c5a2484544de19f7cb041f2eb43605.jpg"]

   label = get_y_true(name_list,gt_dataset,class_info)
   """

   i = 0
   ctime = time.time()
   for img_data,label in get_gt_data(64,gt_dataset,class_info,f"{os.getcwd()}/img"):

      print(img_data.shape)

      i = i + 1

      if i > 10:

         break

   print(time.time()-ctime)
   
   """
   img_list = list(gt_dataset.keys())
   #random.shuffle(img_list)
   
   ctime = time.time()
   
   for i in range(8):

      img_data,label =  get_gt_data(3,img_list,gt_dataset,class_info,f"{os.getcwd()}/img")

      print(img_data.shape)

   print(time.time()-ctime)
   """
   with open(f"{os.getcwd()}/data/label_small.txt","w") as file:

      file.write(json.dumps((label[0]).tolist()))

      file.close()

   with open(f"{os.getcwd()}/data/label_medium.txt","w") as file:

      file.write(json.dumps((label[1]).tolist()))

      file.close()

   with open(f"{os.getcwd()}/data/label_large.txt","w") as file:

      file.write(json.dumps((label[2]).tolist()))

      file.close()      
