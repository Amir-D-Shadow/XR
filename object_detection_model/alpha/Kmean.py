import numpy as np
from numba import jit
import random


def Kmean_IOU(bbox_hw,K=3,threshold=1e-8,max_iterations=2000):

   """
   bbox_hw -- (m,2) : h -> 0 , w -> 1

   return -- [large,medium ,small] <-- list(list)
   """

   m = bbox_hw.shape[0]

   #ONLY for K = 3 
   if m == 1:

      return [ [0] , [] , [] ]

   elif m == 2:

      bbox_1_area = bbox_hw[0,0] * bbox_hw[0,1]
      bbox_2_area = bbox_hw[1,0] * bbox_hw[1,1]

      if bbox_1_area > bbox_2_area:
         
         return [ [0] , [1] , [] ]

      else:

         return [ [1] , [0] , [] ]

   #initialize anchors (K,2)
   anchors = np.zeros((K,2))
   for i in range(K):

      anchors[i,:] = bbox_hw[i,:].copy()
   
   #calculate Kmean
   iteration_i = 0
   
   while iteration_i <= max_iterations:

      #store the data by index
      output_list = [[] for i in range(K)]

      #classify bbox 
      for i in range(m):

         class_id = best_anchor(bbox_hw[i,:].reshape(1,2),anchors)

         output_list[class_id].append(i)

      #update anchor box
      sum_diff = 0
      
      for i in range(K):
         
         new_h,new_w = update_anchor_x(bbox_hw,output_list[i],anchors[i,:].reshape(1,2))

         """
         #sum the changes for h and w of new anchor box 
         sum_diff = sum_diff + ((new_h-anchors[i,0])**2 + (new_w - anchors[i,1])**2)**(0.5)
         """
         #sum the changes for h and w of new anchor box - compare iou
         min_h = min(new_h,anchors[i,0])
         min_w = min(new_w,anchors[i,1])

         intersection = min_h * min_w
         union = new_h * new_w + anchors[i,0] * anchors[i,1] - intersection

         sum_diff = sum_diff + (1 - intersection / (union + 1e-8) )

         #update anchor box
         anchors[i,0] = new_h
         anchors[i,1] = new_w

      if sum_diff < threshold:

         return rearrange_output_list_to_correct_order(bbox_hw,output_list)

      #update iteration
      iteration_i = iteration_i + 1
      
   return rearrange_output_list_to_correct_order(bbox_hw,output_list)
         

def rearrange_output_list_to_correct_order(bbox_hw,output_list):

   #corrected output_list
   corrected_output_list = []
   
   n =len(output_list)
   #find area sum
   sample_area_sum = []
   for i in range(n):

      area = find_area_sum(bbox_hw,output_list[i])
      sample_area_sum.append((i,area))

   #sort the list
   sample_area_sum.sort(key=lambda x:x[1],reverse=True)

   #correct output list
   for ele in sample_area_sum:

      corrected_output_list.append(output_list[ele[0]])

   return corrected_output_list
   

def find_area_sum(bbox_hw,output_list_x):

   area = 0

   for i in output_list_x:

      area = area + bbox_hw[i,0] * bbox_hw[i,1]

   return area
      
   
def update_anchor_x(bbox_hw,output_list_x,anchors_x):

   """
   output_list_x -- [ n elements]
   anchor_x -- (1,2) : h -> 0 , w -> 1
   """

   sum_w = 0
   sum_h = 0

   n = len(output_list_x)

   for i in output_list_x:

      sum_h = sum_h + bbox_hw[i,0]
      sum_w = sum_w + bbox_hw[i,1]

   if n == 0:

      return anchors_x[0,0],anchors_x[0,1]

   mean_h = sum_h/n
   mean_w = sum_w/n

   return mean_h,mean_w


@jit(nopython=True)
def best_anchor(box,anchors):
   
  """
  box -- (1,2) :h -> 0 , w -> 1

  anchors -- (K,2)

  return class_idx
  """
  
  max_iou = 0
  max_index = 0

  K = anchors.shape[0]

  for i in range(K):

    min_h = np.minimum(box[0,0],anchors[i,0])
    min_w = np.minimum(box[0,1],anchors[i,1])

    intersection_area = min_h * min_w

    union_area = box[0,0] * box[0,1] + anchors[i,0] * anchors[i,1] - intersection_area

    cur_iou = intersection_area / (union_area + 1e-8)

    if cur_iou > max_iou:

      max_iou = cur_iou

      max_index = i

    
  return max_index
