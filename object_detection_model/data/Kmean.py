import numpy as np
from numba import jit
import random


def Kmean(bbox_hw,K=9,threshold=0.01,max_iterations=10000):

   """
   bbox_hw -- (m,2) : h -> 0 , w -> 1
   """

   m = bbox_hw.shape[0]

   #sample initial center
   sample_list = random.sample(range(m),K)

   #initialize anchors (K,2)
   anchors = np.zeros((K,2))
   
   for idx,pos in enumerate(sample_list):

     anchors[idx,0] = bbox_hw[pos,0]
     anchors[idx,1] = bbox_hw[pos,1]
   
   #calculate Kmean
   iteration_i = 0
   
   while iteration_i <= max_iterations:

      class_collections = [[] for i in range(K)]

      #classify bbox 
      for i in range(m):

         class_id = best_anchor(bbox_hw[i,:].reshape(1,2),anchors)

         class_collections[class_id].append(bbox_hw[i,:].tolist())

      #update anchor box
      sum_diff = 0
      
      for i in range(K):
         
         new_h,new_w = update_anchor_x(np.array(class_collections[i]),anchors[i,:].reshape(1,2))

         """
         #sum the changes for h and w of new anchor box 
         sum_diff = sum_diff + ((new_h-anchors[i,0])**2 + (new_w - anchors[i,1])**2)**(0.5)
         """
         #sum the changes for h and w of new anchor box - compare iou
         min_h = min(new_h,anchors[i,0])
         min_w = min(new_w,anchors[i,1])

         intersection = min_h * min_w
         union = new_h * new_w + anchors[i,0] * anchors[i,1] - intersection

         sum_diff = sum_diff + (1 - intersection / union)

         #update anchor box
         anchors[i,0] = new_h
         anchors[i,1] = new_w

      if sum_diff < threshold:

         return anchors

   print(sum_diff)
   
   return anchors
         

@jit(nopython=True)
def update_anchor_x(class_package,anchors_box):

  """
  class_package -- (n,2) : h -> 0 , w -> 1
  anchor_box -- (1,2) : h -> 0 , w -> 1
  """

  sum_w = 0
  sum_h = 0

  n = class_package.shape[0]

  for i in range(n):

    sum_h = sum_h + class_package[i,0]
    sum_w = sum_w + class_package[i,1]

  if n == 0:

    return anchors_box[0,0],anchors_box[0,1]

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

    min_h = np.minimum(box[0,0],anchors[i,0]).item()
    min_w = np.minimum(box[0,1],anchors[i,1]).item()

    intersection_area = min_h * min_w

    union_area = box[0,0] * box[0,1] + anchors[i,0] * anchors[i,1] - intersection_area

    cur_iou = intersection_area / union_area 

    if cur_iou > max_iou:

      max_iou = cur_iou

      max_index = i

    
  return max_index
