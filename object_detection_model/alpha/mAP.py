import tensorflow as tf
import numpy as np
import os
from  numba import jit
from generate_data import get_gt_data
import json
from predict import step_predict,load_model
import preprocess_data
import matplotlib.pyplot as plt

def mAP(model_path,image_path,img_info_path,class_info_path,batch_size=4,confidence_threshold=0.7,iou_threshold=0.5):

   mAP_output = []

   #get img_info (i.e. f"{cur_path}/data/gt_dataset.txt" )
   file = open(img_info_path)
   img_info = json.load(file)
   file.close()

   #get class_info (i.e. f"{cur_path}/data/class_map.txt")
   file = open(class_info_path)
   class_info = json.load(file)
   file.close()

   #set up gpu
   one_device_strategy = tf.distribute.OneDeviceStrategy(device="GPU:0")

   with one_device_strategy.scope():

      model = load_model(model_path)

   
   #calculate mAP
   for valid_images,y_true in get_gt_data(batch_size,img_info,class_info,image_path,mul_pos=False):

      y_pred = step_predict(valid_images.astype(np.float64),model)

      for i in range(batch_size):

         #large object
         large_obj = y_pred[0][i].numpy()
         precision_recall = get_precision_recall(y_true[0][i],large_obj,confidence_threshold,iou_threshold)
            
         mAP_output.append(precision_recall)

         #medium obj
         medium_obj = y_pred[1][i].numpy()
         precision_recall = get_precision_recall(y_true[1][i],medium_obj,confidence_threshold,iou_threshold)
            
         mAP_output.append(precision_recall)

         #small_obj
         small_obj = y_pred[2][i].numpy()
         precision_recall = get_precision_recall(y_true[2][i],small_obj,confidence_threshold,iou_threshold)

            
         mAP_output.append(precision_recall)

   #ascending order by recall
   mAP_output.sort(key=lambda x : x[1])

   #convert to array
   mAP_output = np.array(mAP_output)

   #save plot
   plt.scatter(mAP_output[:,1],mAP_output[:,0])
   plt.ylabel("precision")
   plt.xlabel("recall")
   plt.savefig(f"{os.getcwd()}/mAP_plot.png")

   #Area under curve by trapz
   mAP_output = AUC_for_mAP(mAP_output)#np.trapz(y=mAP_output[:,0],x=mAP_output[:,1])

   return mAP_output

@jit(nopython=True)         
def get_precision_recall(y_true,y_pred,confidence_threshold=0.7,iou_threshold=0.5):

   """
   y_true -- (h,w,c)
   y_pred -- (h,w,c)
   """

   #obj_pos_true -- (h,w)
   obj_pos_true = y_true[:,:,0]

   #obj_pos_pred -- (h,w)
   obj_pos_pred = y_pred[:,:,0]
   
   #set up
   feat_IOU = IOU_mAP(y_true,y_pred)

   #$#$#$#$#$#$#$#$#$#$$#$# Get TP #$#$#$#$#$#$#$#$#$#$$#$#

   TP_IOU =  ( feat_IOU > iou_threshold ) * obj_pos_true

   TP = np.sum(TP_IOU)
   
   #$#$#$#$#$#$#$#$#$#$$#$# Get TP #$#$#$#$#$#$#$#$#$#$$#$#

   pred_think = (obj_pos_pred > confidence_threshold)
   pred_think = np.sum(pred_think)

   if pred_think < TP:

      precision = 0

   else:

      precision = TP / (pred_think + 1e-7)
      
   
   recall = TP / (np.sum(obj_pos_true) + 1e-7 )

   #print(precision,recall)
   
   return [precision,recall]
   

@jit(nopython=True)
def IOU_mAP(feat_1,feat_2):

   #feat: (h,w,c)

   #get left pos (h,w,2)
   left_1 = feat_1[:,:,1:3]
   left_2 = feat_2[:,:,1:3]

   #get center (h,w,2)
   center_1 = feat_1[:,:,3:5]
   center_2 = feat_2[:,:,3:5]

   #get width , height (h,w,2)
   wh_1 = (center_1 - left_1)*2
   wh_2 = (center_2 - left_2)*2

   #get right pos (h,w,2)
   right_1 = left_1 + wh_1
   right_2 = left_2 + wh_2

   ################## IOU ##################
   left_intersection = np.maximum(left_1,left_2)
   
   right_intersection = np.minimum(right_1,right_2)
   right_intersection = np.maximum(left_intersection,right_intersection)

   wh_intersection = right_intersection - left_intersection

   #(h,w)
   intersection_area = wh_intersection[:,:,0] * wh_intersection[:,:,1]

   #(h,w)
   union_area = wh_1[:,:,0] * wh_1[:,:,1] + wh_2[:,:,0] * wh_2[:,:,1] - intersection_area
   
   #(h,w)
   iou_val  = intersection_area / (union_area + 1e-10)

   ################## IOU ##################

   return iou_val


"""
#@jit(nopython=True)
def IOU_mAP(y_true,y_pred):

   #y_true: (h,w,c)
   #y_pred: (h,w,c)

   h,w,c = y_true.shape

   if np.sum(y_true[:,:,0]) == 0:

      return np.zeros((h,w))

   #(n,c)
   obj_true = y_true[y_true[:,:,0]==1]
   #(1,n,c)
   obj_true = obj_true[np.newaxis,:,:]

   #(h,w,1,c)
   obj_pred = (y_pred.copy())[:,:,np.newaxis,:]

   #left right pred (h,w,1,2)
   left_pred = obj_pred[:,:,:,1:3]
   wh_pred = (obj_pred[:,:,:,3:5] - left_pred) * 2
   right_pred = wh_pred + left_pred

   #left right true (1,n,2)
   left_true = obj_true[:,:,1:3]
   wh_true = (obj_true[:,:,3:5] - left_true) * 2
   right_true = wh_true + left_true

   #intersection xy (h,w,n,2)
   inter_left = np.maximum(left_pred,left_true)
   
   inter_right = np.minimum(right_pred,right_true)
   inter_right = np.maximum(inter_right,inter_left)

   #inter wh (h,w,n,2)
   inter_wh = inter_right - inter_left

   #inter area (h,w,n)
   inter_area = inter_wh[:,:,:,0] * inter_wh[:,:,:,1]

   #union (h,w,n)
   union_area = wh_pred[:,:,:,0] * wh_pred[:,:,:,1] + wh_true[:,:,0] * wh_true[:,:,1] - inter_area

   #iou (h,w,n)
   iou_val = inter_area / ( union_area + 1e-8 )

   #max iou (h,w)
   iou_val = np.max(iou_val,axis=-1,keepdims=False)

   return iou_val
"""

@jit(nopython=True)
def AUC_for_mAP(pr_val):

   m,_ = pr_val.shape

   area = 0
   
   for i in range(m-1):

      p1 = pr_val[i,0]
      p2 = pr_val[i+1,0]

      curr_p = max(p1,p2)

      curr_step = pr_val[i+1,1] - pr_val[i,1]

      if curr_step == 0:

         curr_step = 1/m

      area = area + curr_p * curr_step

   return area

if __name__ == "__main__":

   path = os.getcwd()
   data_path =  f"{path}/data"

   class_info = preprocess_data.preprocess_class(f"{path}/annotations/train_annotations.csv",data_path)
   class_info_path = f"{data_path}/class_map.txt"
   
   gt_dataset = preprocess_data.preprocessing_label(f"{path}/annotations/valid_annotations.csv",data_path)
   gt_path = f"{data_path}/gt_dataset.txt"

   #model_path = f"{path}/base_model_weights"
   model_path = f"{path}/aux_model_weights"
   image_path = f"{path}/pending_to_analysis"

   mAP_val = mAP(model_path,image_path,gt_path,class_info_path,batch_size=4,confidence_threshold=0.5,iou_threshold=0.5)
   
