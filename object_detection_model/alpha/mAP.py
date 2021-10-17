import numpy as np
from scipy.integrate import simpson
from  numba import jit
from generate_data import get_gt_data
import json
from predict import step_predict,load_model

def mAP(inputfile_path,model_path,image_path,img_info_path,class_info_path,batch_size = 4,confidence_threshold=0.7,iou_threshold=0.5):

   mAP_output = []

   #get img_info (i.e. f"{cur_path}/data/gt_dataset.txt" )
   file = open(img_info_path)
   img_info = json.load(file)
   file.close()

   #get class_info (i.e. f"{cur_path}/data/class_map.txt")
   file = open(img_info_path)
   class_info = json.load(file)
   file.close()

   #set up gpu
   one_device_strategy = tf.distribute.OneDeviceStrategy(device="GPU:0")

   with one_device_strategy.scope():

      model = load_model(model_path)

   
   #calculate mAP
   for valid_images,y_true in get_gt_data(batch_size,img_info,class_info,image_path):

      y_pred = step_predict(valid_images,model)

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

   #Area under curve by simpson
   mAP_output = simpson(y=mAP_output[:,0],x=mAP_output[:,1])/20

   return mAP_output
         
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

   #$#$#$#$#$#$#$#$#$#$$#$# Get FP #$#$#$#$#$#$#$#$#$#$$#$#
   
   FP_IOU =  ( feat_IOU <= iou_threshold )  * obj_pos_true

   FP = np.sum(FP_IOU)
   
   #$#$#$#$#$#$#$#$#$#$$#$# Get FP #$#$#$#$#$#$#$#$#$#$$#$#

   #$#$#$#$#$#$#$#$#$#$$#$# Get FN #$#$#$#$#$#$#$#$#$#$$#$#
   FN_IOU = ( feat_IOU <= iou_threshold )  * (obj_pos_pred > confidence_threshold)

   FN = np.sum(FN_IOU)

   #$#$#$#$#$#$#$#$#$#$$#$# Get FN #$#$#$#$#$#$#$#$#$#$$#$#

   precision = TP / (TP + FP + 1e-7)
   recall = TP / (TP + FN + 1e-7)
   
   return [precision,recall]
   

@jit(nopython=True)
def IOU_mAP(feat_1,feat_2):

   """
   feat: (h,w,c)
   """

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
