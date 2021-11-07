import tensorflow as tf
import cv2
import os
from alpha_simplebasic_CSP import alpha_model
import preprocess_data
import draw
from numba import jit
import numpy as np
import json

@tf.function
def step_predict(img,model):

   predictions = model(img,train_flag=False,training=False)

   return predictions

def load_model(model_path):

   #define model
   #model = tf.keras.models.load_model(model_path)
   model = alpha_model()
   #model.load_weights(f"{cur_path}/gdrive/MyDrive/model_weights")
   model.load_weights(model_path)

   return model

def get_image_generator(batch_size,image_path,standard_shape=(640,640)):

   filename_list = os.listdir(image_path)

   idx = 0

   while True:

      img_data = []

      batch_filename_list = filename_list[ idx * batch_size : (idx+1) * batch_size ]

      if len(batch_filename_list) == 0:

         break

      for name in batch_filename_list:
         
         #read image
         img = cv2.imread(f"{image_path}/{name}").astype(np.float64)

         #pad image
         img = preprocess_data.preprocess_image(img,standard_shape)

         #save img
         img_data.append(img)

      img_data = np.array(img_data)

      yield img_data

      idx = idx + 1

def draw_box_based_on_feat(img,selected_object,reversed_class_map,class_color_map):


   for obj in selected_object:

      class_idx = np.argmax(obj[1][5:])

      #set feat
      feat = [class_idx,obj[1][1],obj[1][2],obj[1][3],obj[1][4],obj[0],obj[-1]]

      #draw box
      img = draw.draw_anchor_box(img,feat,reversed_class_map,class_color_map)


   return img


def analyse_feature(feat,confidence_threshold=0.8,diou_threshold=0.4):

   """
   feat -- numpy array
   """
   
   #filtering high prob anchor box - confirmed_anchor_box -- list: [ (prob1,feat_vec1) , (prob2,feat_vec2) , ... ]
   confirmed_anchor_box = get_confirmed_anchor_box(feat,confidence_threshold)

   #sort in descending order according to prob * class
   confirmed_anchor_box.sort(key=lambda x:x[0],reverse=True)
   
   #non-max suppression
   selected_object = non_max_supression(confirmed_anchor_box,diou_threshold)

   return selected_object
      

def get_confirmed_anchor_box(feat,confidence_threshold=0.8):

   nH,nW,nC = feat.shape

   confirmed_anchor_box = []
   
   for h in range(nH):

      for w in range(nW):

         #feat_vec : (c,)
         feat_vec = feat[h,w,:].copy()
         
         class_idx = np.argmax(feat_vec[5:])
         
         prob = feat_vec[0]#feat_vec[5+class_idx] * feat_vec[0]

         if prob >= confidence_threshold:

            #confirmed_anchor_box.append((prob,feat_vec))
            confirmed_anchor_box.append((prob,feat_vec,feat_vec[0]))

   return confirmed_anchor_box


def non_max_supression(confirmed_anchor_box,diou_threshold=0.4):

   """
   confirmed_anchor_box -- list: [ (prob1,feat_vec1) , (prob2,feat_vec2) , ... ]
   """

   selected_object = []
   m = len(confirmed_anchor_box)

   while m != 0:

      #get and save largest fect_vec -- feat_vec: numpy.array (c,)
      feat_vec = confirmed_anchor_box[0][1]
      
      #selected_object.append(feat_vec)
      selected_object.append((confirmed_anchor_box[0][0],feat_vec,confirmed_anchor_box[0][-1]))

      #pop it from confirmed_anchor_box
      confirmed_anchor_box.pop(0)

      #get confirmed_anchor_box length
      m = len(confirmed_anchor_box)

      #compare it with diou
      idx = 0
      
      for i in range(m):

         #get diou
         diou_val = DIOU(feat_vec,confirmed_anchor_box[idx][1])

         if diou_val > diou_threshold:

            confirmed_anchor_box.pop(idx)

         else:

            idx = idx + 1
      
      m = len(confirmed_anchor_box)

   return selected_object

@jit(nopython=True)
def DIOU(feat_1,feat_2):

   #get left pos
   left_1 = feat_1[1:3]
   left_2 = feat_2[1:3]

   #get center
   center_1 = feat_1[3:5]
   center_2 = feat_2[3:5]

   #get width , height
   wh_1 = (center_1[:] - left_1[:])*2
   wh_2 = (center_2[:] - left_2[:])*2

   #get right pos
   right_1 = left_1[:] + wh_1[:]
   right_2 = left_2[:] + wh_2[:]

   ################## IOU ##################
   left_intersection = np.maximum(left_1,left_2)
   
   right_intersection = np.minimum(right_1,right_2)
   right_intersection = np.maximum(left_intersection,right_intersection)

   wh_intersection = right_intersection[:] - left_intersection[:]

   intersection_area = wh_intersection[0] * wh_intersection[1]

   union_area = wh_1[0] * wh_1[1] + wh_2[0] * wh_2[1] - intersection_area
   
   iou_val  = intersection_area / (union_area + 1e-10)

   ################## IOU ##################

   ################## distance ratio ##################
   outermost_left = np.minimum(left_1,left_2)
   outermost_right = np.maximum(right_1,right_2)

   outermost_distance = np.sum(np.square(outermost_right[:] - outermost_left[:]))

   center_distance = np.sum(np.square(center_1[:] - center_2[:]))

   distance_ratio = center_distance/(outermost_distance+1e-10)

   ################## distance ratio ##################

   diou_val = iou_val - distance_ratio

   return diou_val

def process_frame(model,img,reversed_class_map,class_color_map,confidence_threshold=0.8,diou_threshold=0.4,standard_shape=(640,640)):

   
   y = step_predict(img.copy().astype(np.float64)[np.newaxis,:,:,:],model)

   #large object
   large_obj = y[0][0].numpy()
   large_confirmed_anchor_box = get_confirmed_anchor_box(large_obj,confidence_threshold)
   
   #medium object
   medium_obj = y[1][0].numpy()
   medium_confirmed_anchor_box = get_confirmed_anchor_box(medium_obj,confidence_threshold)
   
   #small object
   small_obj = y[2][0].numpy()
   small_confirmed_anchor_box = get_confirmed_anchor_box(small_obj,confidence_threshold)

   #final anchor box -- final_anchor_box -- list: [ (prob1,feat_vec1) , (prob2,feat_vec2) , ... ]
   final_anchor_box = large_confirmed_anchor_box + medium_confirmed_anchor_box + small_confirmed_anchor_box

   #sort in descending order
   final_anchor_box.sort(key=lambda x:x[0],reverse=True)

   #nms
   selected_object = non_max_supression(final_anchor_box,diou_threshold)
   
   #draw prediction
   img = draw_box_based_on_feat(img,selected_object,reversed_class_map,class_color_map)

   return img

if __name__ == "__main__":

   path = os.getcwd()

   one_device = tf.distribute.OneDeviceStrategy(device="GPU:0")

   model_path = f"{path}/base_model_weights"
   model = load_model(model_path)


   data_path =  f"{path}/data"
   class_info = preprocess_data.preprocess_class(f"{path}/annotations/train_annotations.csv",data_path)
   
   class_color_map = preprocess_data.preprocess_class_color_map(class_info,data_path)
   reversed_class_info = preprocess_data.reverse_class_info(class_info,data_path)
   
   
   cap = cv2.VideoCapture(f"{os.getcwd()}/video/res_1.mp4")

   cap.set(cv2.CAP_PROP_FRAME_WIDTH,640)
   cap.set(cv2.CAP_PROP_FRAME_HEIGHT,640)

   fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
   videoWriter = cv2.VideoWriter(f"{os.getcwd()}/video/res_video_1.mp4",fourcc ,20.0,(640,640))

   if (cap.isOpened()== False):

     print("Error opening video stream or file")

   i = 1

   while cap.isOpened():

       flag,frame = cap.read()

       if flag == False:

          break

       frame = cv2.resize(frame, (640, 360), interpolation=cv2.INTER_AREA)
       frame = preprocess_data.preprocess_image(frame,(640,640))

       #frame = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)

       with one_device.scope():
          
          frame = process_frame(model,frame,reversed_class_info,class_color_map,confidence_threshold=0.6,diou_threshold=0.5)

       #frame = cv2.cvtColor(frame,cv2.COLOR_RGB2BGR)
      
       #cv2.imshow("frame",frame)
       #cv2.imwrite(f"{os.getcwd()}/videoTest/res_{i}.jpg",frame)
       videoWriter.write(frame)
       print(i)
       i = i + 1

         
   cap.release()
   videoWriter.release()
   cv2.destroyAllWindows()
