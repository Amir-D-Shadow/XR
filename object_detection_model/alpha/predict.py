import tensorflow as tf
import tensorflow.keras.backend as K
import preprocess_data
import draw
from numba import jit
import numpy as np
import json
import cv2
import os


@tf.function
def step_predict(img,model):

   predictions = model(img,training=False)

   return predictions

def load_model(model_path):

   #define model
   model = tf.keras.models.load_model(model_path)

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
         img = cv2.imread(f"{image_path}/{name}")

         #normalize image
         img = img / np.float64(255)

         #pad image
         img = preprocess_data.preprocess_image(img,standard_shape)

         #save img
         img_data.append(img)

      img_data = np.array(img_data)

      yield img_data

      idx = idx + 1

def predict(model,image_path,result_path,reversed_class_map,class_color_map,confidence_threshold=0.8,diou_threshold=0.4,batch_size=8,standard_shape=(640,640)):

   #get batch of image
   img_data = next(get_image_generator(batch_size,image_path,standard_shape))

   #get prediction
   y = step_predict(img_data,model)

   #analysis and draw data
   for i in range(batch_size):

      img = img_data[i].copy()

      #large object
      large_obj = y[0][i].numpy()
      selected_object = analyse_feature(large_obj,confidence_threshold,diou_threshold)
      img = draw_box_based_on_feat(img,seleted_object,reversed_class_map,class_color_map)
      
      #medium object
      medium_obj = y[1][i].numpy()
      selected_object = analyse_feature(medium_obj,confidence_threshold,diou_threshold)
      img = draw_box_based_on_feat(img,seleted_object,reversed_class_map,class_color_map)
      
      #small object
      small_obj = y[2][i].numpy()
      selected_object = analyse_feature(small_obj,confidence_threshold,diou_threshold)
      img = draw_box_based_on_feat(img,seleted_object,reversed_class_map,class_color_map)

      #save the result
      cv2.imwrite(f"{result_path}/res_{i}.jpg",img)

   """
   print(type(y))

   print(type(y[0]))
   """
   
   return y

def draw_box_based_on_feat(img,seleted_object,reversed_class_map,class_color_map):


   for obj in selected_object:

      class_idx = np.argmax(obj[5:])

      #set feat
      feat = [class_idx,obj[1],obj[2],obj[3],obj[4]]

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

         feat_vec = feat[h,w,:].copy()
         
         class_idx = np.argmax(feat_vec[5:])
         
         prob = feat_vec[5+class_idx] * feat_vec[0]

         if prob >= confidence_threshold:

            confirmed_anchor_box.append((prob,feat_vec))

   return confirmed_anchor_box


def non_max_supression(confirmed_anchor_box,diou_threshold=0.4):

   """
   confirmed_anchor_box -- list: [ (prob1,feat_vec1) , (prob2,feat_vec2) , ... ]
   """

   selected_object = []
   m = len(confirmed_anchor_box)

   while m != 0:

      #get and save largest fect_vec -- feat_vec: numpy.array
      feat_vec = confirmed_anchor_box[0][1]
      selected_object.append(feat_vec)

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

if __name__ == "__main__":
   
   """
   test_obj = np.random.randn(40,40,85)
   selected_obj = analyse_feature(test_obj)
   """
   
   path = os.getcwd()

   data_path = f"{path}/data"

   #get class info
   file = open(f"{data_path}/class_map.txt")
   class_info = json.load(file)
   file.close()
   
   class_color_map = preprocess_data.preprocess_class_color_map(class_info,data_path)
   reverse_class_info = preprocess_data.reverse_class_info(class_info,data_path)
   
   """
   model_path = f"{path}/model"
   image_path = f"{path}/pending_to_analysis"
   result_path = f"{path}/result"

   model = load_model(model_path)
   
   y = predict(model,image_path,reversed_class_map,class_color_map,confidence_threshold=0.8,diou_threshold=0.4,batch_size=1)
   """

