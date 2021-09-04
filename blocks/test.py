import tensorflow as tf
import tensorflow.keras.backend as K
import numpy as np

def func1(**kwargs):

   for val in kwargs.values():

      print(val)

def func2(*args):

   for i in args:

      print(i)

def func3(info,**kwargs):

   for val in kwargs.values():
      print(1)
      print(val)

   for val in info.values():

      print(val)

if __name__ == "__main__":

   """
   dict_test = {}
   dict_test["hpara1"] = (3,4,5,"same")
   dict_test["hpara2"] = (5,9,2,"valid")
   """
   """
   func1(**dict_test)

   func1(hpara1 =(6,7,8),hpara2=("a","b"))

   func2((7,8,9),(6,5,"same"))

   for i,val in enumerate(dict_test.values()):

      print(f"{i}:{val}")

   print(list(dict_test.keys()))

   for i in dict_test.keys():

      print(type(i))
   """
   """
   func3(dict_test)

   list_sample = []

   for i in range(9):

      list_sample.append(f"sample_{i}")

   """

   """
   a = tf.constant(np.random.randn(3,3))
   b = tf.constant(np.random.randn(3,3))
   c = K.cast(a<0.5,K.dtype(b))
   """

   """
   a = tf.constant(np.random.randn(3,3,1,4))
   b = tf.constant(np.random.randn(3,3,4))
   c = tf.reshape(b,(-1,4))
   """
   
   y_true = tf.constant(np.random.randn(3,3,4))

   ignore_mask = tf.TensorArray(K.dtype(y_true), size=1, dynamic_size=True)

   object_mask = tf.constant(np.random.randint(low=0,high=2,size=(3,3,1)))

   bool_mask = K.cast(object_mask,tf.bool)

   true_box = tf.boolean_mask(y_true,bool_mask[:,:,0])

   b1 =y_true[:,:,tf.newaxis,:]
   b2 = true_box[tf.newaxis,:,:]

   b1_xy = b1[:,:,:,2:4]
   b1_wh = b1[:,:,:,0:2]

   b2_xy = b2[:,:,2:4]
   b2_wh = b2[:,:,0:2]

   intersect_mins = K.maximum(b1_xy,b2_xy)
   intersect_maxes= K.minimum(b1_wh,b2_wh)

   area = intersect_mins[:,:,:,0] * intersect_maxes[:,:,:,1]
   
   best_iou = K.max(area,axis=-1,keepdims=True)

   ignore_mask = ignore_mask.write(0, K.cast(best_iou<0.5, K.dtype(true_box)))

   res = ignore_mask.stack()
 
