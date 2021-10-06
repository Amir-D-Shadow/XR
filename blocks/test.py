import tensorflow as tf
import tensorflow.python.keras.backend as K
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
   """
   y_true = tf.constant(np.random.randn(16,16,4))

   ignore_mask = tf.TensorArray(K.dtype(y_true), size=1, dynamic_size=True)

   object_mask = tf.constant(np.random.randint(low=0,high=2,size=(16,16,1)))

   count_true = K.sum(object_mask)

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
   """
   """
   q = tf.constant(np.random.randn(5,5,9))
   k = tf.constant(np.random.randn(5,5,9))
   v = tf.constant(np.random.randn(5,5,9))

   q = q[:,:,tf.newaxis,:]

   k = tf.reshape(k,(-1,9))

   v = tf.reshape(k,(-1,9))

   res1 = q*k

   res2 = K.sum(res1,axis=-1)

   res3 = tf.keras.layers.Softmax(axis=-1)(res2)

   v = K.cast(v,K.dtype(res3))

   output = tf.matmul(res3,v)
   """

   """
   q = tf.constant(np.random.randn(5,5,9))
   k = tf.constant(np.random.randn(5,5,9))
   v = tf.constant(np.random.randn(5,5,9))

   res = np.fft.fftn(q,axes=(0,1))

   """

   """
   array = tf.TensorArray(tf.float64,size=1,dynamic_size=True)
   
   a = tf.constant(np.random.randn(10))
   b = tf.constant(np.random.randn(10))
   c = tf.constant(np.random.randn(10))
   d = tf.constant(np.random.randn(10))
   e = tf.constant(np.random.randn(10))
   f = tf.constant(np.random.randn(10))

   array = array.write(0,a)
   array = array.write(1,b)
   array = array.write(2,c)
   array = array.write(3,d)
   array = array.write(4,e)
   array = array.write(5,f)
   
   k = tf.constant(np.random.randn(10,4,4,2))

   k_mean = tf.math.reduce_mean(k,axis=(1,2,3))
   k_var = tf.math.reduce_variance(k,axis=(1,2,3))

   array = array.write(6,k_mean)
   array = array.write(7,k_var)
   
   array = array.stack()

   new_k = tf.constant(np.random.randn(2,4,4,2))
   k_mean_new=tf.math.reduce_mean(new_k,axis=(1,2,3),keepdims=True)
   
   test1 = K.maximum(new_k-k_mean_new,0)
   
   mask = K.cast( test1 != 0 , tf.float64)

   test2 = test1*mask

   new_array = tf.reshape(array,shape=(-1,8))
   test_dense1 = tf.keras.layers.Dense(128)(new_array)

   test_dense2 = tf.keras.layers.Dense(256)(test_dense1)

   """

   array = tf.TensorArray(tf.float64,size=1,dynamic_size=True)

   data = tf.constant(np.random.randn(10,32,32,64))

   skip1 = tf.constant(np.random.randn(10,32,32,64))

   a = data[1,:,:,:]
   b = skip1[1,:,:,:]

   bh = b.shape[0]
   bw = b.shape[1]
   bc = b.shape[2]

   b_new = tf.reshape(b,shape=(bc,-1))

   c = tf.matmul(a,b_new)

   array = array.write(0,c)

   a = data[3,:,:,:]
   b = skip1[3,:,:,:]

   bh = b.shape[0]
   bw = b.shape[1]
   bc = b.shape[2]

   b_new = tf.reshape(b,shape=(bc,-1))

   c = tf.matmul(a,b_new)

   array = array.write(1,c)

   array = array.stack()

   res = tf.keras.layers.SeparableConv2D(16,1,1,"same")(data)
