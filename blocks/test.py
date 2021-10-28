import tensorflow as tf
import tensorflow.python.keras.backend as K
import numpy as np

from CBL import CBL
from TCBL import TCBL
from GroupAttentionLayer import GroupAttentionLayer

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

def build_model_single(M_info):

  inputs_x = tf.keras.layers.Input(shape=(32,32,3))

  
  CBL_0 = CBL(64,3,1,"same")(inputs_x)
  MP0 = tf.keras.layers.MaxPool2D(2)(CBL_0)
  
  ga0 = GroupAttentionLayer(M_info)(MP0)
  CBL_1 = CBL(64,3,1,"same")(ga0)

  Add_0 = tf.keras.layers.Add()([MP0,CBL_1])
  TCBL_0 = TCBL(64,3,1,"valid")(Add_0)
  
  MP1 = tf.keras.layers.MaxPool2D(2)(TCBL_0)
  
  Flatten_1 = tf.keras.layers.Flatten()(MP1)
  dense1 = tf.keras.layers.Dense(64,activation=tf.keras.layers.LeakyReLU())(Flatten_1)
  dense_out = tf.keras.layers.Dense(10)(dense1)

  model = tf.keras.Model(inputs=inputs_x,outputs=dense_out)

  model.compile(optimizer="adam",loss= tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),metrics= ["accuracy"])

  return  model

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
   """

   """
   array = tf.TensorArray(tf.float64,size=1,dynamic_size=True)
   
   a = tf.constant(np.random.randn(10,1,1,16))
   b = tf.constant(np.random.randn(10,1,1,16))
   c = tf.constant(np.random.randn(10,1,1,16))
   d = tf.constant(np.random.randn(10,1,1,16))
   e = tf.constant(np.random.randn(10,1,1,16))
   f = tf.constant(np.random.randn(10,1,1,16))

   array = array.write(0,a)
   array = array.write(1,b)
   array = array.write(2,c)
   array = array.write(3,d)
   array = array.write(4,e)
   array = array.write(5,f)
   """


   """
   
   (train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.cifar10.load_data()
   
   one_device_strategy = tf.distribute.OneDeviceStrategy(device="GPU:0")



   model_cluster = {}
   with one_device_strategy.scope():

       ga_info = {}
       ga_info["CBL_Q"] = (64,3,1,"same")
       ga_info["CBL_K"] = (64,3,1,"same")
       ga_info["TCBL_out"] = (64,3,1,"valid")
       ga_info["receptive_field"] = 3
       ga_info["sim_strides"] = 1

       model = build_model_single(ga_info)

   history_FA0 = model.fit(train_images,train_labels,epochs=10,batch_size=64,validation_data=(test_images,test_labels))
   """
   """
   q = tf.constant(np.random.randn(10,16,16,64))

   k = tf.constant(np.random.randn(10,3,3,64))

   v = tf.constant(np.random.randn(10,3,3,64))

   d = tf.einsum("bijk,bpqk->bijpq",q,k)

   f = tf.einsum("bijpq,bpqk->bijk",d,v)

   y_true = tf.constant(np.random.randn(10,16,16,20))
   y_pred = tf.constant(np.random.randn(10,16,16,20))

   #losses = tf.keras.losses.BinaryCrossentropy(from_logits=True,axis=-1,reduction=tf.keras.losses.Reduction.NONE)(y_true,y_pred)
   #import tensorflow_addons as tfa
   #losses =  tfa.losses.SigmoidFocalCrossEntropy(from_logits=True,alpha=1,gamma=2,reduction=tf.keras.losses.Reduction.NONE)(y_true,y_pred)

   losses  = tf.keras.losses.CategoricalCrossentropy(from_logits=True,axis=-1,reduction=tf.keras.losses.Reduction.NONE)(y_true,y_pred)
   """
   """
   a = tf.constant(np.random.randn(10,3,3,20))

   b = tf.keras.layers.Dense(64)(a)
   """
   n = 10
   images = np.random.randn(8,10,10,6)#np.array([[[[x * n + y + 1] for y in range(n)] for x in range(n)]])

   y = tf.image.extract_patches(images=images,
                           sizes=[1, 3, 3, 1],
                           strides=[1, 1, 1, 1],
                           rates=[1, 1, 1, 1],
                           padding='SAME')

   a = np.random.randn(5,10,10,9,8,2)
   b = np.random.randn(5,10,10,9,1,2)
   c = tf.multiply(b,a)
   
