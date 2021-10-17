import tensorflow  as tf
from CBL import CBL
from TCBL import TCBL
from ResMLP import ResMLP
import math




class Involution(tf.keras.layers.Layer):

   def __init__(self,invol_info,**kwargs):


      super(Involution,self).__init__(**kwargs)

      #info
      self.units = invol_info["units"]
      self.kernel_size = invol_info["kernel_size"]
      self.strides = invol_info["strides"]

      #kernel
      self.k1 = tf.keras.layers.Dense(self.units)

      #BN1
      self.BN1 = tf.keras.layers.BatchNormalization(axis=-1)

      #act1
      self.act1 = tf.keras.layers.LeakyReLU()

   def build(self,inputs_shape):

      self.yout = tf.Variable(tf.zeros(shape=inputs_shape),trainable=True)
      
   def call(self,inputs,train_flag=True):


      H = inputs.shape[1]
      W = inputs.shape[2]
      C = inputs.shape[3]


      nH = int((H - self.kernel_size)/self.strides) + 1
      nW = int((W - self.kernel_size)/self.strides) + 1

      self.yout = self.yout * 0
      
      for h in range(nH):

         h_start_f = h * self.strides
         h_end_f = h_start_f + self.kernel_size

         for w in range(nW):

            w_start_f = w * self.strides
            w_end_f = w_start_f + self.kernel_size

            h_idx = (h_start_f + h_end_f)//2
            w_idx = (w_start_f + w_end_f)//2
            
            kernel =  self.k1(inputs[:,h_idx:h_idx+1,w_idx:w_idx+1,:])
            kernel = tf.reshape(kernel,(-1,self.kernel_size,self.kernel_size,1))

            yout[:,h_start_f:h_end_f,w_start_f:w_end_f,:].assign(yout[:,h_start_f:h_end_f,w_start_f:w_end_f,:] + inputs[:,h_start_f:h_end_f,w_start_f:w_end_f,:] * kernel )

      #normalize
      yout = self.BN1(yout,training=train_flag) 

      #act LR
      yout = self.act1(yout)

      return yout
   
