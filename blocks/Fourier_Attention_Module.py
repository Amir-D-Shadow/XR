import tensorflow as tf
from tensorflow.keras import backend as K
import numpy as np
import math
from CBL import CBL

class fourier_attention_step_process_layer(tf.keras.layers.Layer):

   def __init__(self):

      super(fourier_attention_step_process_layer,self).__init__()
      

   def build(self,input_shape):

      """
      lower bound: maximum(val - alphaL,TL)
      
      upper bound: maximum(alphaU - val,TU)
      """

      #threshold
      self.alphaL = tf.Variable(name="alphaL",initial_value=tf.random_normal_initializer()(shape=(1,),dtype="float64"),trainable=True)
      self.alphaU = tf.Variable(name="alphaU",initial_value=tf.random_normal_initializer()(shape=(1,),dtype="float64"),trainable=True)

      #scale factor
      self.betaL =  tf.Variable(name="betaL",initial_value=tf.random_normal_initializer()(shape=(1,),dtype="float64"),trainable=True)
      self.betaU =  tf.Variable(name="betaU",initial_value=tf.random_normal_initializer()(shape=(1,),dtype="float64"),trainable=True)

      
   def call(self,inputs):

      """
      inputs: list [keys,TU,TL] -- [CBL_K,CBL_TU,CBL_TL]

       keys -- (m,h,w,c)
       threshold -- (m,h,w,c)

      """

      keys = inputs[0]
      TU = inputs[1]
      TL = inputs[2]
      

      """
      process:

                                                                                                     
                                                                                                     
      TU -------------------------------------                                                       
                                             |                                                       
                                             |                                                                                     
      key --------------------------------------------FFT_filter_iFFT_key                                                                                                             |
                                             |                                                                                                                     
                                             |                                                                                                                   
      TL -------------------------------------                                                                                                                   
                                                                                                                                                                                           
      

      """


      #cast type for key -- (m,h,w,c)
      feat_keys = K.cast(keys,dtype=tf.complex128)


      #set up --- (m,1,1,1)
      feat_TU = K.sum(TU,axis=(1,2,3),keepdims=True)
      feat_TU = K.cast(feat_TU,dtype=K.dtype(self.alphaU))
      
      feat_TL = K.sum(TL,axis=(1,2,3),keepdims=True)
      feat_TL = K.cast(feat_TL,dtype=K.dtype(self.alphaU))

      #Get FFT key -- (m,h,w,c)
      feat_keys_FT = tf.signal.fft3d(feat_keys) 

      #Get real part -- (m,h,w,c)
      feat_keys_FT_real = tf.math.real(feat_keys_FT)
      feat_keys_FT_real = K.cast(feat_keys_FT_real,dtype=K.dtype(feat_TU))


      #filtering min -- (m,h,w,c)
      tmp_min_residue = feat_keys_FT_real * K.maximum( feat_keys_FT_real - feat_TL , self.alphaL ) / (feat_keys_FT_real - self.betaL * feat_TL )
      tmp_min_residue = feat_keys_FT_real - tmp_min_residue
      
      tmp_min_residue = K.cast(tmp_min_residue,dtype=K.dtype(feat_keys_FT))

      #filtering max -- (m,h,w,c)
      tmp_max_residue = feat_keys_FT_real * K.maximum( feat_TU - feat_keys_FT_real , self.alphaU ) / (self.betaU * feat_TU - feat_keys_FT_real )
      tmp_max_residue = feat_keys_FT_real - tmp_max_residue

      tmp_max_residue = K.cast(tmp_max_residue,dtype=K.dtype(feat_keys_FT))

      #update filtered feat_keys_FT -- (m,h,w,c)
      feat_keys_FT = feat_keys_FT - tmp_max_residue - tmp_min_residue

      #get inverse FFT key -- (m,h,w,c)
      iFT_feat_keys = tf.signal.ifft3d(feat_keys_FT)

      #get output
      output_keys = tf.math.real(iFT_feat_keys)

      output_keys = K.cast(output_keys,K.dtype(keys))


      return output_keys
   
         


      

class FourierAttentionModule(tf.keras.Model):


   def __init__(self,attention_info,**kwargs):

      """

      attention_info -- dictionary containing information: CBL_1,CBL_TU,CBL_K,CBL_TL

                     
      Module Graph:
      

                                                                                       
                     ----- CBL_TU -----------                                              
                     |                      |                                              
      CBL1 ----------|---- CBL_K -----------|--- fourier_attention_step_process_layer 
                     |                      |
                     |---- CBL_TL ----------|

      
      """

      super(FourierAttentionModule,self).__init__(**kwargs)

      #CBL_1
      filters,kernel_size,strides,padding = attention_info["CBL_1"]
      
      self.CBL_1 = CBL(filters,kernel_size,strides,padding)

      #CBL_K
      filters,kernel_size,strides,padding = attention_info["CBL_K"]

      self.CBL_K = CBL(filters,kernel_size,strides,padding)

      #CBL_TU
      filters,kernel_size,strides,padding = attention_info["CBL_TU"]

      self.CBL_TU = CBL(filters,kernel_size,strides,padding)

      #CBL_TL
      filters,kernel_size,strides,padding = attention_info["CBL_TL"]

      self.CBL_TL = CBL(filters,kernel_size,strides,padding)

      #fourier_attention
      self.fourier_attention = fourier_attention_step_process_layer()



   def call(self,inputs,train_flag=True):

      #CBL_1
      CBL_1 = self.CBL_1(inputs,train_flag)

      #CBL_K
      CBL_K = self.CBL_K(CBL_1,train_flag)

      #CBL_TU
      CBL_TU = self.CBL_TU(CBL_1,train_flag)

      #CBL_TL
      CBL_TL = self.CBL_TL(CBL_1,train_flag)


      #fourier_attention
      fourier_attention = self.fourier_attention((CBL_K,CBL_TU,CBL_TL))


      return fourier_attention
      

   def graph_model(self,dim):

      x = tf.keras.layers.Input(shape=dim)
      
      return tf.keras.Model(inputs=x,outputs=self.call(x))


      
