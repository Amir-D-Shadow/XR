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
      inputs: list [query,keys,TU,TL,values] -- [CBL_Q,CBL_K,CBL_TU,CBL_TL,CBL_V]

       query -- (m,h,w,c)
       keys -- (m,h,w,c)
       threshold -- (m,h,w,c)
       values -- (m,h,w,c)
      """
      query = inputs[0]
      keys = inputs[1]
      TU = inputs[2]
      TL = inputs[3]
      values = inputs[4]
      
      """
      process:(backup)

      query ------------------------------------------------------------------------
                                                                                   |
                                                                                   |
                  TU_(1x1xC)--------------------|                                  |--- Softmax[ query +  FFT_keys_spatial_iFFT ]------
                                                |                                  |                                                  |
               |-------- keys ------------------ ==== FFT_keys_spatial_iFFT -------|                                                  |
               |                                |                                                                                     | 
               |  TL_(1x1xC) -------------------|                                                                                     |
               |                                                                                                                      |
       key ----|                                                                                                                      |
               |   TU_(HxWx1)-------------------|                                                                                     |----------- values * Softmax[ query +  FFT_keys_spatial_iFFT ] * Softmax[ query +  FFT_keys_channels_iFFT ]
               |                                |                                                                                     |
               |-------- keys ------------------ ==== FFT_keys_channels_iFFT ------|                                                  |
                                                |                                  |                                                  |
                  TL_(HxWx1) -------------------|                                  |--- Softmax[ query +  FFT_keys_channels_iFFT ]----|
                                                                                   |                                                  |
                                                                                   |                                                  |
      query -----------------------------------------------------------------------|                                                  |              
                                                                                                                                      |                          
      values --------------------------------------------------------------------------------------------------------------------------

      """

      """
      process:

      query ------------------------------------------------------------------------------------------
                                                                                                     |
                                                                                                     |
      TU -------------------------------------                                                       |
                                             |                                                       |
                                             |                                                       |                                
      key --------------------------------------------FFT_filter_iFFT_key-------------- Softmax(FFT_filter_iFFT_key + query)-------------- values * Softmax(FFT_filter_iFFT_key + query)                                                                                                              |
                                             |                                                                                                                   |       
                                             |                                                                                                                   | 
      TL -------------------------------------                                                                                                                   |
                                                                                                                                                                 |                          
      values -----------------------------------------------------------------------------------------------------------------------------------------------------

      """


      #cast type for key -- (m,h,w,c)
      feat_keys = K.cast(keys,dtype=tf.complex128)


      #set up --- (m,1,1,1)
      feat_TU = K.sum(TU,axis=(1,2,3),keepdims=True)
      feat_TU = K.cast(feat_TU,dtype=K.dtype(self.alphaU))
      
      feat_TL = K.sum(TL,axis=(1,2,3),keepdims=True)
      feat_TL = K.cast(feat_TL,dtype=K.dtype(self.alphaU))

      #Get FFT FFT key -- (m,h,w,c)
      feat_keys_FT = tf.signal.fft3d( tf.signal.fft3d(feat_keys) )

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

      #get inverse FFT FFT key -- (m,h,w,c)
      iFT_feat_keys = tf.signal.ifft3d( tf.signal.ifft3d(feat_keys_FT) )

      #Get QK_additive -- (m,h,w,c)
      QK_additive = K.cast( tf.math.abs(iFT_feat_keys),dtype=K.dtype(query) ) + query

      #softmax activated -- (m,h,w,c)
      activated_QK = tf.keras.layers.Softmax(axis=(1,2,3))(QK_additive)

      #attention -- (m,h,w,c)
      output_values = values * activated_QK


      return output_values
   
         


      

class FourierAttentionModule(tf.keras.Model):


   def __init__(self,attention_info,**kwargs):

      """

      attention_info -- dictionary containing information: CBL_1,CBL_Q,CBL_K,CBL_Tmax,CBL_Tmin,CBL_V 

                     
      Module Graph:
      
                ___________________________________________________________________________
               |                                                                           |
               |                                                                           |
               |                                                                           |
               |     ----- CBL_Q ------------                                              |
               |     |                      |                                              |
      CBL1 ----------|---- CBL_K -----------|--- fourier_attention_step_process_layer --- Add --- BN --- leaky relu 
                     |                      |
                     |---- CBL_TU ----------|
                     |---- CBL_TL ----------|
                     |                      |
                     ----- CBL_V ------------
      
      """

      super(FourierAttentionModule,self).__init__(**kwargs)

      #CBL_1
      filters,kernel_size,strides,padding = attention_info["CBL_1"]
      
      self.CBL_1 = CBL(filters,kernel_size,strides,padding)

      #CBL_Q
      filters,kernel_size,strides,padding = attention_info["CBL_Q"]

      self.CBL_Q = CBL(filters,kernel_size,strides,padding)

      #CBL_K
      filters,kernel_size,strides,padding = attention_info["CBL_K"]

      self.CBL_K = CBL(filters,kernel_size,strides,padding)

      #CBL_TU
      filters,kernel_size,strides,padding = attention_info["CBL_TU"]

      self.CBL_TU = CBL(filters,kernel_size,strides,padding)

      #CBL_TL
      filters,kernel_size,strides,padding = attention_info["CBL_TL"]

      self.CBL_TL = CBL(filters,kernel_size,strides,padding)

      #CBL_V
      filters,kernel_size,strides,padding = attention_info["CBL_V"]

      self.CBL_V = CBL(filters,kernel_size,strides,padding)

      #fourier_attention
      self.fourier_attention = fourier_attention_step_process_layer()

      #Add layer
      self.Add_layer = tf.keras.layers.Add()

      #BN_1
      self.BN_1 = tf.keras.layers.BatchNormalization(axis=-1)

      #leakyRelu
      self.output_leakyrelu = tf.keras.layers.LeakyReLU()



   def call(self,inputs,train_flag=True):

      #CBL_1
      CBL_1 = self.CBL_1(inputs,train_flag)

      #CBL_Q
      CBL_Q = self.CBL_Q(CBL_1,train_flag)

      #CBL_K
      CBL_K = self.CBL_K(CBL_1,train_flag)

      #CBL_TU
      CBL_TU = self.CBL_TU(CBL_1,train_flag)

      #CBL_TL
      CBL_TL = self.CBL_TL(CBL_1,train_flag)

      #CBL_V
      CBL_V = self.CBL_V(CBL_1,train_flag)

      #fourier_attention
      fourier_attention = self.fourier_attention((CBL_Q,CBL_K,CBL_TU,CBL_TL,CBL_V))

      #Add layer
      Add_layer = self.Add_layer([CBL_1,fourier_attention])

      #layer normalization
      BN_1 = self.BN_1(Add_layer,training = train_flag)

      #output_leakyrelu
      output_leakyrelu = self.output_leakyrelu(BN_1)

      return output_leakyrelu
      

   def graph_model(self,dim):

      x = tf.keras.layers.Input(shape=dim)
      
      return tf.keras.Model(inputs=x,outputs=self.call(x))


      
