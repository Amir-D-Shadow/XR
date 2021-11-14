import tensorflow as tf
from tensorflow.python.keras import backend as K
import numpy as np
from ResMLP import ResMLP

      
class FourierAttentionModule(tf.keras.layers.Layer):


   def __init__(self,filters,**kwargs):

      """

      attention_info -- dictionary containing information: units,MLP_num,output_units

                     
      Module Graph:
      

                                                                                                                                  
      inputs --------- FFT -- ResMLP * n ------- filtering ----  iFFT 


      
      """

      super(FourierAttentionModule,self).__init__(**kwargs)

      self.filters = filters

      self.conv_output = tf.keras.layers.Conv2D(filters=self.filters,kernel_size=3,strides=1,padding="same",data_format="channels_last",activation=tf.keras.layers.LeakyReLU())

   def build(self,input_shape):
    
      _,H,W,C = input_shape

      self.CBL_q = CBL(C,1,1,"same")
      self.CBL_v = CBL(C,1,1,"same")
      self.reshape_q = tf.keras.layers.Reshape(target_shape=(H*W,C))
      self.reshape_v = tf.keras.layers.Reshape(target_shape=(H*W,C))
      self.attention_layer = tf.keras.layers.Attention()
      self.reshape_out = tf.keras.layers.Reshape(target_shape=(H,W,C))


   def call(self,inputs,train_flag=True):

      #FFT
      feat_FT = tf.signal.fft3d(K.cast(inputs,tf.complex64))

      #real
      feat_FT_real = tf.math.real( feat_FT )
      feat_FT_real = K.cast(feat_FT_real,K.dtype(inputs))
        
      #q v
      q_val = self.CBL_q(feat_FT_real,train_flag)
      q_val = self.reshape_q(q_val)
      v_val = self.CBL_v(feat_FT_real,train_flag)
      v_val = self.reshape_v(v_val)
      
      #attention
      final_tensor = self.attention_layer([q_val,v_val])
      final_tensor = self.reshape_out(final_tensor)

      #final result
      final_tensor = self.conv_output(final_tensor)
      final_tensor = K.cast(final_tensor,K.dtype(inputs))

      return final_tensor
      

   def graph_model(self,dim):

      x = tf.keras.layers.Input(shape=dim)
      
      return tf.keras.Model(inputs=x,outputs=self.call(x))


      
