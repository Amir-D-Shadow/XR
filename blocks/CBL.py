import tensorflow as tf
import numpy  as np

class CBL(tf.keras.Model):

   def __init__(self,filters=32,kernel_size=3,strides=2,padding='valid',**kwargs):

      #initialization
      super(CBL,self).__init__(**kwargs)

      #define layers
      self.conv2D_x = tf.keras.layers.Conv2D(filters=filters,kernel_size=kernel_size,strides=strides,padding=padding)

      self.BN_x = tf.keras.layers.BatchNormalization(axis=-1)

      self.output_leaky_relu = tf.keras.layers.LeakyReLU()
      

   def call(self,inputs):

      """
      input -- tensorflow layer with shape (m,n_H,n_W,n_C)
      """

      #convolution layer
      conv2D_x = self.conv2D_x(inputs)

      #Batch Normalization layer
      BN_x = self.BN_x(conv2D_x)

      #activate by Leaky relu
      output_leaky_relu = self.output_leaky_relu(BN_x)

      return output_leaky_relu
      
