import tensorflow as tf
import numpy  as np

class CBS(tf.keras.Model):

   def __init__(self,filters=32,kernel_size=3,strides=2,padding='valid',**kwargs):

      #initialization
      super(CBS,self).__init__(**kwargs)

      #define layers
      self.conv2D_x = tf.keras.layers.Conv2D(filters=filters,kernel_size=kernel_size,strides=strides,padding=padding,data_format="channels_last")

      self.BN_x = tf.keras.layers.BatchNormalization(axis=-1)
      

   def call(self,inputs,train_flag=True):

      """
      input -- tensorflow layer with shape (m,n_H,n_W,n_C)
      """

      #convolution layer
      conv2D_x = self.conv2D_x(inputs)

      #Batch Normalization layer
      BN_x = self.BN_x(conv2D_x,training=train_flag)

      #activate by sigmoid
      output_sigmoid = tf.keras.activations.sigmoid(BN_x)

      return output_sigmoid
