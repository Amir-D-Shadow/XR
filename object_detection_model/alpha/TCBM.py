import tensorflow as tf
import numpy  as np

#activation Mish
def Mish(x):

   softplus = tf.math.softplus(x)
   tanh_s = tf.math.tanh(softplus)

   return (x * tanh_s)


#TCBM Module
class TCBM(tf.keras.Model):

   def __init__(self,filters=32,kernel_size=3,strides=2,padding="valid",**kwargs):

      #initialization
      super(TCBM,self).__init__(**kwargs)

      #define layers
      self.conv2D_transpose_x = tf.keras.layers.Conv2DTranspose(filters=filters,kernel_size=kernel_size,strides=strides,padding=padding,data_format="channels_last")

      self.BN_x = tf.keras.layers.BatchNormalization(axis=-1)

      self.output_Mish = tf.keras.layers.Lambda(Mish)

      

   def call(self,inputs,train_flag=True):

      """
      input -- tensorflow layer with shape (m,n_H,n_W,n_C)
      """

      #Transpose Convolution 2D layer
      conv2D_transpose_x = self.conv2D_transpose_x(inputs)

      #Batch Normalization layer
      BN_x = self.BN_x(conv2D_transpose_x,training=train_flag)

      #activate by Mish
      output_Mish = self.output_Mish(BN_x)

      return output_Mish
