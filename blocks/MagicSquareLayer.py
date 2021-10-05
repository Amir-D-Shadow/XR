import tensorflow as tf
import numpy as np


class MagicSquareLayer(tf.keras.layers.Layer):

    def __init__(self,feat_C = 1,**kwargs):

        super(MagicSquareLayer,self).__init__(**kwargs)

        self.num_C = feat_C


    def build(self,input_shape):

        self.M_kernel = tf.Variable(name="Magic Kernel",initial_value=tf.random_normal_initializer()(shape=(input_shape[-1],input_shape[-1]*self.num_C),dtype=tf.float64),trainable=True)

        self.M_bias = tf.Variable(name="Magic Bias",initial_value=tf.random_normal_initializer()(shape=(1,1,1,input_shape[-1]*self.num_C),dtype=tf.float64),trainable=True)


    def call(self,inputs):

        """
        inputs : (m,h,w,c)
        M_kernel : (c,c)
        M_bias: (1,1,1,c) 
        """

        #cast input type to fit the kernel's type
        x = tf.keras.backend.cast(inputs,dtype=tf.float64)

        #output_M: (m,h,w,c)
        output_M = tf.matmul(x,self.M_kernel) + self.M_bias

        return output_M

        
