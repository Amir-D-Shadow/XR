import tensorflow as tf
import numpy as np


class MagicSquareLayer(tf.keras.layers.Layer):

    def __init__(self,feat_alpha=0.03,feat_scale = 1,**kwargs):

        super(MagicSquareLayer,self).__init__(**kwargs)

        self.layer_scale = feat_scale
        
        self.layer_alpha = feat_alpha
        

    def build(self,input_shape):

        #scale dimension
        H_W = input_shape[1]*input_shape[2] / self.layer_scale

        H_W = tf.cast(H_W,dtype=tf.int64)

        #initialization
        self.K_Kernel = tf.Variable(name="K Kernel",initial_value=tf.random_normal_initializer()(shape=(input_shape[-1],H_W),dtype=tf.float64),trainable=True)

        self.V_Kernel = tf.Variable(name="V Kernel",initial_value=tf.random_normal_initializer()(shape=(H_W,input_shape[-1]),dtype=tf.float64),trainable=True)

        self.V_bias = tf.Variable(name="V Bias",initial_value=tf.random_normal_initializer()(shape=(1,1,1,input_shape[-1]),dtype=tf.float64),trainable=True)


    def call(self,inputs):

        """
        inputs : (m,h,w,c)
        K_Kernel : (c,h x w)
        V_Kernel : (h x w,c)
        V_bias: (1,1,1,c) 
        """

        #cast input type to fit the kernel's type
        x = tf.keras.backend.cast(inputs,dtype=tf.float64)

        #output_M: (m,H,W,H_W)
        output_M = tf.matmul(x,self.K_Kernel)

        #output_M: (m,H,W,H_W)
        output_M = tf.math.maximum(output_M, self.layer_alpha * output_M )

        #output_M: (m,H,W,c)
        output_M = tf.matmul(output_M,self.V_Kernel) + self.V_bias

        return output_M

        
