import tensorflow  as tf
from CBL import CBL
from TCBL import TCBL
import math


class GroupAttentionLayer(tf.keras.layers.Layer):

   def __init__(self,ga_info,**kwargs):

      """

      chain_info -- dictionary containing information: CBL_Q ,CBL_K,CBL_V,receptive_field,sim_strides

                  
      Module Graph:

                      _________ CBL_Q _______________________                  
                     |                                       |                 
                     |                                       |                 
      inputs ------------------Conv_K -------------  similarity process ------ CBL_O
            |                                              |
            |______________________________________________|

      """


      super(GroupAttentionLayer,self).__init__(**kwargs)

      #CBL_Q
      filters,kernel_size,strides,padding = ga_info["CBL_Q"]
      
      self.CBL_Q = CBL(filters,kernel_size,strides,padding)

      #Conv_K
      filters,kernel_size,strides,padding = ga_info["Conv_K"]

      self.Conv_K =tf.keras.layers.Conv2D(filters=filters,kernel_size=kernel_size,strides=strides,padding=padding,data_format="channels_last",activation=None)

      #CBL_O
      filters,kernel_size,strides,padding = ga_info["CBL_O"]

      self.CBL_O = CBL(filters,kernel_size,strides,padding)

      #BN1
      self.BN1 = tf.keras.layers.BatchNormalization(axis=-1)


      #softmax C
      self.act_softmax_C = tf.keras.layers.Softmax(axis=(-1,-2))

      #softmax HW
      self.act_softmax_HW = tf.keras.layers.Softmax(axis=(1,2))

      #similarity info
      self.sim_RF = ga_info["receptive_field"] # height width determined by same size

      self.sim_strides = ga_info["sim_strides"] # height width share same strides


      
   def call(self,inputs,train_flag=True):

      #inputs: (m,h,w,c)
      inputs = tf.cast(inputs,tf.float64)

      #CBL_Q : (m,h,w,c)
      CBL_Q = self.CBL_Q(inputs,train_flag)
      CBL_Q = tf.cast(CBL_Q,tf.float64)

      #Conv_K : (m,h,w,c)
      Conv_K = self.Conv_K(inputs)
      Conv_K = tf.cast(Conv_K,tf.float64)

      #$#$#$#$#$#$#$#$#$$## similarity process $#$$#$#$#$#$#$#$#$$#$#$

      H = CBL_Q.shape[1]
      W = CBL_Q.shape[2]


      nH = int((H - self.sim_RF)/self.sim_strides) + 1
      nW = int((W - self.sim_RF)/self.sim_strides) + 1

      rf = tf.cast(self.sim_RF,tf.float64)
      
      #similarity process
      for h in range(nH):

         h_start_f = h * self.sim_strides
         h_end_f = h_start_f + self.sim_RF

         for w in range(nW):

            w_start_f = w * self.sim_strides
            w_end_f = w_start_f + self.sim_RF

            #dot similiarity feat_block :  (m,h,w,f,f)
            feat_block = tf.einsum("bijk,bpqk->bijpq",inputs,CBL_Q[:,h_start_f:h_end_f,w_start_f:w_end_f,:])

            #normalize feat_block :  (m,h,w,f,f)
            feat_block = feat_block / tf.math.sqrt(rf*rf)

            #activate feat_block :  (m,h,w,f,f)
            feat_block = self.act_softmax_C(feat_block)
            feat_block = tf.cast(feat_block,tf.float64)

            #dot with value : (m,h,w,c)
            feat_block = tf.einsum("bijpq,bpqk->bijk",feat_block,inputs[:,h_start_f:h_end_f,w_start_f:w_end_f,:])

            #add the dot similarity effect (m,h,w,c)
            Conv_K = Conv_K + feat_block


      #normalize (m,h,w,c)
      BN1 = self.BN1(Conv_K,training=train_flag) 

      #act softmax (m,h,w,c)
      yout = self.act_softmax_HW(BN1)

      #CBL_O (m,h,w,c)
      CBL_O = self.CBL_O(yout,train_flag)

      return CBL_O
      
