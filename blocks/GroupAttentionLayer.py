import tensorflow  as tf
from CBL import CBL
from TCBL import TCBL
import math


class GroupAttentionLayer(tf.keras.layers.Layer):

   def __init__(self,ga_info,**kwargs):

      """

      chain_info -- dictionary containing information: CBL_Q ,CBL_K,CBL_V,receptive_field,sim_strides

                  
      Module Graph:

                      _________ CBL_Q ____                  
                     |                    |                 
                     |                    |                 
      inputs ------------------CBL_K -------------  similarity process 
                     |                    |
                     |_________ CBL_V ____|

      """


      super(GroupAttentionLayer,self).__init__(**kwargs)

      #CBL_Q
      filters,kernel_size,strides,padding = ga_info["CBL_Q"]
      
      self.CBL_Q = CBL(filters,kernel_size,strides,padding)

      #CBL_K
      filters,kernel_size,strides,padding = ga_info["CBL_K"]

      self.CBL_K = CBL(filters,kernel_size,strides,padding)

      #CBL_V
      filters,kernel_size,strides,padding = ga_info["CBL_V"]

      self.CBL_V = CBL(filters,kernel_size,strides,padding)

      #BN1
      self.BN1 = tf.keras.layers.BatchNormalization(axis=-1)

      #LR_act
      self.act_LR = tf.keras.layers.LeakyReLU()

      #softmax
      self.act_softmax = tf.keras.layers.Softmax(axis=(1,2))

      #similarity info
      self.sim_RF = ga_info["receptive_field"] # height width determined by same size

      self.sim_strides = ga_info["sim_strides"] # height width share same strides


   def build(self,input_shape):
      
      H,W = input_shape[1],input_shape[2]
      
      nH = int((H - self.sim_RF)/self.sim_strides) + 1
      nW = int((W - self.sim_RF)/self.sim_strides) + 1

      #self.beta_bias =  tf.Variable(name="beta_bias",initial_value=tf.random_normal_initializer()(shape=(1,1,1,nH*nW),dtype="float64"),trainable=True)      
      
   def call(self,inputs,train_flag=True):


      #CBL_Q : (m,h,w,c)
      CBL_Q = self.CBL_Q(inputs,train_flag)
      CBL_Q = tf.cast(CBL_Q,tf.float64)

      #CBL_K : (m,h,w,c)
      CBL_K = self.CBL_K(inputs,train_flag)
      CBL_K = tf.cast(CBL_K,tf.float64)

      #CBL_V : (m,h,w,c)
      CBL_V = self.CBL_V(inputs,train_flag)
      CBL_V = tf.cast(CBL_V,tf.float64)

      #$#$#$#$#$#$#$#$#$$## similarity process $#$$#$#$#$#$#$#$#$$#$#$

      H = CBL_Q.shape[1]
      W = CBL_Q.shape[2]
      C = CBL_Q.shape[3]


      nH = int((H - self.sim_RF)/self.sim_strides) + 1
      nW = int((W - self.sim_RF)/self.sim_strides) + 1

      yout = 0.0
      yout = tf.cast(yout,tf.float64)

      C = tf.cast(C,tf.float64)
      
      #similarity process
      for h in range(nH):

         h_start_f = h * self.sim_strides
         h_end_f = h_start_f + self.sim_RF

         for w in range(nW):

            w_start_f = w * self.sim_strides
            w_end_f = w_start_f + self.sim_RF

            #dot similiarity feat_block :  (m,h,w,f,f)
            feat_block = tf.einsum("bijk,bpqk->bijpq",CBL_Q,CBL_K[:,h_start_f:h_end_f,w_start_f:w_end_f,:])

            #normalize feat_block :  (m,h,w,f,f)
            feat_block = feat_block / tf.math.sqrt(C)

            #activate feat_block :  (m,h,w,f,f)
            feat_block = self.act_LR(feat_block)
            feat_block = tf.cast(feat_block,tf.float64)

            #dot with value : (m,h,w,c)
            feat_block = tf.einsum("bijpq,bpqk->bijk",feat_block,CBL_V[:,h_start_f:h_end_f,w_start_f:w_end_f,:])

            #add the dot similarity effect
            yout = yout + feat_block


      #normalize
      yout = self.BN1(yout,training=train_flag) 

      #act softmax
      yout = self.act_softmax(yout)

      return yout
      
