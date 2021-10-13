import tensorflow  as tf
from CBL import CBL
from TCBL import TCBL
import math


class GroupAttentionLayer(tf.keras.layers.Layer):

   def __init__(self,ga_info,**kwargs):

      """

      chain_info -- dictionary containing information: CBL_Q ,CBL_O,DSC

                  
      Module Graph:

                      _________ CBL_Q ____                  
                     |                    |                 
                     |                    |                 
      inputs ----------                     ---------  similarity process ------------ TCBL_out
                     |                    |
                     |_________ CBL_K ____|

      """


      super(GroupAttentionLayer,self).__init__(**kwargs)

      #CBL_Q
      filters,kernel_size,strides,padding = ga_info["CBL_Q"]
      
      self.CBL_Q = CBL(filters,kernel_size,strides,padding)

      #CBL_K
      filters,kernel_size,strides,padding = ga_info["CBL_K"]

      self.CBL_K = CBL(filters,kernel_size,strides,padding)

      #BN
      self.BN1 = tf.keras.layers.BatchNormalization(axis=-1)

      #act
      self.act1 = tf.keras.layers.LeakyReLU()

      #CBL_out
      filters,kernel_size,strides,padding = ga_info["TCBL_out"]

      self.TCBL_out = TCBL(filters,kernel_size,strides,padding)

      #similarity info
      self.sim_RF = ga_info["receptive_field"] # height width determined by same size

      self.sim_strides = ga_info["sim_strides"] # height width share same strides


   def build(self,input_shape):
      
      H,W = input_shape[1],input_shape[2]
      
      nH = int((H - self.sim_RF)/self.sim_strides) + 1
      nW = int((W - self.sim_RF)/self.sim_strides) + 1

      self.beta_bias =  tf.Variable(name="beta_bias",initial_value=tf.random_normal_initializer()(shape=(1,1,1,nH*nW),dtype="float64"),trainable=True)      
      
   def call(self,inputs,train_flag=True):

      """
      inputs : (m,h,w,c)
      """

      #CBL_Q
      CBL_Q = self.CBL_Q(inputs,train_flag)
      CBL_Q = tf.cast(CBL_Q,tf.float64)

      #CBL_K
      CBL_K = self.CBL_K(inputs,train_flag)
      CBL_K = tf.cast(CBL_K,tf.float64)

      #$#$#$#$#$#$#$#$#$$## similarity process $#$$#$#$#$#$#$#$#$$#$#$

      H,W,C = CBL_Q.shape[1],CBL_Q.shape[2],CBL_Q.shape[3]

      nH = int((H - self.sim_RF)/self.sim_strides) + 1
      nW = int((W - self.sim_RF)/self.sim_strides) + 1

      sim_out = tf.TensorArray(tf.float64,size=1,dynamic_size=True)

      C_out = 0

      #conv over CBL_Q from CBL_K tensors
      for h in range(nH):

         h_start_f = h * self.sim_strides
         h_end_f = h_start_f + self.sim_RF

         for w in range(nW):

            w_start_f = w * self.sim_strides
            w_end_f = w_start_f + self.sim_RF

            #h_start_f , w_start_f , h_end_f , w_end_f :filter perspective

            #create feat_vec to store attention result of a group of feat vec
            #expected shape in output : (m,nH * nW) 
            feat_vec = tf.TensorArray(tf.float64,size=1,dynamic_size = True)

            nc = 0

            #loop via CBL_Q
            for hq in range(nH):

               h_start_q = hq * self.sim_strides
               h_end_q = h_start_q + self.sim_RF

               for wq in range(nW):

                  w_start_q = wq * self.sim_strides
                  w_end_q = w_start_q + self.sim_RF

                  #dot product similarity
                  res = CBL_Q[:,h_start_q:h_end_q,w_start_q:w_end_q,:] * CBL_K[:,h_start_f:h_end_f,w_start_f:w_end_f,:]
            
                  res = tf.math.reduce_sum(res,axis=(1,2,3))

                  feat_vec = feat_vec.write(nc,res)

                  #update nc
                  nc = nc + 1

            #feat_vec : (nH*nW,m)
            feat_vec = feat_vec.stack()

            #feat_vec reshape : (m,nH*nW)
            feat_vec = tf.transpose( feat_vec , perm = [1,0])

            #save result
            sim_out = sim_out.write(C_out,feat_vec)

            #update C_out
            C_out = C_out + 1

      #sim_out : (nH * nW ,m ,nH * nW)
      sim_out = sim_out.stack()

      #sim_out : (m,n_H*n_W,n_H*n_W)
      sim_out = tf.transpose(sim_out , perm=(1,0,2))

      #sim_out : (m,n_H,n_W,n_H*n_W)
      sim_out = tf.reshape(sim_out , shape = (-1,nH,nW,nH*nW))
                  
      #$#$#$#$#$#$#$#$#$$## similarity process $#$$#$#$#$#$#$#$#$$#$#$

      sim_out = sim_out + self.beta_bias

      BN1 = self.BN1(sim_out,train_flag)

      act1 = self.act1(BN1)

      TCBL_out = self.TCBL_out(act1,train_flag)

      return TCBL_out
      
