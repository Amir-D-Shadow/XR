import tensorflow  as tf
from CBL import CBL


class Involution(tf.keras.layers.Layer):

   def __init__(self,filters,kernel_size,strides,group_number=1,rates=1,**kwargs):


      super(Involution,self).__init__(**kwargs)

      #info
      self.filters = filters
      self.kernel_size = kernel_size
      self.strides = strides
      self.group_number = group_number
      self.rates = rates


   def build(self,input_shape):

      #(m,H,W,C)
      _,H,W,C = input_shape

      #bias
      self.V_bias = tf.Variable(name="V Bias",initial_value=tf.random_normal_initializer()(shape=(1,1,1,1,C//self.group_number,self.group_number),dtype=tf.float64),trainable=True)


      #kernel generator

      #from (m,H,W,K*K*C) to (m,H,W,filter)
      self.CBL_KG_p1 = CBL(self.filters,1,1,"same")

      #from (m,H,W,filter) to (m,H,W,K*K*G) [G:group_number]
      self.CBL_KG_p2 = tf.keras.layers.Conv2D(filters=self.kernel_size*self.kernel_size*self.group_number,kernel_size=1,strides=1,padding="same",data_format="channels_last") #CBL(self.kernel_size*self.kernel_size*self.group_number,1,1,"same")

      #reshape inputs (m,H,W,K*K*C) --> (m,H,W,K*K,C//G,G) [G:group_number]
      self.input_reshape = tf.keras.layers.Reshape(target_shape=(H,W,self.kernel_size * self.kernel_size,C//self.group_number,self.group_number))

      #reshape kernel (m,H,W,K*K*group_number) --> (m,H,W,K*K,1,G) [G:group_number]
      self.kernel_reshape = tf.keras.layers.Reshape(target_shape=(H,W,self.kernel_size*self.kernel_size,1,self.group_number))

      #mult layer
      self.MultLayer = tf.keras.layers.Multiply()

      #reshape output (m,H,W,C)
      self.output_reshape = tf.keras.layers.Reshape(target_shape=(H,W,C))

      #BN_out
      self.BN_out = tf.keras.layers.BatchNormalization(axis=-1)

      #act_out
      self.act_out = tf.keras.layers.LeakyReLU()

   def call(self,inputs,train_flag=True):

      #(m,H,W,K*K*C)
      feat_patch = tf.image.extract_patches(inputs,sizes=[1,self.kernel_size,self.kernel_size,1],strides=[1,self.strides,self.strides,1],rates=[1,self.rates,self.rates,1],padding="SAME")

      #get kernel

      #phase 1 (m,H,W,filter)
      CBL_KG_p1 = self.CBL_KG_p1(feat_patch,train_flag)

      #phase 2 (m,H,W,K*K*G)
      CBL_KG_p2 = self.CBL_KG_p2(CBL_KG_p1)

      #reshape kernel (m,H,W,K*K,1,G)
      CBL_KG_p2 = self.kernel_reshape(CBL_KG_p2)
      

      #reshape input to (m,H,W,K*K,C//G,G) [G:group_number]
      feat_patch = self.input_reshape(feat_patch)

      #multiply (m,H,W,K*K,C//G,G)
      feat_out = self.MultLayer([feat_patch,CBL_KG_p2])

      #add bias
      V_bias = tf.cast(self.V_bias,tf.keras.backend.dtype(feat_out))
      feat_out = feat_out + V_bias

      #sum over kernel dim from (m,H,W,K*K,C//G,G) to (m,H,W,C//G,G)
      feat_out = tf.math.reduce_sum(feat_out,axis=3)

      #reshape output (m,H,W,C)
      feat_out = self.output_reshape(feat_out)

      #BN_out
      feat_out = self.BN_out(feat_out,training=train_flag)

      #activate
      feat_out = self.act_out(feat_out)
      
      return feat_out


      
