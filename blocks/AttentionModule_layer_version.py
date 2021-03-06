import tensorflow as tf
import tensorflow.keras.backend as K
import numpy  as np
from CBL import CBL

class attention_step_process_layer(tf.keras.layers.Layer):

    """
    ** @ -- multiply and sum over

    process:

    query -----------------
                          | ---- @ ------- softmax---
    keys ------------------                          |------------------ output
                                                     |
    values -------------------------------------------

    """

   def __init__(self):

      super(attention_step_process_layer,self).__init__()

   def call(self,query,keys,values):

      """
      query -- (m,h,w,c)
      keys -- (m,h,w,c)
      values -- (m,h,w,c)
      """

      #get batch_size
      m = query.shape[0]

      #get number of channel
      num_of_channels = query.shape[-1]

      #set up
      output_tensor = tf.TensorArray(K.dtype(query),size=1,dynamic_size=True)

      i = 0

      #loop via batch size
      while i < m:

        #get feat query -- (h,w,c)
        feat_query = query[i,:,:,:]

        #get feat keys -- (h,w,c)
        feat_keys = keys[i,:,:,:]

        #get feat values -- (h,w,c)
        feat_values = values[i,:,:,:]

        #reshape feat_keys to (h x w,c)
        feat_keys = tf.reshape(feat_keys,shape=(-1,num_of_channels))

        #fine tune to (1 , h x w , c)
        feat_keys = feat_keys[tf.newaxis,:,:]

        #fine tune query to (h,w,1,c)
        feat_query = feat_query[:,:,tf.newaxis,:]

        #Globally Multiply feat_query with feat keys  -- (h,w, h x w , c)
        first_phase_output = feat_query * feat_keys

        #sum over -- (h,w,c)
        first_phase_output = K.sum(first_phase_output,axis=-2,keepdims=False)
        
        #softmax activate -- phase 2 -- (h,w,c)
        second_phase_attentions = tf.keras.layers.Softmax(axis = [0,1,2])(first_phase_output)

        #Multiply feat_values with second_phase_attentions -- (h,w,c)
        second_phase_output = feat_values * second_phase_attentions

        #save
        output_tensor = output_tensor.write(i,second_phase_output)

        #update i
        i = i + 1
         
      #stack to tensor
      output_tensor = output_tensor.stack()

      return output_tensor


class AttentionModule(tf.keras.Model):

   def __init__(self,attention_info,**kwargs):

      """

      attention_info -- dictionary containing information: CBL_1 ,conv_query ,conv_keys ,conv_values

                     
      Module Graph:
                ___________________________________________________________________
               |                                                                  |
               |     ----- conv_query ------                                      |
               |     |                      |                                     |
               |     |                      |______                               |
      CBL1 ----------|---- conv_keys ------- ______  attention_step_process_1 --- Add --- LN --- leaky relu 
                     |                      |
                     |                      |
                     ----- conv_values -----
      
      """

      super(AttentionModule,self).__init__(**kwargs)

      #CBL_1
      filters,kernel_size,strides,padding = attention_info["CBL_1"]
      
      self.CBL_1 = CBL(filters,kernel_size,strides,padding)

      #query
      filters,kernel_size,strides,padding = attention_info["conv_query"]
      
      self.conv_query = tf.keras.layers.Conv2D(filters=filters,kernel_size=kernel_size,strides=strides,padding=padding,data_format="channels_last")
      
      #keys
      filters,kernel_size,strides,padding = attention_info["conv_keys"]
      
      self.conv_keys = tf.keras.layers.Conv2D(filters=filters,kernel_size=kernel_size,strides=strides,padding=padding,data_format="channels_last")
      
      #values
      filters,kernel_size,strides,padding = attention_info["conv_values"]
      
      self.conv_values = tf.keras.layers.Conv2D(filters=filters,kernel_size=kernel_size,strides=strides,padding=padding,data_format="channels_last")
      
      #step process
      self.attention_step_process_1 = attention_step_process_layer()

      #Add
      self.Add_layer = tf.keras.layers.Add()

      #LN_1
      self.LN_1 = tf.keras.layers.LayerNormalization(axis=[1,2,3])

      #leakyRelu
      self.output_leakyrelu = tf.keras.layers.LeakyReLU()

   def call(self,inputs,train_flag=True):

      #CBL_1
      CBL_1 = self.CBL_1(inputs,train_flag)

      #query
      conv_query = self.conv_query(CBL_1)

      #keys
      conv_keys = self.conv_keys(CBL_1)

      #values
      conv_values = self.conv_values(CBL_1)

      #step process
      attention_step_process_1 = self.attention_step_process_1(conv_query,conv_keys,conv_values)

      #Add
      Add_layer = self.Add_layer([CBL_1,attention_step_process_1])

      #LN_1
      LN_1 = self.LN_1(attention_step_process_1,training=train_flag)

      #output_leakyrelu
      output_leakyrelu = self.output_leakyrelu(LN_1)

      return output_leakyrelu

      
