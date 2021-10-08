import tensorflow as tf
from CBL import CBL

class res_unit(tf.keras.Model):

   def __init__(self,block_info,**kwargs):

      """
      block_info -- dictionary containing blocks' hyperparameters (filters,kernel_size,strides,padding)

      Module Graph:

      ------ CBL_1 ------ CBL_2 ------ Add
         |                              |
         |                              |
         |                              |
         --------------------------------

      
      """

      #initialization
      super(res_unit,self).__init__(**kwargs)
      
      #1st CBL block
      filters,kernel_size,strides,padding = block_info["CBL_1"]
      
      self.CBL_1 = CBL(filters,kernel_size,strides,padding)

      #2nd CBL block
      filters,kernel_size,strides,padding = block_info["CBL_2"]

      self.CBL_2 = CBL(filters,kernel_size,strides,padding)

      #Add Layer
      self.Add_layer = tf.keras.layers.Add()


   def call(self,inputs,train_flag=True):

      x = inputs

      #1st CBL block
      CBL_1 = self.CBL_1(inputs,train_flag)

      #2nd CBL block
      CBL_2 = self.CBL_2(CBL_1,train_flag)

      #Add Layer
      output_shortcut = self.Add_layer([CBL_2,x])

      return output_shortcut
