import tensorflow as tf
from CBL import CBL
from Involution import Involution

class inv_res_unit_parallel(tf.keras.Model):

   def __init__(self,block_info,**kwargs):

      """
      block_info -- dictionary containing blocks' hyperparameters (filters,kernel_size,strides,padding)

      Module Graph:

                  _____ inv1 _____
                 |                |   
      ------------                 ---- CBL_2 ---- Add
         |       |_____ CBL_1 ____|                 |
         |                                          |
         |                                          |
         -------------------------------------------

      
      """

      #initialization
      super(inv_res_unit_parallel,self).__init__(**kwargs)
      
      #1st CBL block
      filters,kernel_size,strides,padding = block_info["CBL_1"]
      
      self.CBL_1 = CBL(filters,kernel_size,strides,padding)

      #inv1
      filters,kernel_size,strides,group_number = block_info["inv1"]

      self.inv1 = Involution(filters,kernel_size,strides,group_number)

      #2nd CBL block
      filters,kernel_size,strides,padding = block_info["CBL_2"]

      self.CBL_2 = CBL(filters,kernel_size,strides,padding)

      #Add Layer
      self.Add_layer = tf.keras.layers.Add()


   def call(self,inputs,train_flag=True):

      x = inputs

      #1st CBL block
      CBL_1 = self.CBL_1(inputs,train_flag)

      #inv1
      inv1 = self.inv1(inputs,train_flag)

      #concat
      concat1 = tf.keras.layers.concatenate(inputs=[inv1,CBL_1],axis=-1)

      #2nd CBL block
      CBL_2 = self.CBL_2(concat1,train_flag)

      #Add Layer
      output_shortcut = self.Add_layer([CBL_2,x])

      return output_shortcut
