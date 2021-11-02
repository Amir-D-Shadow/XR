import tensorflow as tf
from CBL import CBL
from Involution import Involution

class inv_res_unit_series(tf.keras.Model):

   def __init__(self,block_info,**kwargs):

      """
      block_info -- dictionary containing blocks' hyperparameters (filters,kernel_size,strides,padding)

      Module Graph:

      ------ inv1 ------ CBL_2 ------ Add
         |                              |
         |                              |
         |                              |
         --------------------------------

      
      """

      #initialization
      super(inv_res_unit_series,self).__init__(**kwargs)
      
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
      inv1 = self.inv1(inputs,train_flag)

      #2nd CBL block
      CBL_2 = self.CBL_2(inv1,train_flag)

      #Add Layer
      output_shortcut = self.Add_layer([CBL_2,x])

      return output_shortcut
