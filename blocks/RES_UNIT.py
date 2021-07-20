import tensorflow as tf
from CBM import CBM

class res_unit(tf.keras.Model):

   def __init__(self,block_info,**kwargs):

      """
      block_info -- dictionary containing blocks' hyperparameters (filters,kernel_size,strides,padding)

      Module Graph:

      ------ CBM_1 ------ CBM_2 ------ Add
         |                              |
         |                              |
         |                              |
         --------------------------------

      
      """

      #initialization
      super(res_unit,self).__init__(**kwargs)
      
      #1st CBM block
      filters,kernel_size,strides,padding = block_info["CBM_1"]
      
      self.CBM_1 = CBM(filters,kernel_size,strides,padding)

      #2nd CBM block
      filters,kernel_size,strides,padding = block_info["CBM_2"]

      self.CBM_2 = CBM(filters,kernel_size,strides,padding)

      #Add Layer
      self.Add_layer = tf.keras.layers.Add()


   def call(self,inputs):

      x = inputs

      #1st CBM block
      CBM_1 = self.CBM_1(inputs)

      #2nd CBM block
      CBM_2 = self.CBM_2(CBM_1)

      #Add Layer
      output_shortcut = self.Add_layer([CBM_2,x])

      return output_shortcut
