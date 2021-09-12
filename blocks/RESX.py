import tensorflow as tf
from RES_UNIT import res_unit

class RESX(tf.keras.Model):

   def __init__(self,RESX_info,**kwargs):

      """
      block_info -- dictionary containing : number_of_res_unit,CBL_1,res_unit_i (**i start from 1)

      Module Graph:

      ------ CBL_1 ------ res_unit * X

      
      """

      super(RESX,self).__init__(**kwargs)

      #CBL_1
      filters,kernel_size,strides,padding = RESX_info["CBL_1"]
      
      self.CBL_1 = CBL(filters,kernel_size,strides,padding)

      #number of res_unit
      self.num_of_res_unit = RESX_info["num_of_res_unit"]

      #res unit
      self.res_unit_seq = {}
      
      for i in range(1,self.num_of_res_unit+1):

         #get res unit i info
         res_unit_info = RESX_info[f"res_unit_{i}"]

         #def res unit i
         self.res_unit_seq[f"res_unit_{i}"] = res_unit(res_unit_info)


   def call(self,inputs,train_flag=True):

      #CBL1
      CBL_1 = self.CBL_1(inputs,train_flag)

      #res_unit
      res_unit_i = CBL_1

      for i in range(1,self.num_of_res_unit+1):

         res_unit_i = self.res_unit_seq[f"res_unit_{i}"](res_unit_i,train_flag)


      return res_unit_i
