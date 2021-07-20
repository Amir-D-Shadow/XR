import tensorflow as tf
from CBM import CBM
from CBL import CBL
from RES_UNIT import res_unit

class CSPX(tf.keras.Model):

   def __init__(self,CSPX_info,**kwargs):

      """
      CSPX_info -- dictionary containing information: num_of_res_unit , res_unit block info , CBM block info , CBL_info

                     - hpara: (filters,kernel_size,strides,padding)

                     
      Module Graph:
      
      ------ CBL_1 ------ CBL_2 ------ res_unit * X ------ CBL_3 -----
                     |                                               |
                     |                                               |______
                     |                                                ______  Concat --- BN --- leaky relu --- CBM_1 
                     |                                               |
                     |                                               |
                     -------------------------------- CBL_4 ----------
      """

      #initialization
      super(CSPX,self).__init__(**kwargs)

      #extract num_of_res_unit
      self.num_of_res_unit = CSPX_info["num_of_res_unit"]

      #define layers

      #res_unit
      self.res_unit_branch = {}

      #Important: When defining the CSPX layer, remember to define res unit info (dictionary key) in the form of res_unit_i : i start from 1
      for i in range(1,self.num_of_res_unit+1):

         #Extract res_unit_i info
         res_unit_info = CSPX_info[f"res_unit_{i}"]

         #define resunit layer
         self.res_unit_branch[f"res_unit_{i}"] = res_unit(res_unit_info)
         

      #CBL_1
      filters,kernel_size,strides,padding = CSPX_info["CBL_1"]
      
      self.CBL_1 = CBL(filters,kernel_size,strides,padding)

      #CBL_2
      filters,kernel_size,strides,padding = CSPX_info["CBL_2"]
      
      self.CBL_2 = CBL(filters,kernel_size,strides,padding)

      #CBL_3
      filters,kernel_size,strides,padding = CSPX_info["CBL_3"]
      
      self.CBL_3 = CBL(filters,kernel_size,strides,padding)

      #CBL_4
      filters,kernel_size,strides,padding = CSPX_info["CBL_4"]
      
      self.CBL_4 = CBL(filters,kernel_size,strides,padding)

      #BN
      self.BN_x = tf.keras.layers.BatchNormalization(axis=3)

      #leaky relu
      self.leaky_relu_x = tf.keras.layers.LeakyReLU()

      #CBM_1
      filters,kernel_size,strides,padding = CSPX_info["CBM_1"]

      self.CBM_1 = CBM(filters,kernel_size,strides,padding)
      

   def call(self,inputs):

      x = inputs

      #CBL_1
      CBL_1 = self.CBL_1(x)

      #CBL_2
      CBL_2 = self.CBL_2(CBL_1)

      #res_unit block
      res_unit_block = CBL_2
      
      for i in range(1,self.num_of_res_unit+1):

         res_unit_block =  (self.res_unit_branch[f"res_unit_{i}"])(res_unit_block) 

      #CBL3
      CBL_3 = self.CBL_3(res_unit_block)
      
      #CBL_4
      CBL_4 = self.CBL_4(CBL_1)

      #Concat
      mid_concat = tf.keras.layers.concatenate(inputs=[CBL_3,CBL_4],axis=3)

      #Batch Normalization
      BN_x = self.BN_x(mid_concat)

      #leaky_relu_x
      leaky_relu_x = self.leaky_relu_x(BN_x)

      #output_CBM
      output_CBM = self.CBM_1(leaky_relu_x)

      return output_CBM
      
