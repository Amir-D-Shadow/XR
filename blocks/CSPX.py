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
      
      ------ CBL_1 ------ CBL_2 ------ [res_unit] ------ CBL_3 -------
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
      self.res_units_branch = {}

      #Important: When defining the CSPX layer, remember to define res unit info in the form of res_unit_i : i start from 1
      for i in range(1,num_of_res_units+1):

         #Extract res_unit_i info
         res_unit_info = CSPX_info[f"res_unit_{i}"]

         #define resunit layer
         self.res_units_branch[f"res_unit_{i}"] = res_unit(res_unit_info)
         

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
      self.leaky_relu_x = tf.keras.layers.LeakyReLu()

      #CBM_1
      filters,kernel_size,strides,padding = CSPX_info["CBL_1"]

      self.CBM_1 = CBM(filters,kernel_size,strides,padding)
      

   def call(self,inputs):

      pass
