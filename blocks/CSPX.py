import tensorflow as tf
from CBM import CBM
from CBL import CBL
from RES_UNIT import res_unit

#Backbone CSPX
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
      self.res_unit_seq = {}

      #Important: When defining the CSPX layer, remember to define res unit info (dictionary key) in the form of res_unit_i : i start from 1
      for i in range(1,self.num_of_res_unit+1):

         #Extract res_unit_i info
         res_unit_info = CSPX_info[f"res_unit_{i}"]

         #define resunit layer
         self.res_unit_seq[f"res_unit_{i}"] = res_unit(res_unit_info)
         

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

      """
      input -- tensorflow layer with shape (m,n_H,n_W,n_C)
      """

      x = inputs

      #CBL_1
      CBL_1 = self.CBL_1(x)

      #CBL_2
      CBL_2 = self.CBL_2(CBL_1)

      #res_unit block
      res_unit_block = CBL_2
      
      for i in range(1,self.num_of_res_unit+1):

         res_unit_block =  (self.res_unit_seq[f"res_unit_{i}"])(res_unit_block) 

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
      



#Neck CSPX
class CSPX_Neck(tf.keras.Model):

   def __init__(self,NECK_info,**kwargs):
      
      """
      NECK_info -- dictionary containing information: num_of_CBL, CBM block info , CBL block info , conv2D info

                     - hpara: (filters,kernel_size,strides,padding)

                     
      Module Graph:
      
      ----------- CBL * X ------ conv2D_1 ----------------
         |                                               |
         |                                               |______
         |                                                ______  Concat --- BN --- leaky relu --- CBM_1 
         |                                               |
         |                                               |
         -------------------------------- conv2D_2 -------
         
      """
      
      #initialization
      super(CSPX_Neck,self).__init__(**kwargs)

      #Get num_of_CBL
      self.num_of_CBL = NECK_info["num_of_CBL"]

      #define layers

      #CBL_X
      self.CBL_seq = {}

      for i in range(1,self.num_of_CBL+1):

         filters,kernel_size,strides,padding = NECK_info[f"CBL_{i}"]

         self.CBL_seq[f"CBL_{i}"] = CBL(filters,kernel_size,strides,padding)


      #Conv2D_1
      filters,kernel_size,strides,padding = NECK_info["conv2D_1"]
      
      self.conv2D_1 = tf.keras.layers.Conv2D(filters=filters,kernel_size=kernel_size,strides=strides,padding=padding)

      #conv2D_2
      filters,kernel_size,strides,padding = NECK_info["conv2D_2"]

      self.conv2D_2 = tf.keras.layers.Conv2D(filters=filters,kernel_size=kernel_size,strides=strides,padding=padding)

      #Batch Normalization
      self.BN_x = tf.keras.layers.BatchNormalization(axis=3)
      
      #leaky relu
      self.leaky_relu_x = tf.keras.layers.LeakyReLU()

      #CBM_1
      filters,kernel_size,strides,padding = NECK_info["CBM_1"]

      self.CBM_1 = CBM(filters,kernel_size,strides,padding)


   def call(self,inputs):

      """
      input -- tensorflow layer with shape (m,n_H,n_W,n_C)
      """

      #CBL_X
      CBL_block = inputs
      
      for i in range(1,self.num_of_CBL+1):

         CBL_block = (self.CBL_seq[f"CBL_{i}"])(CBL_block)


      #conv2D_1
      conv2D_1 = self.conv2D_1(CBL_block)

      #conv2D_2
      conv2D_2 = self.conv2D_2(inputs)

      #concat
      mid_concat = tf.keras.layers.concatenate(inputs=[conv2D_1,conv2D_2],axis=3)

      #BN_x
      BN_x = self.BN_x(mid_concat)

      #leaky relu
      leaky_relu_x = self.leaky_relu_x(BN_x)

      #CBM_1
      output_CBM = self.CBM_1(leaky_relu_x)

      return output_CBM







