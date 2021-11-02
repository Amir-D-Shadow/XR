import tensorflow as tf
from CBL import CBL
from inv_res_unit_parallel import inv_res_unit_parallel




class iCSPX_parallel(tf.keras.Model):

   def __init__(self,iCSPX_parallel_info,**kwargs):

      """
      CSPX_info -- dictionary containing information: num_of_res_unit , res_unit block info , CBM block info , CBL_info

                     - hpara: (filters,kernel_size,strides,padding)

                     
      Module Graph:
      
      ------ CBL_1 ------ CBL_2 ------ inv_res_unit_parallel * X ------ CBL_3 -----
                     |                                                             |
                     |                                                             |______
                     |                                                              ______  Concat --- CBL_5
                     |                                                             |
                     |                                                             |
                     --------------------------------------------- CBL_4 ----------
      """

      #initialization
      super(iCSPX_parallel,self).__init__(**kwargs)

      #extract num_of_res_unit
      self.num_of_inv_res_unit_parallel = iCSPX_parallel_info["num_of_inv_res_unit_parallel"]

      #define layers

      #res_unit
      self.res_unit_seq = {}

      #Important: When defining the iCSPX layer, remember to define res unit info (dictionary key) in the form of res_unit_i : i start from 1
      for i in range(1,self.num_of_inv_res_unit_parallel+1):

         #Extract res_unit_i info
         res_unit_info = iCSPX_parallel_info[f"inv_res_unit_parallel_info"]

         #define resunit layer
         self.res_unit_seq[f"res_unit_{i}"] = inv_res_unit_parallel(res_unit_info)
         

      #CBL_1
      filters,kernel_size,strides,padding = iCSPX_parallel_info["CBL_1"]
      
      self.CBL_1 = CBL(filters,kernel_size,strides,padding)

      #CBL_2
      filters,kernel_size,strides,padding = iCSPX_parallel_info["CBL_2"]
      
      self.CBL_2 = CBL(filters,kernel_size,strides,padding)

      #CBL_3
      filters,kernel_size,strides,padding = iCSPX_parallel_info["CBL_3"]
      
      self.CBL_3 = CBL(filters,kernel_size,strides,padding)

      #CBL_4
      filters,kernel_size,strides,padding = iCSPX_parallel_info["CBL_4"]
      
      self.CBL_4 = CBL(filters,kernel_size,strides,padding)

      #CBL_5
      filters,kernel_size,strides,padding = iCSPX_parallel_info["CBL_5"]

      self.CBL_5 = CBL(filters,kernel_size,strides,padding)

   def call(self,inputs,train_flag=True):

      """
      input -- tensorflow layer with shape (m,n_H,n_W,n_C)
      """

      x = inputs

      #CBL_1
      CBL_1 = self.CBL_1(x,train_flag)

      #CBL_2
      CBL_2 = self.CBL_2(CBL_1,train_flag)

      #res_unit block
      res_unit_block = CBL_2
      
      for i in range(1,self.num_of_inv_res_unit_parallel+1):

         res_unit_block =  (self.res_unit_seq[f"res_unit_{i}"])(res_unit_block,train_flag) 

      #CBL3
      CBL_3 = self.CBL_3(res_unit_block,train_flag)
      
      #CBL_4
      CBL_4 = self.CBL_4(CBL_1,train_flag)

      #Concat
      mid_concat = tf.keras.layers.concatenate(inputs=[CBL_3,CBL_4],axis=-1)

      #CBL5
      CBL_5 = self.CBL_5(mid_concat,train_flag)

      return CBL_5
