import tensorflow as tf
from CBL import CBL
from RES_UNIT import res_unit

class MultiViewREX(tf.keras.Model):

   def __init__(self,MVREX_info,**kwargs):

      """
      block_info -- dictionary containing blocks' hyperparameters : CBL_1 , CBL_2, CBL_X , res_unit 

      Module Graph:

      --------CBL_1 ------ res_unit --- res_unit---- res_unit --- res_unit------ ...... --------------
                |              |             |          |            |                               |
                |              |             |          |____________|__________ concat -----CBL_X---|      
                |              |             |                                                       |  
                |              -------------------concat-----CBL_X------------------------------------
                |                                                                                    |
                |                                                                                    |
                |____________________________________________________________________________________|__________ concat --- CBL_2
      """

      #initialization
      super(MultiViewREX,self).__init__(**kwargs)

      #res unit
      self.num_res_units = MVREX_info["num_res_units"]
      
      #1st CBL block
      filters,kernel_size,strides,padding = MVREX_info["CBL_1"]
      
      self.CBL_1 = CBL(filters,kernel_size,strides,padding)

      #2nd CBL block
      filters,kernel_size,strides,padding = MVREX_info["CBL_2"]

      self.CBL_2 = CBL(filters,kernel_size,strides,padding)

      #res_units (**index start from 1)
      self.res_unit_seq = {}

      for i in range(1,self.num_res_units+1):

         res_units_info = MVREX_info["res_unit"]
         self.res_unit_seq[f"res_unit_{i}"] = res_unit(res_units_info)

         if (i%2) == 0:
            
            filters,kernel_size,strides,padding = MVREX_info["CBL_X"]
            
            self.res_unit_seq[f"CBL_X_{i}"] = CBL(filters,kernel_size,strides,padding)


   def call(self,inputs,train_flag=True):

      x = inputs

      #1st CBL block
      CBL_1 = self.CBL_1(inputs,train_flag)

      #res_unit
      final_concat = []
      final_concat.append(CBL_1)
      
      mid_concat = []
      res_unit_X = CBL_1

      for i in range(1,self.num_res_units+1):

         res_unit_X = (self.res_unit_seq[f"res_unit_{i}"])(res_unit_X,train_flag)

         mid_concat.append(res_unit_X)

         if (i%2) == 0:

            #mid concat
            mid_concat_X = tf.keras.layers.concatenate(inputs=mid_concat,axis=-1)

            #CBL_X
            CBL_X = (self.res_unit_seq[f"CBL_X_{i}"])(mid_concat_X,train_flag)

            #concat CBL_X to final concat
            final_concat.append(CBL_X)

            #refresh mid_concat list
            mid_concat = []

      if self.num_res_units == 1:

         final_concat.append(res_unit_X)
         
      #final concat
      final_concat_layer = tf.keras.layers.concatenate(inputs=final_concat,axis=-1)

      #2nd CBL block
      CBL_2 = self.CBL_2(final_concat_layer,train_flag)

      return CBL_2

   def graph_model(self,dim):

      x = tf.keras.layers.Input(shape=dim)
      
      return tf.keras.Model(inputs=x,outputs=self.call(x))
