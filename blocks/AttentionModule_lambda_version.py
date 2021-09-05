import tensorflow as tf
import tensorflow.keras.backend as K
import numpy  as np
from CBL import CBL



def attention_step_process_layer(input_list):

    """
    ** X -- multiply and sum over

    process:

    query -----------------
                          | ---- X ------- softmax---
    keys ----- softmax ----                          |------------------ output
                                                    |
    values -------------------------------------------

    """
    query = input_list[0] 
    keys = input_list[1] 
    values = input_list[2] 

    """
    query -- (m,h,w,c)
    keys -- (m,h,w,c)
    values -- (m,h,w,c)
    """

    #get batch_size
    m = query.shape[0]

    if m is None:

       m = 10

    #get number of channel
    num_of_channels = query.shape[-1]

    #set up
    output_tensor = tf.TensorArray(K.dtype(query),size=1,dynamic_size=True)

    i = 0

    #loop via batch size
    while i < m:

        #get feat query -- (h,w,c)
        feat_query = query[i,:,:,:]

        #get feat keys -- (h,w,c)
        feat_keys = keys[i,:,:,:]

        #get feat values -- (h,w,c)
        feat_values = values[i,:,:,:]

        #reshape feat_keys to (h x w,c)
        feat_keys = tf.reshape(feat_keys,shape=(-1,num_of_channels))

        #fine tune to (1 , h x w , c)
        feat_keys = feat_keys[tf.newaxis,:,:]

        #softmax activate -- phase 1
        first_phase_attentions = tf.keras.layers.Softmax(axis = 1)(feat_keys)

        #fine tune query to (h,w,1,c)
        feat_query = feat_query[:,:,tf.newaxis,:]

        #Multiply feat_query with first_phase_attentions -- (h,w, h x w , c)
        first_phase_output = feat_query * first_phase_attentions

        #sum over -- (h,w,c)
        first_phase_output = K.sum(first_phase_output,axis=-2,keepdims=False)

        #reshape -- (h x w, c )
        first_phase_output = tf.reshape(first_phase_output,shape=(-1,num_of_channels))

        #fine tune first_phase_output to (1,h x w,c)
        first_phase_output = first_phase_output[tf.newaxis,:,:]
        
        #softmax activate -- phase 2
        second_phase_attentions = tf.keras.layers.Softmax(axis = 1)(first_phase_output)

        #fine tune values to (h,w,1,c)
        feat_values = feat_values[:,:,tf.newaxis,:]

        #Multiply feat_values with second_phase_attentions -- (h,w, h x w , c)
        second_phase_output = feat_values * second_phase_attentions

        #sum over -- (h,w,c)
        second_phase_output = K.sum(second_phase_output,axis=-2,keepdims=False)

        #save
        output_tensor = output_tensor.write(i,second_phase_output)

        #update i
        i = i + 1
        
    #stack to tensor
    output_tensor = output_tensor.stack()

    return output_tensor


class AttentionModule(tf.keras.Model):

   def __init__(self,attention_info,**kwargs):

      """

      attention_info -- dictionary containing information: CBL_1 ,conv_query ,conv_keys ,conv_values

                     
      Module Graph:
                ___________________________________________________________________
               |                                                                  |
               |     ----- conv_query ------                                      |
               |     |                      |                                     |
               |     |                      |______                               |
      CBL1 ----------|---- conv_keys ------- ______  attention_step_process_1 --- Add --- LN --- leaky relu 
                     |                      |
                     |                      |
                     ----- conv_values -----
      
      """

      super(AttentionModule,self).__init__(**kwargs)

      #CBL_1
      filters,kernel_size,strides,padding = attention_info["CBL_1"]
      
      self.CBL_1 = CBL(filters,kernel_size,strides,padding)

      #query
      filters,kernel_size,strides,padding = attention_info["conv_query"]
      
      self.conv_query = tf.keras.layers.Conv2D(filters=filters,kernel_size=kernel_size,strides=strides,padding=padding,data_format="channels_last")
      
      #keys
      filters,kernel_size,strides,padding = attention_info["conv_keys"]
      
      self.conv_keys = tf.keras.layers.Conv2D(filters=filters,kernel_size=kernel_size,strides=strides,padding=padding,data_format="channels_last")
      
      #values
      filters,kernel_size,strides,padding = attention_info["conv_values"]
      
      self.conv_values = tf.keras.layers.Conv2D(filters=filters,kernel_size=kernel_size,strides=strides,padding=padding,data_format="channels_last")
      
      #step process
      self.attention_step_process_1 = tf.keras.layers.Lambda(attention_step_process_layer)

      #Add
      self.Add_layer = tf.keras.layers.Add()

      #LN_1
      self.LN_1 = tf.keras.layers.LayerNormalization(axis=[1,2,3])

      #leakyRelu
      self.output_leakyrelu = tf.keras.layers.LeakyReLU()

   def call(self,inputs,train_flag=True):

      #CBL_1
      CBL_1 = self.CBL_1(inputs,train_flag)

      #query
      conv_query = self.conv_query(CBL_1)

      #keys
      conv_keys = self.conv_keys(CBL_1)

      #values
      conv_values = self.conv_values(CBL_1)

      #step process
      attention_step_process_1 = self.attention_step_process_1([conv_query,conv_keys,conv_values])

      #Add
      Add_layer = self.Add_layer([CBL_1,attention_step_process_1])
      
      #LN_1
      LN_1 = self.LN_1(Add_layer,training=train_flag)

      #output_leakyrelu
      output_leakyrelu = self.output_leakyrelu(LN_1)

      return output_leakyrelu

   def graph_model(self,dim):

      #x = tf.constant(np.random.randn(dim[0],dim[1],dim[2],dim[3]))
      x = tf.keras.layers.Input(shape=dim)
      
      return tf.keras.Model(inputs=x,outputs=self.call(x))

if __name__ == "__main__":

   attention_info = {}
   attention_info["CBL_1"] = (20,3,1,"same")
   attention_info["conv_query"] = (20,3,1,"same")
   attention_info["conv_keys"] = (20,3,1,"same")
   attention_info["conv_values"] = (20,3,1,"same")

   y = AttentionModule(attention_info)
   model = y.graph_model(dim=(80,80,6))
   plot_model(model,show_shapes=True, show_layer_names=True)
