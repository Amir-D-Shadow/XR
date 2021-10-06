import tensorflow as tf
from CBL import CBL



class ChainLayer(tf.keras.layers.Layer):

    def __init__(self,chain_info,**kwargs):

        """

        chain_info -- dictionary containing information: CBL_Q ,CBL_O,DSC

                     
        Module Graph:

        inputs ----------CBL_Q -------  similarity process: CBL_K --- DSC


        """

        super(ChainLayer,self).__init__(**kwargs)

        #CBL_Q
        filters,kernel_size,strides,padding = chain_info["CBL_Q"]
        
        self.CBL_Q = CBL(filters,kernel_size,strides,padding)
        

        #DSC BN act
        filters,kernel_size,strides,padding = chain_info["DSC"]
        
        self.DSC = tf.keras.layers.SeparableConv2D(filters=filters,kernel_size=kernel_size,strides=strides,padding=padding,data_format="channels_last")#CBL(filters,kernel_size,strides,padding)

        self.BN = tf.keras.layers.BatchNormalization(axis=-1)

        self.act = tf.keras.layers.LeakyReLU(alpha=0.03)

        

    def call(self,inputs,train_flag=True):

        """
        inputs : (m,h,w,c)
        V_bias: (1,1,1,h x w) 
        """

        #CBL_Q
        CBL_Q = self.CBL_Q(inputs,train_flag)

        #similarity CBL_K
        CBL_K = self.CBL_K(CBL_Q,train_flag)

        #output_M: (m,h,w,c)
        DSC = self.DSC(CBL_K)
        
        output_M = self.BN(DSC,training=train_flag)

        output_M = self.act(output_M)
        

        return output_M
