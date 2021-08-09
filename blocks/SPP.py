import tensorflow as tf

class SPP(tf.keras.Model):

   def __init__(self,**kwargs):

      #initialization
      super(SPP,self).__init__(**kwargs)

      #define layers
      self.maxpool_5x5 = tf.keras.layers.MaxPooling2D(pool_size=5,strides=1,padding="same",data_format="channels_last")

      self.maxpool_9x9 = tf.keras.layers.MaxPooling2D(pool_size=9,strides=1,padding="same",data_format="channels_last")

      self.maxpool_13x13 = tf.keras.layers.MaxPooling2D(pool_size=13,strides=1,padding="same",data_format="channels_last")


   def call(self,inputs):

      """
      input -- tensorflow layer with shape (m,n_H,n_W,n_C)
      """
      
      #5x5
      maxpool_5x5 = self.maxpool_5x5(inputs)

      #9x9
      maxpool_9x9 = self.maxpool_9x9(inputs)

      #13x13
      maxpool_13x13 = self.maxpool_13x13(inputs)

      #concatenate
      output_concat = tf.keras.layers.concatenate(inputs=[maxpool_5x5,maxpool_9x9,maxpool_13x13,inputs],axis=-1)

      return output_concat


if __name__ == "__main__":

   a = tf.keras.layers.MaxPooling2D(5,1)
