import tensorflow as tf


class ResMLP(tf.keras.Model):

   def __init__(self,units,**kwargs):

      """
      ResMLP_info -- dictionary containing information: units

                     
      Module Graph:
      
      ---------- MLP ------ MLP ------ Add
         |                              |
         |                              |
         |                              |
         --------------------------------
         
      """

      super(ResMLP,self).__init__(**kwargs)

      self.MLP_1 = tf.keras.layers.Dense(units)

      self.norm_1 = tf.keras.layers.LayerNormalization(axis=-1)

      self.act_1 = tf.keras.layers.LeakyReLU()
      

      self.MLP_2 = tf.keras.layers.Dense(units)

      self.norm_2 = tf.keras.layers.LayerNormalization(axis=-1)
      
      self.act_2 = tf.keras.layers.LeakyReLU()

      self.Add_1 = tf.keras.layers.Add()


   def call(self,inputs,train_flag=True):

     #MLP_1 -- (m,units)
     MLP_1 = self.MLP_1(inputs)

     norm_1 = self.norm_1(MLP_1,training=train_flag)

     act_1 = self.act_1(norm_1)

     #MLP_2 -- (m,units)
     MLP_2 = self.MLP_2(act_1)

     norm_2 = self.norm_2(MLP_2,training=train_flag)

     act_2 = self.act_2(norm_2)

     Add_1 = self.Add_1([inputs,act_2])

     return Add_1
   
   def graph_model(self,dim):

      #x = tf.constant(np.random.randn(dim[0],dim[1],dim[2],dim[3]))
      x = tf.keras.layers.Input(shape=dim)
      
      return tf.keras.Model(inputs=x,outputs=self.call(x))
      
