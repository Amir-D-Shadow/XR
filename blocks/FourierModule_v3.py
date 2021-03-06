import tensorflow as tf
from tensorflow.python.keras import backend as K
import numpy as np
from ResMLP import ResMLP
import CBL as CBL

      
class FourierModule(tf.keras.Model):


   def __init__(self,F_info,**kwargs):

      """

      attention_info -- dictionary containing information: units,MLP_num,output_units

                     
      Module Graph:
      

                                                                                                                                  
      inputs --------- FFT -- ResMLP * n ------- filtering ----  iFFT 


      
      """

      super(FourierModule,self).__init__(**kwargs)

      self.units = F_info["units"]
      self.MLP_num = F_info["MLP_num"]


      self.input_MLP = tf.keras.layers.Dense(self.units)
      self.input_norm = tf.keras.layers.LayerNormalization(axis=-1)
      self.input_act = tf.keras.layers.LeakyReLU()
      self.input_drop = tf.keras.layers.Dropout(rate=0.5)

      self.ResMLP_block = {}

      for i in range(1,self.MLP_num+1):

         self.ResMLP_block[f"ResMLP_{i}"] = ResMLP(self.units)

      self.output_units = F_info["output_units"]
      
      self.output_MLP = tf.keras.layers.Dense(self.output_units)
      self.output_act = tf.keras.layers.LeakyReLU()

      self.CBL_output = CBL(64,1,1,"same")


   def call(self,inputs,train_flag=True):

      #FFT
      feat_FT = tf.signal.fft3d(K.cast(inputs,tf.complex128))

      #real
      feat_FT_real = tf.math.real( feat_FT )

      #imag
      feat_FT_imag = tf.math.imag( feat_FT )
      feat_FT_imag = K.cast(feat_FT_imag,K.dtype(feat_FT_real))

      #set up
      feat_analysis_inputs = tf.TensorArray(tf.float64,size=1,dynamic_size=True)

      #mean 
      feat_mean = tf.math.reduce_mean(feat_FT_real,axis=(1,2,3))
      feat_analysis_inputs = feat_analysis_inputs.write(0,feat_mean)

      #var
      feat_var = tf.math.sqrt( tf.math.reduce_variance(feat_FT_real,axis=(1,2,3)) )
      feat_analysis_inputs = feat_analysis_inputs.write(1,feat_var)

      #percentile
      for i in range(1,20):

       val = tfp.stats.percentile(feat_FT_real,i*5,axis=(1,2,3),interpolation="linear")
       feat_analysis_inputs = feat_analysis_inputs.write(i+1,val)


      #stack tensor -- (21,m)
      feat_analysis_inputs = feat_analysis_inputs.stack()

      #reshape -- (m,21)
      feat_analysis_inputs = tf.reshape(feat_analysis_inputs,shape=(-1,21))

      ################# ResMLP Filtering #################

      input_MLP = self.input_MLP(feat_analysis_inputs)
      input_norm = self.input_norm(input_MLP,training=train_flag)
      input_act = self.input_act(input_norm)
      input_drop = self.input_drop(input_act)

      ResMLP_block = input_drop#input_act
      
      for i in range(1,self.MLP_num+1):

         ResMLP_block = (self.ResMLP_block[f"ResMLP_{i}"])(ResMLP_block,train_flag)
         

      output_MLP = self.output_MLP(ResMLP_block)
      output_act = self.output_act(output_MLP)

      ################# ResMLP Filtering #################

      # Tensor output_act [TU1,TL1,TU2,TL2,TU3,TL3,TU4,TL4]
      final_list = []

      for i in range(self.output_units//2):

         #first interval
         TU = output_act[:,0]
         TU = TU[:,tf.newaxis,tf.newaxis,tf.newaxis]
         TU = K.cast(TU,K.dtype(feat_FT_real))
         TU_filtering = feat_FT_real * K.maximum( TU -feat_FT_real ,0 ) / (TU -feat_FT_real + 1)

         TL = output_act[:,1]
         TL = TL[:,tf.newaxis,tf.newaxis,tf.newaxis]
         TL = K.cast(TL,K.dtype(feat_FT_real))
         TL_filtering = TU_filtering * K.maximum( TU_filtering - TL, 0 )  / (TU_filtering - TL + 1)
         
         TL_filtering = tf.complex(TL_filtering,feat_FT_imag)
         TL_filtering = K.cast(TL_filtering,K.dtype(feat_FT))

         feat_intv1 = tf.signal.ifft3d( TL_filtering )
         feat_intv1 = tf.math.real( feat_intv1 )
         feat_intv1 = K.cast(feat_intv1,K.dtype(inputs))

         final_list.append(feat_intv1)



      #final result
      final_tensor = tf.keras.layers.concatenate(final_list,axis=-1)
      final_tensor = self.CBL_output(final_tensor,train_flag)

      final_tensor = K.cast(final_tensor,K.dtype(inputs))

      return final_tensor
      

   def graph_model(self,dim):

      x = tf.keras.layers.Input(shape=dim)
      
      return tf.keras.Model(inputs=x,outputs=self.call(x))

      
