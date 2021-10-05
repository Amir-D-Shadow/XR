import tensorflow as tf
import MagicSquareLayer

class M_Module(tf.keras.Model):

    def __init__(self,M_info,**kwargs):

        super(M_Module,self).__init__(**kwargs)

        self.num_C = M_info["num_C"]

        self.MSL = MagicSquareLayer(self.num_C)

        self.BN = tf.keras.layers.BatchNormalization(axis=-1)

        self.LR = tf.keras.layers.LeakyReLU()


    def call(self,inputs,train_flag=True):

        MSL = self.MSL(inputs)

        BN = self.BN(MSL,training=train_flag)

        output_LR = self.LR(BN)

        return output_LR
