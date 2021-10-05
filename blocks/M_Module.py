import tensorflow as tf
import MagicSquareLayer

class M_Module(tf.keras.Model):

    def __init__(self,M_info,**kwargs):

        super(M_Module,self).__init__(**kwargs)

        self.feat_scale = M_info["feat_scale"]

        self.feat_alpha = M_info["feat_alpha"]

        self.MSL = MagicSquareLayer(feat_alpha = self.feat_alpha,feat_scale = self.feat_scale)

        self.BN = tf.keras.layers.BatchNormalization(axis=-1)

        self.LR = tf.keras.layers.LeakyReLU()


    def call(self,inputs,train_flag=True):

        MSL = self.MSL(inputs)

        BN = self.BN(MSL,training=train_flag)

        output_LR = self.LR(BN)

        return output_LR


    def graph_model(self,dim):

        x = tf.keras.layers.Input(shape=dim)
        
        return tf.keras.Model(inputs=x,outputs=self.call(x))
