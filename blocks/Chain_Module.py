import tensorflow as tf
import ChainLayer

class Chain_Module(tf.keras.Model):

    def __init__(self,C_info,**kwargs):

        super(Chain_Module,self).__init__(**kwargs)

        self.feat_scale = C_info["feat_scale"]

        self.feat_alpha = C_info["feat_alpha"]

        self.MSL = ChainLayer(feat_alpha = self.feat_alpha,feat_scale = self.feat_scale)

        self.BN = tf.keras.layers.BatchNormalization(axis=-1)

        self.LR = tf.keras.layers.LeakyReLU()


    def call(self,inputs,train_flag=True):

        MSL = self.MSL(inputs,train_flag)

        BN = self.BN(MSL,training=train_flag)

        output_LR = self.LR(BN)

        return output_LR


    def graph_model(self,dim):

        x = tf.keras.layers.Input(shape=dim)
        
        return tf.keras.Model(inputs=x,outputs=self.call(x))
