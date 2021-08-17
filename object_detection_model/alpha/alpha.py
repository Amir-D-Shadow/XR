import tensorflow as tf
from CBL import CBL
from CBM import CBM
from CSPX import CSPX,rCSP
from TCBM import TCBM
from SPP import SPP
from TCBL import TCBL
from CBS import CBS


class alpha_model(tf.keras.Model):

   def __init__(self,**kwargs):

      #initialization
      super(alpha_model,self).__init__(**kwargs)

      #CBM_1 in : 640 x 640 x 3 out: 640 x 640 x 32
      filters=32
      kernel_size=3
      strides=1
      padding="same"
      
      self.CBM_1 = CBM(filters,kernel_size,strides,padding)

      #----------------------------------------------------------------

      #CSP1 in : 640 x 640 x 32 out : 319 x 319 x 64
      CSPX_info = {}

      #num_of_res_unit
      CSPX_info["num_of_res_unit"] = 1

      #res_unit_info

      res_unit_1 = {}
      res_unit_1["CBM_1"] = (32,1,1,"same")
      res_unit_1["CBM_2"] = (64,3,1,"same")
      CSPX_info["res_unit_1"] = res_unit_1


      #CBL_1
      CSPX_info["CBL_1"] = (64,3,2,"valid")

      #CBL_2
      CSPX_info["CBL_2"] = (64,1,1,"same")

      #CBL_3
      CSPX_info["CBL_3"] = (64,3,1,"same")

      #CBL_4
      CSPX_info["CBL_4"] = (64,3,1,"same")

      #CBM_1
      CSPX_info["CBM_1"] = (64,1,1,"same")

      #define CSP1
      self.CSP1 = CSPX(CSPX_info)

      #----------------------------------------------------------------

      #CSP2 in : 319 x 319 x 32 out : 159 x 159 x 128
      CSPX_info = {}

      #num_of_res_unit
      CSPX_info["num_of_res_unit"] = 2

      #res_unit_info

      res_unit_1 = {}
      res_unit_1["CBM_1"] = (64,1,1,"same")
      res_unit_1["CBM_2"] = (64,3,1,"same")
      CSPX_info["res_unit_1"] = res_unit_1

      res_unit_2 = {}
      res_unit_2["CBM_1"] = (64,1,1,"same")
      res_unit_2["CBM_2"] = (64,3,1,"same")
      CSPX_info["res_unit_2"] = res_unit_2


      #CBL_1
      CSPX_info["CBL_1"] = (128,3,2,"valid")

      #CBL_2
      CSPX_info["CBL_2"] = (64,1,1,"same")

      #CBL_3
      CSPX_info["CBL_3"] = (64,3,1,"same")

      #CBL_4
      CSPX_info["CBL_4"] = (64,3,1,"same")

      #CBM_1
      CSPX_info["CBM_1"] = (128,1,1,"same")

      #define CSP2
      self.CSP2 = CSPX(CSPX_info)

      #----------------------------------------------------------------

      #CSP8_1 in : 159 x 159 x 128 out : 79 x 79 x 256
      CSPX_info = {}

      #num_of_res_unit
      CSPX_info["num_of_res_unit"] = 8

      #res_unit_info

      res_unit_1 = {}
      res_unit_1["CBM_1"] = (128,1,1,"same")
      res_unit_1["CBM_2"] = (128,3,1,"same")
      CSPX_info["res_unit_1"] = res_unit_1

      res_unit_2 = {}
      res_unit_2["CBM_1"] = (128,1,1,"same")
      res_unit_2["CBM_2"] = (128,3,1,"same")
      CSPX_info["res_unit_2"] = res_unit_2

      res_unit_3 = {}
      res_unit_3["CBM_1"] = (128,1,1,"same")
      res_unit_3["CBM_2"] = (128,3,1,"same")
      CSPX_info["res_unit_3"] = res_unit_3

      res_unit_4 = {}
      res_unit_4["CBM_1"] = (128,1,1,"same")
      res_unit_4["CBM_2"] = (128,3,1,"same")
      CSPX_info["res_unit_4"] = res_unit_4

      res_unit_5 = {}
      res_unit_5["CBM_1"] = (128,1,1,"same")
      res_unit_5["CBM_2"] = (128,3,1,"same")
      CSPX_info["res_unit_5"] = res_unit_5

      res_unit_6 = {}
      res_unit_6["CBM_1"] = (128,1,1,"same")
      res_unit_6["CBM_2"] = (128,3,1,"same")
      CSPX_info["res_unit_6"] = res_unit_6

      res_unit_7 = {}
      res_unit_7["CBM_1"] = (128,1,1,"same")
      res_unit_7["CBM_2"] = (128,3,1,"same")
      CSPX_info["res_unit_7"] = res_unit_7

      res_unit_8 = {}
      res_unit_8["CBM_1"] = (128,1,1,"same")
      res_unit_8["CBM_2"] = (128,3,1,"same")
      CSPX_info["res_unit_8"] = res_unit_8



      #CBL_1
      CSPX_info["CBL_1"] = (256,3,2,"valid")

      #CBL_2
      CSPX_info["CBL_2"] = (128,1,1,"same")

      #CBL_3
      CSPX_info["CBL_3"] = (128,3,1,"same")

      #CBL_4
      CSPX_info["CBL_4"] = (128,3,1,"same")

      #CBM_1
      CSPX_info["CBM_1"] = (256,1,1,"same")

      #define CSP8_1
      self.CSP8_1 = CSPX(CSPX_info)

      #----------------------------------------------------------------

      #spatialdropout_1 in : 79 x 79 x 256 out : 79 x 79 x 256 , ( branch 1 -- out : 79 x 79 x 256 )
      self.SPA_drop_1 = tf.keras.layers.SpatialDropout2D(rate = 0.5,data_format="channels_last")

      #----------------------------------------------------------------

      #CSP8_2 in : 79 x 79 x 256 out : 39 x 39 x 512
      CSPX_info = {}

      #num_of_res_unit
      CSPX_info["num_of_res_unit"] = 8

      #res_unit_info

      res_unit_1 = {}
      res_unit_1["CBM_1"] = (256,1,1,"same")
      res_unit_1["CBM_2"] = (256,3,1,"same")
      CSPX_info["res_unit_1"] = res_unit_1

      res_unit_2 = {}
      res_unit_2["CBM_1"] = (256,1,1,"same")
      res_unit_2["CBM_2"] = (256,3,1,"same")
      CSPX_info["res_unit_2"] = res_unit_2

      res_unit_3 = {}
      res_unit_3["CBM_1"] = (256,1,1,"same")
      res_unit_3["CBM_2"] = (256,3,1,"same")
      CSPX_info["res_unit_3"] = res_unit_3

      res_unit_4 = {}
      res_unit_4["CBM_1"] = (256,1,1,"same")
      res_unit_4["CBM_2"] = (256,3,1,"same")
      CSPX_info["res_unit_4"] = res_unit_4

      res_unit_5 = {}
      res_unit_5["CBM_1"] = (256,1,1,"same")
      res_unit_5["CBM_2"] = (256,3,1,"same")
      CSPX_info["res_unit_5"] = res_unit_5

      res_unit_6 = {}
      res_unit_6["CBM_1"] = (256,1,1,"same")
      res_unit_6["CBM_2"] = (256,3,1,"same")
      CSPX_info["res_unit_6"] = res_unit_6

      res_unit_7 = {}
      res_unit_7["CBM_1"] = (256,1,1,"same")
      res_unit_7["CBM_2"] = (256,3,1,"same")
      CSPX_info["res_unit_7"] = res_unit_7

      res_unit_8 = {}
      res_unit_8["CBM_1"] = (256,1,1,"same")
      res_unit_8["CBM_2"] = (256,3,1,"same")
      CSPX_info["res_unit_8"] = res_unit_8



      #CBL_1
      CSPX_info["CBL_1"] = (512,3,2,"valid")

      #CBL_2
      CSPX_info["CBL_2"] = (256,1,1,"same")

      #CBL_3
      CSPX_info["CBL_3"] = (256,3,1,"same")

      #CBL_4
      CSPX_info["CBL_4"] = (256,3,1,"same")

      #CBM_1
      CSPX_info["CBM_1"] = (512,1,1,"same")

      #define CSP8_2
      self.CSP8_2 = CSPX(CSPX_info)

      #----------------------------------------------------------------

      #spatialdropout_2 in : 39 x 39 x 512 out : 39 x 39 x 512 , ( branch 2 -- out : 39 x 39 x 512 )
      self.SPA_drop_2 = tf.keras.layers.SpatialDropout2D(rate = 0.5,data_format="channels_last")

      #----------------------------------------------------------------

      #CSP4 in : 39 x 39 x 512 out : 19 x 19 x 1024
      CSPX_info = {}

      #num_of_res_unit
      CSPX_info["num_of_res_unit"] = 4

      #res_unit_info

      res_unit_1 = {}
      res_unit_1["CBM_1"] = (512,1,1,"same")
      res_unit_1["CBM_2"] = (512,3,1,"same")
      CSPX_info["res_unit_1"] = res_unit_1

      res_unit_2 = {}
      res_unit_2["CBM_1"] = (512,1,1,"same")
      res_unit_2["CBM_2"] = (512,3,1,"same")
      CSPX_info["res_unit_2"] = res_unit_2

      res_unit_3 = {}
      res_unit_3["CBM_1"] = (512,1,1,"same")
      res_unit_3["CBM_2"] = (512,3,1,"same")
      CSPX_info["res_unit_3"] = res_unit_3

      res_unit_4 = {}
      res_unit_4["CBM_1"] = (512,1,1,"same")
      res_unit_4["CBM_2"] = (512,3,1,"same")
      CSPX_info["res_unit_4"] = res_unit_4


      #CBL_1
      CSPX_info["CBL_1"] = (1024,3,2,"valid")

      #CBL_2
      CSPX_info["CBL_2"] = (512,1,1,"same")

      #CBL_3
      CSPX_info["CBL_3"] = (512,3,1,"same")

      #CBL_4
      CSPX_info["CBL_4"] = (512,3,1,"same")

      #CBM_1
      CSPX_info["CBM_1"] = (1024,1,1,"same")

      #define CSP4
      self.CSP4 = CSPX(CSPX_info)

      #----------------------------------------------------------------

      #spatialdropout_3 in : 19 x 19 x 1024  out : 19 x 19 x 1024
      self.SPA_drop_3 = tf.keras.layers.SpatialDropout2D(rate = 0.5,data_format="channels_last")

      #----------------------------------------------------------------

      #rCSP1 in : 19 x 19 x 1024 out : 19 x 19 x 512 , ( branch 3 -- out: 19 x 19 x 512 )
      rCSP_info = {}

      rCSP_info["CBL_1"] = (1024,3,1,"same")
      rCSP_info["CBL_2"] = (512,1,1,"same")
      rCSP_info["CBL_3"] = (1024,3,1,"same")
      rCSP_info["CBL_4"] = (512,1,1,"same")
      rCSP_info["CBL_5"] = (512,1,1,"same")
      rCSP_info["CBL_6"] = (512,3,1,"same")
      rCSP_info["CBL_7"] = (512,1,1,"same")

      #define rCSP1
      self.rCSP1 = rCSP(rCSP_info)
      #----------------------------------------------------------------

      #CBL_1 in : 19 x 19 x 512 out : 19 x 19 x 256
      self.CBL_1 = CBL(256,1,1,"same")

      #----------------------------------------------------------------

      #bilinear upsampling x4_1 in : 19 x 19 x 256 out : 76 x 76 x 256
      self.upsample_bilinear_x4_1 = tf.keras.layers.UpSampling2D(size=4,interpolation="bilinear",data_format = "channels_last")

      #TCBM_1 in : 76 x 76 x 256 out : 80 x 80 x 256
      self.TCBM_1 = TCBM(256,5,1,"valid")

      #bilinear upsampling x2_1 in : 39 x 39 x 512 out : 78 x 78 x 512 - connect branch 2
      self.upsample_bilinear_x2_1 = tf.keras.layers.UpSampling2D(size=2,interpolation="bilinear",data_format = "channels_last")      

      #TCBM_2 in : 78 x 78 x 512 out : 80 x 80 x 256
      self.TCBM_2 = TCBM(256,3,1,"valid")

      #concat TCBM_1 -- TCBM_2 , out: 80 x 80 x 512

      #----------------------------------------------------------------

      #rCSP2 in : 80 x 80 x 512 out : 39 x 39 x 256 ( branch 4 -- out: 39 x 39 x 256 )
      rCSP_info = {}

      rCSP_info["CBL_1"] = (1024,3,2,"valid")
      rCSP_info["CBL_2"] = (512,1,1,"same")
      rCSP_info["CBL_3"] = (1024,3,1,"same")
      rCSP_info["CBL_4"] = (512,1,1,"same")
      rCSP_info["CBL_5"] = (512,1,1,"same")
      rCSP_info["CBL_6"] = (512,3,1,"same")
      rCSP_info["CBL_7"] = (256,1,1,"same")

      #define rCSP2
      self.rCSP2 = rCSP(rCSP_info)

      #----------------------------------------------------------------

      #CBL_2 in : 39 x 39 x 256 out : 39 x 39 x 128
      self.CBL_2 = CBL(128,1,1,"same")

      #----------------------------------------------------------------

      #bilinear upsampling x2_2 in : 39 x 39 x 128 out : 78 x 78 x 128
      self.upsample_bilinear_x2_2 = tf.keras.layers.UpSampling2D(size=2,interpolation="bilinear",data_format = "channels_last")

      #TCBM_3 in : 78 x 78 x 128 out : 80 x 80 x 128
      self.TCBM_3 = TCBM(128,3,1,"valid")

      #TCBM_4  in : 79 x 79 x 256 out : 80 x 80 x 128 - connect branch 1 
      self.TCBM_4 = TCBM(128,2,1,"valid")

      #concat TCBM_3 -- TCBM_4 , out: 80 x 80 x 256

      #----------------------------------------------------------------

      #rCSP3 in : 80 x 80 x 256 out : 80 x 80 x 128 ( branch 5 -- out: 80 x 80 x 128 )
      rCSP_info = {}

      rCSP_info["CBL_1"] = (256,3,1,"same")
      rCSP_info["CBL_2"] = (128,1,1,"same")
      rCSP_info["CBL_3"] = (256,3,1,"same")
      rCSP_info["CBL_4"] = (128,1,1,"same")
      rCSP_info["CBL_5"] = (128,1,1,"same")
      rCSP_info["CBL_6"] = (128,3,1,"same")
      rCSP_info["CBL_7"] = (128,1,1,"same")

      #define rCSP3
      self.rCSP3 = rCSP(rCSP_info)

      #----------------------------------------------------------------

      #decouple head -- small object , in : 80 x 80 x 256  out: 80 x 80 x (1 + 2 + 2 + 80)

      #reg
      self.TCBL_reg_small  = TCBL(256,1,1,"valid")

      self.CBL_left_small = CBL(2,3,1,"same")

      self.CBL_center_small = CBL(2,3,1,"same")
      
      #class + prob
      self.TCBL_clsp_small  = TCBL(256,1,1,"valid")

      self.CBL_prob_small = CBL(1,3,1,"same")

      self.CBL_class_small = CBL(80,3,1,"same")

      self.CBS_prob_small = CBS(1,1,1,"same")

      self.CBS_class_small =  CBS(80,1,1,"same")

      #concat CBS_prob_small -- CBL_left_small -- CBL_center_small -- CBS_class_small  , out: 80 x 80 x 85

      #output small
      #self.conv2D_small = tf.keras.layers.Conv2D(85,1,1,padding="same",data_format = "channels_last",name="output_small")
      

      #----------------------------------------------------------------

      #connect_branch_5_CBL in : 80 x 80 x 128  out: 39 x 39 x 256
      self.connect_branch_5_CBL = CBL(256,3,2,padding="valid")
      
      #concat branch 4 -- branch 5  , out: 39 x 39 x 512

      #----------------------------------------------------------------

      #rCSP4 in : 39 x 39 x 512  out : 39 x 39 x 256 ( branch 6 -- out: 39 x 39 x 256 )
      rCSP_info = {}

      rCSP_info["CBL_1"] = (512,3,1,"same")
      rCSP_info["CBL_2"] = (256,1,1,"same")
      rCSP_info["CBL_3"] = (512,3,1,"same")
      rCSP_info["CBL_4"] = (256,1,1,"same")
      rCSP_info["CBL_5"] = (256,1,1,"same")
      rCSP_info["CBL_6"] = (256,3,1,"same")
      rCSP_info["CBL_7"] = (256,1,1,"same")

      #define rCSP4
      self.rCSP4 = rCSP(rCSP_info)

      #----------------------------------------------------------------

      #decouple head -- medium object , in : 39 x 39 x 256  out: 40 x 40 x (1 + 2 + 2 + 2 + 2 + 80)

      #reg
      self.TCBL_reg_medium  = TCBL(256,2,1,"valid")

      self.CBL_left_medium = CBL(2,3,1,"same")

      self.CBL_center_medium = CBL(2,3,1,"same")

      #class + prob
      self.TCBL_clsp_medium  = TCBL(256,2,1,"valid")

      self.CBL_prob_medium = CBL(1,3,1,"same")

      self.CBL_class_medium = CBL(80,3,1,"same")

      self.CBS_prob_medium = CBS(1,1,1,"same")

      self.CBS_class_medium =  CBS(80,1,1,"same")
      
      #concat CBS_prob_medium  -- CBL_left_medium -- CBL_center_medium -- CBS_class_medium  , out: 40 x 40 x 85
      
      #output medium
      #self.conv2D_medium = tf.keras.layers.Conv2D(85,1,1,padding="same",data_format = "channels_last",name="output_medium")

      #----------------------------------------------------------------

      #connect_branch_6_CBL in : 39 x 39 x 256  out: 19 x 19 x 512
      self.connect_branch_6_CBL = CBL(512,3,2,padding="valid")

      #concat branch 3 -- branch 6  , out: 19 x 19 x 1024

      #----------------------------------------------------------------

      #rCSP5 in : 19 x 19 x 1024  out : 19 x 19 x 512 
      rCSP_info = {}

      rCSP_info["CBL_1"] = (1024,3,1,"same")
      rCSP_info["CBL_2"] = (512,1,1,"same")
      rCSP_info["CBL_3"] = (1024,3,1,"same")
      rCSP_info["CBL_4"] = (512,1,1,"same")
      rCSP_info["CBL_5"] = (512,1,1,"same")
      rCSP_info["CBL_6"] = (512,3,1,"same")
      rCSP_info["CBL_7"] = (512,1,1,"same")

      #define rCSP5
      self.rCSP5 = rCSP(rCSP_info)

      #----------------------------------------------------------------

      #decouple head -- large object , in : 19 x 19 x 512   out: 20 x 20 x (1 + 2 + 2 + 2 + 2 + 80)

      #reg
      self.TCBL_reg_large  = TCBL(512,2,1,"valid")

      self.CBL_left_large = CBL(2,3,1,"same")

      self.CBL_center_large = CBL(2,3,1,"same")

      #class + prob
      self.TCBL_clsp_large  = TCBL(512,2,1,"valid")

      self.CBL_prob_large = CBL(1,3,1,"same")

      self.CBL_class_large = CBL(80,3,1,"same")

      self.CBS_prob_large = CBS(1,1,1,"same")

      self.CBS_class_large =  CBS(80,1,1,"same")

      #concat CBS_prob_large -- CBL_left_large -- CBL_center_large -- CBS_class_large  , out: 20 x 20 x 85
      
      #output large
      #self.conv2D_large = tf.keras.layers.Conv2D(85,1,1,padding="same",data_format = "channels_last",name="output_large")

      #----------------------------------------------------------------
      
   def call(self,inputs,train_flag=True):

      #CBM_1
      CBM_1 = self.CBM_1(inputs,train_flag)

      #CSP1
      CSP1 = self.CSP1(CBM_1,train_flag)

      #CSP2
      CSP2 = self.CSP2(CSP1,train_flag)

      #CSP8_1
      CSP8_1 = self.CSP8_1(CSP2,train_flag)

      #spatialdropout_1 -- branch 1
      SPA_drop_1 = self.SPA_drop_1(CSP8_1,training=train_flag)

      #CSP8_2
      CSP8_2 = self.CSP8_2(SPA_drop_1,train_flag)

      #spatialdropout_2 -- branch 2
      SPA_drop_2 = self.SPA_drop_2(CSP8_2,training=train_flag)

      #CSP4
      CSP4 = self.CSP4(SPA_drop_2,train_flag)

      #spatialdropout_3 
      SPA_drop_3 = self.SPA_drop_3(CSP4,training=train_flag)

      #rCSP1 -- branch 3
      rCSP1 = self.rCSP1(SPA_drop_3,train_flag)

      #CBL_1
      CBL_1 = self.CBL_1(rCSP1,train_flag)

      #bilinear upsampling x4_1
      upsample_bilinear_x4_1 = self.upsample_bilinear_x4_1(CBL_1)

      #TCBM_1
      TCBM_1 = self.TCBM_1(upsample_bilinear_x4_1,train_flag)
      
      #bilinear upsampling x2_1 - connect branch 2
      upsample_bilinear_x2_1 = self.upsample_bilinear_x2_1(SPA_drop_2)

      ##TCBM_2 
      TCBM_2 = self.TCBM_2(upsample_bilinear_x2_1,train_flag)

      #mid concat 1 -- concat TCBM_1 -- TCBM_2
      mid_concat_1 = tf.keras.layers.concatenate(inputs=[TCBM_1,TCBM_2],axis=-1)

      #rCSP2 -- branch 4
      rCSP2 = self.rCSP2(mid_concat_1,train_flag)

      #CBL_2
      CBL_2 = self.CBL_2(rCSP2,train_flag)

      #bilinear upsampling x2_2
      upsample_bilinear_x2_2 = self.upsample_bilinear_x2_2(CBL_2)

      #TCBM_3
      TCBM_3 = self.TCBM_3(upsample_bilinear_x2_2,train_flag)

      #TCBM_4 - connect branch 1
      TCBM_4 = self.TCBM_4(SPA_drop_1,train_flag)

      #mid concat 2 -- concat TCBM_3 -- TCBM_4
      mid_concat_2 = tf.keras.layers.concatenate(inputs=[TCBM_3,TCBM_4],axis=-1)

      #rCSP3 -- branch 5
      rCSP3 = self.rCSP3(mid_concat_2,train_flag)

      #decouple head -- small object

      #reg -- small
      TCBL_reg_small = self.TCBL_reg_small(rCSP3,train_flag)

      CBL_left_small = self.CBL_left_small(TCBL_reg_small,train_flag)

      CBL_center_small = self.CBL_center_small(TCBL_reg_small,train_flag)

      #class + prob -- small
      TCBL_clsp_small = self.TCBL_clsp_small(rCSP3,train_flag)

      CBL_prob_small = self.CBL_prob_small(TCBL_clsp_small,train_flag)

      CBL_class_small = self.CBL_class_small(TCBL_clsp_small,train_flag)

      CBS_prob_small = self.CBS_prob_small(CBL_prob_small,train_flag)

      CBS_class_small = self.CBS_class_small(CBL_class_small,train_flag)
      
      #concat CBS_prob_small -- CBL_left_small -- CBL_center_small -- CBS_class_small  , out: 80 x 80 x 85
      #small_concat = tf.keras.layers.concatenate(inputs=[CBL_prob_small,CBL_left_small,CBL_center_small,CBL_class_small],axis=-1)

      #**************** output small ****************
      
      #output_small = self.conv2D_small(small_concat)
      output_small = tf.keras.layers.concatenate(inputs=[CBS_prob_small,CBL_left_small,CBL_center_small,CBS_class_small],axis=-1,name="output_small")
      
      #**************** output small ****************

      #connect_branch_5_CBL
      connect_branch_5_CBL = self.connect_branch_5_CBL(rCSP3,train_flag)

      #concat branch 4 -- branch 5  , out: 40 x 40 x 512
      mid_concat_br45 = tf.keras.layers.concatenate(inputs=[rCSP2,connect_branch_5_CBL],axis=-1)

      #rCSP4 -- branch 6
      rCSP4 = self.rCSP4(mid_concat_br45,train_flag)

      #decouple head -- medium object

      #reg -- medium
      TCBL_reg_medium = self.TCBL_reg_medium(rCSP4,train_flag)

      CBL_left_medium = self.CBL_left_medium(TCBL_reg_medium,train_flag)

      CBL_center_medium = self.CBL_center_medium(TCBL_reg_medium,train_flag)


      #class + prob -- medium
      TCBL_clsp_medium = self.TCBL_clsp_medium(rCSP4,train_flag)

      CBL_prob_medium = self.CBL_prob_medium(TCBL_clsp_medium,train_flag)

      CBL_class_medium = self.CBL_class_medium(TCBL_clsp_medium,train_flag)

      CBS_prob_medium = self.CBS_prob_medium(CBL_prob_medium,train_flag)

      CBS_class_medium = self.CBS_class_medium(CBL_class_medium,train_flag)


      #concat CBS_prob_medium  -- CBL_left_medium -- CBL_center_medium -- CBS_class_medium  , out: 40 x 40 x 85
      #medium_concat = tf.keras.layers.concatenate(inputs=[CBL_prob_medium,CBL_left_medium,CBL_center_medium,CBL_class_medium],axis=-1)
      
      #**************** output medium ****************
      
      #output_medium = self.conv2D_medium(medium_concat)
      output_medium = tf.keras.layers.concatenate(inputs=[CBS_prob_medium,CBL_left_medium,CBL_center_medium,CBS_class_medium],axis=-1,name="output_medium")

      #**************** output medium ****************

      #connect_branch_6_CBL
      connect_branch_6_CBL = self.connect_branch_6_CBL(rCSP4,train_flag)

      #concat branch 3 -- branch 6  , out: 19 x 19 x 1024
      mid_concat_br36 = tf.keras.layers.concatenate(inputs=[rCSP1,connect_branch_6_CBL],axis=-1)
      
      #rCSP5
      rCSP5 = self.rCSP5(mid_concat_br36,train_flag)

      #decouple head -- large object 

      #reg -- large
      TCBL_reg_large = self.TCBL_reg_large(rCSP5,train_flag)

      CBL_left_large = self.CBL_left_large(TCBL_reg_large,train_flag)

      CBL_center_large = self.CBL_center_large(TCBL_reg_large,train_flag)

      #class + prob -- large
      TCBL_clsp_large = self.TCBL_clsp_large(rCSP5,train_flag)

      CBL_prob_large = self.CBL_prob_large(TCBL_clsp_large,train_flag)

      CBL_class_large = self.CBL_class_large(TCBL_clsp_large,train_flag)

      CBS_prob_large = self.CBS_prob_large(CBL_prob_large,train_flag)

      CBS_class_large = self.CBS_class_large(CBL_class_large,train_flag)

      #concat CBS_prob_large -- CBL_left_large -- CBL_center_large -- CBS_class_large  , out: 20 x 20 x 85
      #large_concat = tf.keras.layers.concatenate(inputs=[CBL_prob_large,CBL_left_large,CBL_center_large,CBL_class_large],axis=-1)
      
      #**************** output large ****************
      
      #output_large = self.conv2D_large(large_concat)
      output_large = tf.keras.layers.concatenate(inputs=[CBS_prob_large,CBL_left_large,CBL_center_large,CBS_class_large],axis=-1,name="output_large")

      #**************** output large ****************

      return [output_large,output_medium,output_small]

   def graph_model(self,dim):

      x = tf.keras.layers.Input(shape=dim)
      
      return tf.keras.Model(inputs=x,outputs=self.call(x))
