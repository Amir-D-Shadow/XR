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

      #$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$
      
      #CBM_1 in : 640 x 640 x 3 out: 640 x 640 x 32
      filters=32
      kernel_size=3
      strides=1
      padding="same"

      self.CBM_1 = CBM(filters,kernel_size,strides,padding)

      #$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$

      #CSP_1 in : 640 x 640 x 32 out: 319 x 319 x 64
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
      CSPX_info["CBL_3"] = (64,1,1,"same")

      #CBL_4
      CSPX_info["CBL_4"] = (64,1,1,"same")

      #CBM_1
      CSPX_info["CBM_1"] = (64,1,1,"same")

      #define CSP1
      self.CSP1 = CSPX(CSPX_info)

      #$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$

      #CSP2 in : 319 x 319 x 64 out : 159 x 159 x 128
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
      CSPX_info["CBL_3"] = (64,1,1,"same")

      #CBL_4
      CSPX_info["CBL_4"] = (64,1,1,"same")

      #CBM_1
      CSPX_info["CBM_1"] = (128,1,1,"same")

      #define CSP2
      self.CSP2 = CSPX(CSPX_info)

      #$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$

      #CSP8_1 in : 159 x 159 x 128  out : 79 x 79 x 256 -- branch 1
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
      CSPX_info["CBL_3"] = (128,1,1,"same")

      #CBL_4
      CSPX_info["CBL_4"] = (128,1,1,"same")

      #CBM_1
      CSPX_info["CBM_1"] = (256,1,1,"same")

      #define CSP8_1 -- branch_1
      self.CSP8_branch_1 = CSPX(CSPX_info)

      #$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$

      #CSP8_2 in : 79 x 79 x 256 out : 39 x 39 x 512 --- branch_2
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
      CSPX_info["CBL_3"] = (256,1,1,"same")

      #CBL_4
      CSPX_info["CBL_4"] = (256,1,1,"same")

      #CBM_1
      CSPX_info["CBM_1"] = (512,1,1,"same")

      #define CSP8_2 --- branch_2
      self.CSP8_branch_2 = CSPX(CSPX_info)

      #$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$

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
      CSPX_info["CBL_3"] = (512,1,1,"same")

      #CBL_4
      CSPX_info["CBL_4"] = (512,1,1,"same")

      #CBM_1
      CSPX_info["CBM_1"] = (1024,1,1,"same")

      #define CSP4
      self.CSP4 = CSPX(CSPX_info)

      #$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$

      #AttentionModule_neck_1  in : 19 x 19 x 1024 out : 19 x 19 x 1024
      attention_info = {}
      attention_info["CBL_1"] = (1024,3,1,"same")
      attention_info["conv_query"] = (1024,3,1,"same")
      attention_info["conv_keys"] = (1024,3,1,"same")
      attention_info["conv_values"] = (1024,3,1,"same")

      self.AttentionModule_neck_1 = AttentionModule(attention_info)

      #AttentionModule_neck_2  in : 19 x 19 x 1024 out : 19 x 19 x 1024
      attention_info = {}
      attention_info["CBL_1"] = (1024,3,1,"same")
      attention_info["conv_query"] = (1024,3,1,"same")
      attention_info["conv_keys"] = (1024,3,1,"same")
      attention_info["conv_values"] = (1024,3,1,"same")

      self.AttentionModule_neck_2 = AttentionModule(attention_info)

      #AttentionModule_neck_3  in : 19 x 19 x 1024 out : 19 x 19 x 1024
      attention_info = {}
      attention_info["CBL_1"] = (1024,3,1,"same")
      attention_info["conv_query"] = (1024,3,1,"same")
      attention_info["conv_keys"] = (1024,3,1,"same")
      attention_info["conv_values"] = (1024,3,1,"same")

      self.AttentionModule_neck_3 = AttentionModule(attention_info)

      #AttentionModule_neck_4  in : 19 x 19 x 1024 out : 19 x 19 x 1024
      attention_info = {}
      attention_info["CBL_1"] = (1024,3,1,"same")
      attention_info["conv_query"] = (1024,3,1,"same")
      attention_info["conv_keys"] = (1024,3,1,"same")
      attention_info["conv_values"] = (1024,3,1,"same")

      self.AttentionModule_neck_4 = AttentionModule(attention_info)

      #AttentionModule_neck_5  in : 19 x 19 x 1024 out : 19 x 19 x 1024
      attention_info = {}
      attention_info["CBL_1"] = (1024,3,1,"same")
      attention_info["conv_query"] = (1024,3,1,"same")
      attention_info["conv_keys"] = (1024,3,1,"same")
      attention_info["conv_values"] = (1024,3,1,"same")

      self.AttentionModule_neck_5 = AttentionModule(attention_info)

      #AttentionModule_neck_6_branch_3  in : 19 x 19 x 1024 out : 19 x 19 x 512 -- branch 3
      attention_info = {}
      attention_info["CBL_1"] = (512,1,1,"same")
      attention_info["conv_query"] = (512,3,1,"same")
      attention_info["conv_keys"] = (512,3,1,"same")
      attention_info["conv_values"] = (512,3,1,"same")

      self.AttentionModule_neck_6_branch_3 = AttentionModule(attention_info)

      #$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$
      
      #CBL 7 neck in : 19 x 19 x 512 out : 19 x 19 x 256
      self.CBL_7_neck = CBL(256,1,1,"same")

      #TCBL1 in : 19 x 19 x 256 out: 39 x 39 x 256
      self.TCBL1 = TCBL(256,21,1,"valid")

      #CBL_connect_branch_2 in : 39 x 39 x 512 out: 39 x 39 x 256
      self.CBL_connect_branch_2 = CBL(256,1,1,"same")

      #concat TCBL1 -- CBL_connect_branch_2 , out: 39 x 39 x 512

      #$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$

      #CBL_1 phase1 in : 39 x 39 x 512 out : 39 x 39 x 256
      self.CBL_1_phase1 = CBL(256,1,1,"same")

      #CBL_2 phase1 in : 39 x 39 x 256 out : 39 x 39 x 512
      self.CBL_2_phase1 = CBL(512,3,1,"same")

      #CBL_3 phase1 in : 39 x 39 x 512 out : 39 x 39 x 256
      self.CBL_3_phase1 = CBL(256,1,1,"same")

      #CBL_4 phase1 in : 39 x 39 x 256 out : 39 x 39 x 512
      self.CBL_4_phase1 = CBL(512,3,1,"same")

      #CBL_5 phase1_branch_4 in : 39 x 39 x 512 out : 39 x 39 x 256 -- branch 4
      self.CBL_5_phase1_branch_4 = CBL(256,1,1,"same")

      #$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$
      
      #CBL_6 phase1 in : 39 x 39 x 256 out : 39 x 39 x 128
      self.CBL_6_phase1 = CBL(128,1,1,"same")

      #TCBL2 in : 39 x 39 x 128 out: 79 x 79 x 128
      self.TCBL2 = TCBL(128,41,1,"valid")

      #CBL_connect_branch_1 in : 79 x 79 x 256 out: 79 x 79 x 128
      self.CBL_connect_branch_1 = CBL(128,1,1,"same")

      #concat TCBL2 -- CBL_connect_branch_1 , out: 79 x 79 x 256

      #$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$

      #CBL_1 phase2 in : 79 x 79 x 256 out : 79 x 79 x 128
      self.CBL_1_phase2 = CBL(128,1,1,"same")

      #CBL_2 phase2 in : 79 x 79 x 128 out : 79 x 79 x 256
      self.CBL_2_phase2 = CBL(256,3,1,"same")

      #CBL_3 phase2 in : 79 x 79 x 256 out : 79 x 79 x 128
      self.CBL_3_phase2 = CBL(128,1,1,"same")

      #CBL_4 phase2 in : 79 x 79 x 128 out : 79 x 79 x 256
      self.CBL_4_phase2 = CBL(256,3,1,"same")

      #CBL_5 phase2_branch_5 in : 79 x 79 x 256 out : 79 x 79 x 128 -- branch 5
      self.CBL_5_phase2_branch_5 = CBL(128,1,1,"same")

      #$#$#$#$#$#$#$#$#$#$#$#$#$# small output $#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$

      #TCBL_small in : 79 x 79 x 128 out : 80 x 80 x 128
      self.TCBL_small = TCBL(128,2,1,"valid")

      #prob info

      #CBL_prob_class_small_1 in : 80 x 80 x 128 out : 80 x 80 x 256
      self.CBL_prob_class_small_1 = CBL(256,3,1,"same")

      #CBL_prob_class_small_2 in : 80 x 80 x 256 out : 80 x 80 x 256
      self.CBL_prob_class_small_2 = CBL(256,1,1,"same")

      #conv_prob_small in : 80 x 80 x 256 out : 80 x 80 x 1
      self.conv_prob_small = tf.keras.layers.Conv2D(filters=1,kernel_size=1,strides=1,padding="same",data_format="channels_last",activation="sigmoid")

      #conv_class_small in : 80 x 80 x 256 out : 80 x 80 x 80
      self.conv_class_small = tf.keras.layers.Conv2D(filters=80,kernel_size=1,strides=1,padding="same",data_format="channels_last",activation="sigmoid")
      
      #Reg info
      
      #CBL_left_center_small_1 in : 80 x 80 x 128 out : 80 x 80 x 256
      self.CBL_left_center_small_1 = CBL(256,3,1,"same")

      #CBL_left_center_small_2 in : 80 x 80 x 256 out : 80 x 80 x 256
      self.CBL_left_center_small_2 = CBL(256,1,1,"same")

      #conv_pos_info_small in : 80 x 80 x 256 out : 80 x 80 x 4
      self.conv_pos_info_small = tf.keras.layers.Conv2D(filters=4,kernel_size=1,strides=1,padding="same",data_format="channels_last",activation=tf.keras.layers.LeakyReLU())

      #concat conv_prob_small -- conv_pos_info_small -- conv_class_small , out: 80 x 80 x 85

      #$#$#$#$#$#$#$#$#$#$#$#$#$# small output $#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$

      #CBL_connect_branch_5 in : 79 x 79 x 128 out : 39 x 39 x 256
      self.CBL_connect_branch_5 = CBL(256,3,2,"valid")

      #concat CBL_connect_branch_5 -- CBL_5_phase1_branch_4 , out: 39 x 39 x 512

      #$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$

      #CBL_1 phase3 in : 39 x 39 x 512 out : 39 x 39 x 256
      self.CBL_1_phase3 = CBL(256,1,1,"same")

      #CBL_2 phase3 in : 39 x 39 x 256 out : 39 x 39 x 512
      self.CBL_2_phase3 = CBL(512,3,1,"same")

      #CBL_3 phase3 in : 39 x 39 x 512 out : 39 x 39 x 256
      self.CBL_3_phase3 = CBL(256,1,1,"same")

      #CBL_4 phase3 in : 39 x 39 x 256 out : 39 x 39 x 512
      self.CBL_4_phase3 = CBL(512,3,1,"same")

      #CBL_5 phase3_branch_6 in : 39 x 39 x 512 out : 39 x 39 x 256 -- branch 6
      self.CBL_5_phase3_branch_6 = CBL(256,1,1,"same")

      #$#$#$#$#$#$#$#$#$#$#$#$#$# medium output $#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$

      #TCBL_medium in : 39 x 39 x 256 out : 40 x 40 x 256
      self.TCBL_medium = TCBL(128,2,1,"valid")

      #prob info

      #CBL_prob_class_medium_1 in : 40 x 40 x 256 out : 40 x 40 x 512
      self.CBL_prob_class_medium_1 = CBL(512,3,1,"same")

      #CBL_prob_class_medium_2 in : 40 x 40 x 512 out : 40 x 40 x 512
      self.CBL_prob_class_medium_2 = CBL(512,1,1,"same")

      #conv_prob_medium in : 40 x 40 x 512 out : 40 x 40 x 1
      self.conv_prob_medium = tf.keras.layers.Conv2D(filters=1,kernel_size=1,strides=1,padding="same",data_format="channels_last",activation="sigmoid")

      #conv_class_medium in : 40 x 40 x 512 out : 40 x 40 x 80
      self.conv_class_medium = tf.keras.layers.Conv2D(filters=80,kernel_size=1,strides=1,padding="same",data_format="channels_last",activation="sigmoid")


      #Reg info
      
      #CBL_left_center_medium_1 in : 40 x 40 x 256 out : 40 x 40 x 512
      self.CBL_left_center_medium_1 = CBL(512,3,1,"same")

      #CBL_left_center_medium_2 in :  40 x 40 x 512 out :  40 x 40 x 512
      self.CBL_left_center_medium_2 = CBL(512,1,1,"same")

      #conv_pos_info_medium in : 40 x 40 x 512 out : 40 x 40 x 4
      self.conv_pos_info_medium = tf.keras.layers.Conv2D(filters=4,kernel_size=1,strides=1,padding="same",data_format="channels_last",activation=tf.keras.layers.LeakyReLU())

      #concat conv_prob_medium -- conv_pos_info_medium -- conv_class_medium , out: 40 x 40 x 85

      #$#$#$#$#$#$#$#$#$#$#$#$#$# medium output $#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$

      #CBL_connect_branch_6 in : 39 x 39 x 256 out : 19 x 19 x 512
      self.CBL_connect_branch_6 = CBL(512,3,2,"valid")

      #concat CBL_connect_branch_6 -- AttentionModule_neck_6_branch_3 , out: 19 x 19 x 1024

      #$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$

      #CBL_1 phase4 in : 19 x 19 x 1024 out : 19 x 19 x 512
      self.CBL_1_phase4 = CBL(512,1,1,"same")

      #CBL_2 phase4 in : 19 x 19 x 512 out : 19 x 19 x 1024
      self.CBL_2_phase4 = CBL(1024,3,1,"same")

      #CBL_3 phase4 in : 19 x 19 x 1024 out : 19 x 19 x 512
      self.CBL_3_phase4 = CBL(512,1,1,"same")

      #CBL_4 phase4 in : 19 x 19 x 512 out : 19 x 19 x 1024
      self.CBL_4_phase4 = CBL(1024,3,1,"same")

      #CBL_5 phase4 in : 19 x 19 x 1024 out : 19 x 19 x 512 
      self.CBL_5_phase4 = CBL(512,1,1,"same")

      
      #$#$#$#$#$#$#$#$#$#$#$#$#$# large output $#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$
      
      #TCBL_large in : 19 x 19 x 512 out : 20 x 20 x 512
      self.TCBL_large = TCBL(512,2,1,"valid")

      #prob info

      #CBL_prob_class_large_1 in : 20 x 20 x 512 out : 20 x 20 x 1024
      self.CBL_prob_class_large_1 = CBL(1024,3,1,"same")

      #CBL_prob_class_large_2 in : 20 x 20 x 1024 out : 20 x 20 x 1024
      self.CBL_prob_class_large_2 = CBL(1024,1,1,"same")

      #conv_prob_large in : 20 x 20 x 512 out : 20 x 20 x 1
      self.conv_prob_large = tf.keras.layers.Conv2D(filters=1,kernel_size=1,strides=1,padding="same",data_format="channels_last",activation="sigmoid")

      #conv_class_large in : 20 x 20 x 512 out : 20 x 20 x 80
      self.conv_class_large = tf.keras.layers.Conv2D(filters=80,kernel_size=1,strides=1,padding="same",data_format="channels_last",activation="sigmoid")


      #Reg info
      
      #CBL_left_center_large_1 in : 20 x 20 x 512 out : 20 x 20 x 1024
      self.CBL_left_center_large_1 = CBL(1024,3,1,"same")

      #CBL_left_center_large_2 in :  20 x 20 x 1024 out :  20 x 20 x 1024
      self.CBL_left_center_large_2 = CBL(1024,1,1,"same")

      #conv_pos_info_large in : 20 x 20 x 1024 out : 20 x 20 x 4
      self.conv_pos_info_large = tf.keras.layers.Conv2D(filters=4,kernel_size=1,strides=1,padding="same",data_format="channels_last",activation=tf.keras.layers.LeakyReLU())

      #concat conv_prob_large -- conv_pos_info_large -- conv_class_large , out: 20 x 20 x 85

      #$#$#$#$#$#$#$#$#$#$#$#$#$# large output $#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$
      
      
   def call(self,inputs,train_flag=True):

      #CBM1
      CBM_1 = self.CBM_1(inputs,train_flag)

      #CSP1
      CSP1 = self.CSP1(CBM_1,train_flag)

      #CSP2
      CSP2 = self.CSP2(CSP1,train_flag)

      #CSP8_branch_1 ----------------------------------------- branch 1
      CSP8_branch_1 = self.CSP8_branch_1(CSP2,train_flag)

      #CSP8_branch_2 ----------------------------------------- branch 2
      CSP8_branch_2 = self.CSP8_branch_2(CSP8_branch_1,train_flag)

      #CSP4
      CSP4 = self.CSP4(CSP8_branch_2,train_flag)

      #AttentionModule_neck_1
      AttentionModule_neck_1 = self.AttentionModule_neck_1(CSP4,train_flag)

      #AttentionModule_neck_2
      AttentionModule_neck_2 = self.AttentionModule_neck_2(AttentionModule_neck_1,train_flag)

      #AttentionModule_neck_3
      AttentionModule_neck_3 = self.AttentionModule_neck_3(AttentionModule_neck_2,train_flag)

      #AttentionModule_neck_4
      AttentionModule_neck_4 = self.AttentionModule_neck_4(AttentionModule_neck_3,train_flag)

      #AttentionModule_neck_5
      AttentionModule_neck_5 = self.AttentionModule_neck_5(AttentionModule_neck_4,train_flag)

      #AttentionModule_neck_6_branch_3 ----------------------------------------- branch 3
      AttentionModule_neck_6_branch_3 = self.AttentionModule_neck_6_branch_3(AttentionModule_neck_5,train_flag)

      #CBL_7_neck
      CBL_7_neck = self.CBL_7_neck(AttentionModule_neck_6_branch_3,train_flag)

      #TCBL1
      TCBL1 = self.TCBL1(CBL_7_neck,train_flag)

      #CBL_connect_branch_2
      CBL_connect_branch_2 = self.CBL_connect_branch_2(CSP8_branch_2,train_flag)

      #concat TCBL1 -- CBL_connect_branch_2
      concat_TCBL1_CBL_connect_branch_2 = tf.keras.layers.concatenate(inputs=[TCBL1,CBL_connect_branch_2],axis=-1)

      #CBL_1 phase1
      CBL_1_phase1 = self.CBL_1_phase1(concat_TCBL1_CBL_connect_branch_2,train_flag)

      #CBL_2_phase1
      CBL_2_phase1 = self.CBL_2_phase1(CBL_1_phase1,train_flag)

      #CBL_3_phase1
      CBL_3_phase1 = self.CBL_3_phase1(CBL_2_phase1,train_flag)

      #CBL_4_phase1
      CBL_4_phase1 = self.CBL_4_phase1(CBL_3_phase1,train_flag)

      #CBL_5_phase1_branch_4 ----------------------------------------- branch 4
      CBL_5_phase1_branch_4 = self.CBL_5_phase1_branch_4(CBL_4_phase1,train_flag)

      #CBL_6_phase1
      CBL_6_phase1 = self.CBL_6_phase1(CBL_5_phase1_branch_4,train_flag)

      #TCBL2
      TCBL2 = self.TCBL2(CBL_6_phase1,train_flag)

      #CBL_connect_branch_1
      CBL_connect_branch_1 = self.CBL_connect_branch_1(CSP8_branch_1,train_flag)

      #concat TCBL2 -- CBL_connect_branch_1 , out: 79 x 79 x 256
      concat_TCBL2_CBL_connect_branch_1 = tf.keras.layers.concatenate(inputs=[TCBL2,CBL_connect_branch_1],axis=-1)

      #CBL_1_phase2
      CBL_1_phase2 = self.CBL_1_phase2(concat_TCBL2_CBL_connect_branch_1,train_flag)

      #CBL_2_phase2
      CBL_2_phase2 = self.CBL_2_phase2(CBL_1_phase2,train_flag)

      #CBL_3_phase2
      CBL_3_phase2 = self.CBL_3_phase2(CBL_2_phase2,train_flag)

      #CBL_4_phase2
      CBL_4_phase2 = self.CBL_4_phase2(CBL_3_phase2,train_flag)

      #CBL_5_phase2_branch_5 ----------------------------------------- branch 5
      CBL_5_phase2_branch_5 = self.CBL_5_phase2_branch_5(CBL_4_phase2,train_flag)

      #$#$#$#$#$#$#$#$#$#$#$#$#$# small output $#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$

      #TCBL_small
      TCBL_small = self.TCBL_small(CBL_5_phase2_branch_5,train_flag)

      #prob info

      #CBL_prob_class_1 
      CBL_prob_class_small_1 = self.CBL_prob_class_small_1(TCBL_small,train_flag)

      #CBL_prob_class_2 in : 80 x 80 x 256 out : 80 x 80 x 256
      CBL_prob_class_small_2 = self.CBL_prob_class_small_2(CBL_prob_class_small_1,train_flag)

      #CBL_prob_small 
      conv_prob_small = self.conv_prob_small(CBL_prob_class_small_2)

      #CBL_class_small 
      conv_class_small = self.conv_class_small(CBL_prob_class_small_2)

      #reg info
      
      #CBL_left_center_small_1
      CBL_left_center_small_1 = self.CBL_left_center_small_1(TCBL_small,train_flag)

      #CBL_left_center_small_2
      CBL_left_center_small_2 = self.CBL_left_center_small_2(CBL_left_center_small_1,train_flag)

      #conv_pos_info_small
      conv_pos_info_small = self.conv_pos_info_small(CBL_left_center_small_2)

      #concat conv_prob_small -- conv_pos_info_small -- conv_class_small , out: 80 x 80 x 85
      output_small = tf.keras.layers.concatenate(inputs=[conv_prob_small,conv_pos_info_small,conv_class_small],axis=-1,name="output_small")

      #$#$#$#$#$#$#$#$#$#$#$#$#$# small output $#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$

      #CBL_connect_branch_5
      CBL_connect_branch_5 = self.CBL_connect_branch_5(CBL_5_phase2_branch_5,train_flag)

      #concat CBL_connect_branch_5 -- CBL_5_phase1_branch_4 
      concat_CBL_connect_branch_5_CBL_5_phase1_branch_4 = tf.keras.layers.concatenate(inputs=[CBL_connect_branch_5,CBL_5_phase1_branch_4],axis=-1)
      

      #CBL_1 phase3 
      CBL_1_phase3 = self.CBL_1_phase3(concat_CBL_connect_branch_5_CBL_5_phase1_branch_4,train_flag)

      #CBL_2 phase3 
      CBL_2_phase3 = self.CBL_2_phase3(CBL_1_phase3,train_flag)

      #CBL_3 phase3 
      CBL_3_phase3 = self.CBL_3_phase3(CBL_2_phase3,train_flag)

      #CBL_4 phase3 
      CBL_4_phase3 = self.CBL_4_phase3(CBL_3_phase3,train_flag)

      #CBL_5 phase3_branch_6 ----------------------------------------- branch 6
      CBL_5_phase3_branch_6  = self.CBL_5_phase3_branch_6(CBL_4_phase3,train_flag)

      #$#$#$#$#$#$#$#$#$#$#$#$#$# medium output $#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$
      
      #TCBL_medium
      TCBL_medium = self.TCBL_medium(CBL_5_phase3_branch_6,train_flag)

      #prob info

      #CBL_prob_class_medium_1 
      CBL_prob_class_medium_1 = self.CBL_prob_class_medium_1(TCBL_medium,train_flag)

      #CBL_prob_class_medium_2
      CBL_prob_class_medium_2 = self.CBL_prob_class_medium_2(CBL_prob_class_medium_1,train_flag)

      #conv_prob_medium 
      conv_prob_medium = self.conv_prob_medium(CBL_prob_class_medium_2)

      #conv_class_medium 
      conv_class_medium = self.conv_class_medium(CBL_prob_class_medium_2)


      #Reg info
      
      #CBL_left_center_medium_1 
      CBL_left_center_medium_1 = self.CBL_left_center_medium_1(TCBL_medium,train_flag)

      #CBL_left_center_medium_2 
      CBL_left_center_medium_2 = self.CBL_left_center_medium_2(CBL_left_center_medium_1,train_flag)

      #conv_pos_info_medium 
      conv_pos_info_medium = self.conv_pos_info_medium(CBL_left_center_medium_2)

      #concat conv_prob_medium -- conv_pos_info_medium -- conv_class_medium , out: 40 x 40 x 85
      output_medium = tf.keras.layers.concatenate(inputs=[conv_prob_medium,conv_pos_info_medium,conv_class_medium],axis=-1,name="output_medium")

      #$#$#$#$#$#$#$#$#$#$#$#$#$# medium output $#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$

      #CBL_connect_branch_6 
      CBL_connect_branch_6 = self.CBL_connect_branch_6(CBL_5_phase3_branch_6,train_flag)

      #concat CBL_connect_branch_6 -- AttentionModule_neck_6_branch_3
      concat_CBL_connect_branch_6_AttentionModule_neck_6_branch_3 = tf.keras.layers.concatenate(inputs=[CBL_connect_branch_6,AttentionModule_neck_6_branch_3],axis=-1)

      #CBL_1 phase4 
      CBL_1_phase4 = self.CBL_1_phase4(concat_CBL_connect_branch_6_AttentionModule_neck_6_branch_3,train_flag)

      #CBL_2 phase4 
      CBL_2_phase4 = self.CBL_2_phase4(CBL_1_phase4,train_flag)

      #CBL_3 phase4
      CBL_3_phase4 = self.CBL_3_phase4(CBL_2_phase4,train_flag)

      #CBL_4 phase4 
      CBL_4_phase4 = self.CBL_4_phase4(CBL_3_phase4,train_flag)

      #CBL_5 phase4 
      CBL_5_phase4 = self.CBL_5_phase4(CBL_4_phase4,train_flag)

      #$#$#$#$#$#$#$#$#$#$#$#$#$# large output $#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$
      
      #TCBL_large 
      TCBL_large = self.TCBL_large(CBL_5_phase4,train_flag)

      #prob info

      #CBL_prob_class_large_1 
      CBL_prob_class_large_1 = self.CBL_prob_class_large_1(TCBL_large,train_flag)

      #CBL_prob_class_large_2
      CBL_prob_class_large_2 = self.CBL_prob_class_large_2(CBL_prob_class_large_1,train_flag)

      #conv_prob_large 
      conv_prob_large = self.conv_prob_large(CBL_prob_class_large_2)

      #conv_class_large 
      conv_class_large = self.conv_class_large(CBL_prob_class_large_2)

      #Reg info
      
      #CBL_left_center_large_1 
      CBL_left_center_large_1 = self.CBL_left_center_large_1(TCBL_large,train_flag)

      #CBL_left_center_large_2 
      CBL_left_center_large_2 = self.CBL_left_center_large_2(CBL_left_center_large_1,train_flag)

      #conv_pos_info_large 
      conv_pos_info_large = self.conv_pos_info_large(CBL_left_center_large_2) 

      #concat conv_prob_large -- conv_pos_info_large -- conv_class_large , out: 20 x 20 x 85
      output_large = tf.keras.layers.concatenate(inputs=[conv_prob_large,conv_pos_info_large,conv_class_large],axis=-1,name="output_large")

      #$#$#$#$#$#$#$#$#$#$#$#$#$# large output $#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$

      return [output_large,output_medium,output_small]

   def graph_model(self,dim):

      x = tf.keras.layers.Input(shape=dim)
      
      return tf.keras.Model(inputs=x,outputs=self.call(x))
