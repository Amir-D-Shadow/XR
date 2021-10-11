import tensorflow as tf
from CBL import CBL
from CSPX import CSPX,rCSP
from SPP import SPP
from TCBL import TCBL
from CBS import CBS
from RESX import RESX



class alpha_model(tf.keras.Model):

   def __init__(self,**kwargs):

      #initialization
      super(alpha_model,self).__init__(**kwargs)

      #$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$
      
      #CBL_1 in : 640 x 640 x 3 out: 640 x 640 x 32
      filters=32
      kernel_size=3
      strides=1
      padding="same"

      self.CBL_1 = CBL(filters,kernel_size,strides,padding)

      #$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$

      #resX_1 in : 640 x 640 x 32 out: 319 x 319 x 64
      RESX_info = {}

      #CBL_1
      RESX_info["CBL_1"] = (64,3,2,"valid")

      #number of res unit
      RESX_info["num_of_res_unit"] = 1

      #res unit info
      res_unit_1 = {}
      res_unit_1["CBL_1"] = (32,1,1,"same")
      res_unit_1["CBL_2"] = (64,3,1,"same")
      RESX_info["res_unit_1"] = res_unit_1

      #resX_1
      self.resX_1 = RESX(RESX_info)

      #$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$

      #resX_2 in : 319 x 319 x 64 out : 159 x 159 x 128
      RESX_info = {}

      #CBL_1
      RESX_info["CBL_1"] = (128,3,2,"valid")

      #number of res unit
      RESX_info["num_of_res_unit"] = 2

      #res unit info
      res_unit_1 = {}
      res_unit_1["CBL_1"] = (64,1,1,"same")
      res_unit_1["CBL_2"] = (128,3,1,"same")
      RESX_info["res_unit_1"] = res_unit_1

      res_unit_2 = {}
      res_unit_2["CBL_1"] = (64,1,1,"same")
      res_unit_2["CBL_2"] = (128,3,1,"same")
      RESX_info["res_unit_2"] = res_unit_2

      #resX_2
      self.resX_2 = RESX(RESX_info)

      #$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$

      #resX_8_1 in : 159 x 159 x 128  out : 79 x 79 x 256 -- branch 1
      RESX_info = {}

      #num_of_res_unit
      RESX_info["num_of_res_unit"] = 8

      #CBL_1
      RESX_info["CBL_1"] = (256,3,2,"valid")

      #res_unit_info
      res_unit_1 = {}
      res_unit_1["CBL_1"] = (128,1,1,"same")
      res_unit_1["CBL_2"] = (256,3,1,"same")
      RESX_info["res_unit_1"] = res_unit_1

      res_unit_2 = {}
      res_unit_2["CBL_1"] = (128,1,1,"same")
      res_unit_2["CBL_2"] = (256,3,1,"same")
      RESX_info["res_unit_2"] = res_unit_2

      res_unit_3 = {}
      res_unit_3["CBL_1"] = (128,1,1,"same")
      res_unit_3["CBL_2"] = (256,3,1,"same")
      RESX_info["res_unit_3"] = res_unit_3

      res_unit_4 = {}
      res_unit_4["CBL_1"] = (128,1,1,"same")
      res_unit_4["CBL_2"] = (256,3,1,"same")
      RESX_info["res_unit_4"] = res_unit_4

      res_unit_5 = {}
      res_unit_5["CBL_1"] = (128,1,1,"same")
      res_unit_5["CBL_2"] = (256,3,1,"same")
      RESX_info["res_unit_5"] = res_unit_5

      res_unit_6 = {}
      res_unit_6["CBL_1"] = (128,1,1,"same")
      res_unit_6["CBL_2"] = (256,3,1,"same")
      RESX_info["res_unit_6"] = res_unit_6

      res_unit_7 = {}
      res_unit_7["CBL_1"] = (128,1,1,"same")
      res_unit_7["CBL_2"] = (256,3,1,"same")
      RESX_info["res_unit_7"] = res_unit_7

      res_unit_8 = {}
      res_unit_8["CBL_1"] = (128,1,1,"same")
      res_unit_8["CBL_2"] = (256,3,1,"same")
      RESX_info["res_unit_8"] = res_unit_8

      #resX_8_1 -- branch_1
      self.resX_8_branch_1 = RESX(RESX_info)

      #$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$

      #resX_8_2 in : 79 x 79 x 256 out : 39 x 39 x 512 --- branch_2
      RESX_info = {}

      #num_of_res_unit
      RESX_info["num_of_res_unit"] = 8

      #CBL_1
      RESX_info["CBL_1"] = (512,3,2,"valid")

      #res_unit_info
      res_unit_1 = {}
      res_unit_1["CBL_1"] = (256,1,1,"same")
      res_unit_1["CBL_2"] = (512,3,1,"same")
      RESX_info["res_unit_1"] = res_unit_1

      res_unit_2 = {}
      res_unit_2["CBL_1"] = (256,1,1,"same")
      res_unit_2["CBL_2"] = (512,3,1,"same")
      RESX_info["res_unit_2"] = res_unit_2

      res_unit_3 = {}
      res_unit_3["CBL_1"] = (256,1,1,"same")
      res_unit_3["CBL_2"] = (512,3,1,"same")
      RESX_info["res_unit_3"] = res_unit_3

      res_unit_4 = {}
      res_unit_4["CBL_1"] = (256,1,1,"same")
      res_unit_4["CBL_2"] = (512,3,1,"same")
      RESX_info["res_unit_4"] = res_unit_4

      res_unit_5 = {}
      res_unit_5["CBL_1"] = (256,1,1,"same")
      res_unit_5["CBL_2"] = (512,3,1,"same")
      RESX_info["res_unit_5"] = res_unit_5

      res_unit_6 = {}
      res_unit_6["CBL_1"] = (256,1,1,"same")
      res_unit_6["CBL_2"] = (512,3,1,"same")
      RESX_info["res_unit_6"] = res_unit_6

      res_unit_7 = {}
      res_unit_7["CBL_1"] = (256,1,1,"same")
      res_unit_7["CBL_2"] = (512,3,1,"same")
      RESX_info["res_unit_7"] = res_unit_7

      res_unit_8 = {}
      res_unit_8["CBL_1"] = (256,1,1,"same")
      res_unit_8["CBL_2"] = (512,3,1,"same")
      RESX_info["res_unit_8"] = res_unit_8
      
      #resX_8_2 -- branch_2
      self.resX_8_branch_2 = RESX(RESX_info)

      #$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$

      #resX_4 in : 39 x 39 x 512 out : 19 x 19 x 1024
      RESX_info = {}

      #num_of_res_unit
      RESX_info["num_of_res_unit"] = 4

      #CBL_1
      RESX_info["CBL_1"] = (1024,3,2,"valid")

      #res_unit_info
      res_unit_1 = {}
      res_unit_1["CBL_1"] = (512,1,1,"same")
      res_unit_1["CBL_2"] = (1024,3,1,"same")
      RESX_info["res_unit_1"] = res_unit_1

      res_unit_2 = {}
      res_unit_2["CBL_1"] = (512,1,1,"same")
      res_unit_2["CBL_2"] = (1024,3,1,"same")
      RESX_info["res_unit_2"] = res_unit_2

      res_unit_3 = {}
      res_unit_3["CBL_1"] = (512,1,1,"same")
      res_unit_3["CBL_2"] = (1024,3,1,"same")
      RESX_info["res_unit_3"] = res_unit_3

      res_unit_4 = {}
      res_unit_4["CBL_1"] = (512,1,1,"same")
      res_unit_4["CBL_2"] = (1024,3,1,"same")
      RESX_info["res_unit_4"] = res_unit_4

      #resX_4 
      self.resX_4 = RESX(RESX_info)

      #$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$

      #CBL_1 neck in : 19 x 19 x 1024 out : 19 x 19 x 512
      self.CBL_1_neck = CBL(512,3,1,"same")

      #CBL_2 neck in : 19 x 19 x 512 out : 19 x 19 x 1024
      self.CBL_2_neck = CBL(1024,1,1,"same")

      #CBL_3 neck in : 19 x 19 x 1024 out : 19 x 19 x 512
      self.CBL_3_neck = CBL(512,3,1,"same")

      #SPP_neck in:19 x 19 x 512 out: 19 x 19 x 2048
      self.SPP_neck = SPP()

      #CBL_4 neck in : 19 x 19 x 2048 out : 19 x 19 x 512
      self.CBL_4_neck = CBL(512,3,1,"same")

      #CBL_5 neck in : 19 x 19 x 512 out : 19 x 19 x 1024
      self.CBL_5_neck = CBL(1024,1,1,"same")

      #CBL_6 branch 3 in : 19 x 19 x 1024 out : 19 x 19 x 512 -- branch 3
      self.CBL_6_branch_3 = CBL(512,3,1,"same")

      #TCBL_connect_branch_3 in : 19 x 19 x 512 out : 20 x 20 x 512
      self.TCBL_connect_branch_3 = TCBL(512,2,1,"valid")

      #$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$
      
      #CBL 7 neck in : 19 x 19 x 512 out : 19 x 19 x 512
      self.CBL_7_neck = CBL(512,1,1,"same")

      #TCBL1,2,3,4,5,6,7,8,9,10 in : 19 x 19 x 256 out: 40 x 40 x 256
      self.TCBL1 = TCBL(256,3,1,"valid")

      self.TCBL2 = TCBL(256,3,1,"valid")

      self.TCBL3 = TCBL(256,3,1,"valid")

      self.TCBL4 = TCBL(256,3,1,"valid")

      self.TCBL5 = TCBL(256,3,1,"valid")

      self.TCBL6 = TCBL(256,3,1,"valid")

      self.TCBL7 = TCBL(256,3,1,"valid")

      self.TCBL8 = TCBL(256,3,1,"valid")

      self.TCBL9 = TCBL(256,3,1,"valid")

      self.TCBL10 = TCBL(256,4,1,"valid")

      #TCBL_connect_branch_2 in : 39 x 39 x 512 out: 40 x 40 x 256
      self.TCBL_connect_branch_2 = TCBL(256,2,1,"valid")

      #concat TCBL10 -- TCBL_connect_branch_2 , out: 40 x 40 x 512

      #$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$

      #CBL_1 phase1 in : 40 x 40 x 512 out : 40 x 40 x 256
      self.CBL_1_phase1 = CBL(256,3,1,"same")

      #CBL_2 phase1 in : 40 x 40 x 256 out : 40 x 40 x 512
      self.CBL_2_phase1 = CBL(512,1,1,"same")

      #CBL_3 phase1 in : 40 x 40 x 512 out : 40 x 40 x 256
      self.CBL_3_phase1 = CBL(256,3,1,"same")

      #CBL_4 phase1 in : 40 x 40 x 256 out : 40 x 40 x 512
      self.CBL_4_phase1 = CBL(512,1,1,"same")

      #CBL_5 phase1_branch_4 in : 40 x 40 x 512 out : 40 x 40 x 256 -- branch 4
      self.CBL_5_phase1_branch_4 = CBL(256,3,1,"same")

      #$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$
      
      #CBL_6 phase1 in : 40 x 40 x 256 out : 40 x 40 x 256
      self.CBL_6_phase1 = CBL(256,1,1,"same")

      #TCBL2 in : 40 x 40 x 128 out: 80 x 80 x 128
      self.TCBL1a = TCBL(128,3,1,"valid")

      self.TCBL2a = TCBL(128,3,1,"valid")

      self.TCBL3a = TCBL(128,3,1,"valid")

      self.TCBL4a = TCBL(128,3,1,"valid")

      self.TCBL5a = TCBL(128,3,1,"valid")

      self.TCBL6a = TCBL(128,3,1,"valid")

      self.TCBL7a = TCBL(128,3,1,"valid")

      self.TCBL8a = TCBL(128,3,1,"valid")

      self.TCBL9a = TCBL(128,3,1,"valid")

      self.TCBL10a = TCBL(128,3,1,"valid")

      self.TCBL11a = TCBL(128,3,1,"valid")

      self.TCBL12a = TCBL(128,3,1,"valid")

      self.TCBL13a = TCBL(128,3,1,"valid")

      self.TCBL14a = TCBL(128,3,1,"valid")

      self.TCBL15a = TCBL(128,3,1,"valid")

      self.TCBL16a = TCBL(128,3,1,"valid")

      self.TCBL17a = TCBL(128,3,1,"valid")

      self.TCBL18a = TCBL(128,3,1,"valid")

      self.TCBL19a = TCBL(128,3,1,"valid")

      self.TCBL20a = TCBL(128,3,1,"valid")

      #TCBL_connect_branch_1 in : 79 x 79 x 256 out: 80 x 80 x 128
      self.TCBL_connect_branch_1 = TCBL(128,2,1,"valid")

      #concat TCBL2 -- TCBL_connect_branch_1 , out: 80 x 80 x 256

      #$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$

      #CBL_1 phase2 in : 80 x 80 x 256 out : 80 x 80 x 128
      self.CBL_1_phase2 = CBL(128,3,1,"same")

      #CBL_2 phase2 in : 80 x 80 x 128 out : 80 x 80 x 256
      self.CBL_2_phase2 = CBL(256,1,1,"same")

      #CBL_3 phase2 in : 80 x 80 x 256 out : 80 x 80 x 128
      self.CBL_3_phase2 = CBL(128,3,1,"same")

      #CBL_4 phase2 in : 80 x 80 x 128 out : 80 x 80 x 256
      self.CBL_4_phase2 = CBL(256,1,1,"same")

      #CBL_5 phase2_branch_5 in : 80 x 80 x 256 out : 80 x 80 x 128 -- branch 5
      self.CBL_5_phase2_branch_5 = CBL(128,3,1,"same")

      #$#$#$#$#$#$#$#$#$#$#$#$#$# small output $#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$

      #CBL_small in : 80 x 80 x 128 out : 80 x 80 x 256
      self.CBL_small = CBL(256,1,1,"same")

      #prob info

      #CBL_prob_class_small_1 in : 80 x 80 x 256 out : 80 x 80 x 128
      self.CBL_prob_class_small_1 = CBL(128,3,1,"same")

      #CBL_prob_class_small_2 in : 80 x 80 x 128 out : 80 x 80 x 256
      self.CBL_prob_class_small_2 = CBL(256,1,1,"same")

      #conv_prob_small in : 80 x 80 x 256 out : 80 x 80 x 1
      self.conv_prob_small = tf.keras.layers.Conv2D(filters=1,kernel_size=3,strides=1,padding="same",data_format="channels_last",activation="sigmoid")

      #conv_class_small in : 80 x 80 x 256 out : 80 x 80 x 20
      self.conv_class_small = tf.keras.layers.Conv2D(filters=20,kernel_size=3,strides=1,padding="same",data_format="channels_last",activation=tf.keras.layers.Softmax(axis=-1))
      
      #Reg info
      
      #CBL_left_center_small_1 in : 80 x 80 x 256 out : 80 x 80 x 128
      self.CBL_left_center_small_1 = CBL(128,3,1,"same")

      #CBL_left_center_small_2 in : 80 x 80 x 128 out : 80 x 80 x 256
      self.CBL_left_center_small_2 = CBL(256,1,1,"same")

      #conv_pos_info_small in : 80 x 80 x 256 out : 80 x 80 x 4
      self.conv_pos_info_small = tf.keras.layers.Conv2D(filters=4,kernel_size=3,strides=1,padding="same",data_format="channels_last",activation=tf.keras.layers.LeakyReLU())

      #concat conv_prob_small -- conv_pos_info_small -- conv_class_small , out: 80 x 80 x 25

      #$#$#$#$#$#$#$#$#$#$#$#$#$# small output $#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$

      #CBL_connect_branch_5 in : 80 x 80 x 128 out : 40 x 40 x 256
      self.CBL_connect_branch_5 = CBL(256,2,2,"valid")

      #concat CBL_connect_branch_5 -- CBL_5_phase1_branch_4 , out: 40 x 40 x 512

      #$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$

      #CBL_1 phase3 in : 40 x 40 x 512 out : 40 x 40 x 256
      self.CBL_1_phase3 = CBL(256,3,1,"same")

      #CBL_2 phase3 in : 40 x 40 x 256 out : 40 x 40 x 512
      self.CBL_2_phase3 = CBL(512,1,1,"same")

      #CBL_3 phase3 in : 40 x 40 x 512 out : 40 x 40 x 256
      self.CBL_3_phase3 = CBL(256,3,1,"same")

      #CBL_4 phase3 in : 40 x 40 x 256 out : 40 x 40 x 512
      self.CBL_4_phase3 = CBL(512,1,1,"same")

      #CBL_5 phase3_branch_6 in : 40 x 40 x 512 out : 40 x 40 x 256 -- branch 6
      self.CBL_5_phase3_branch_6 = CBL(256,3,1,"same")

      #$#$#$#$#$#$#$#$#$#$#$#$#$# medium output $#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$

      #CBL_medium in : 40 x 40 x 256 out : 40 x 40 x 512
      self.CBL_medium = CBL(512,1,1,"same")

      #prob info

      #CBL_prob_class_medium_1 in : 40 x 40 x 512 out : 40 x 40 x 256
      self.CBL_prob_class_medium_1 = CBL(256,3,1,"same")

      #CBL_prob_class_medium_2 in : 40 x 40 x 256 out : 40 x 40 x 512
      self.CBL_prob_class_medium_2 = CBL(512,1,1,"same")

      #conv_prob_medium in : 40 x 40 x 512 out : 40 x 40 x 1
      self.conv_prob_medium = tf.keras.layers.Conv2D(filters=1,kernel_size=3,strides=1,padding="same",data_format="channels_last",activation="sigmoid")

      #conv_class_medium in : 40 x 40 x 512 out : 40 x 40 x 20
      self.conv_class_medium = tf.keras.layers.Conv2D(filters=20,kernel_size=3,strides=1,padding="same",data_format="channels_last",activation=tf.keras.layers.Softmax(axis=-1))


      #Reg info
      
      #CBL_left_center_medium_1 in : 40 x 40 x 512 out : 40 x 40 x 256
      self.CBL_left_center_medium_1 = CBL(256,3,1,"same")

      #CBL_left_center_medium_2 in :  40 x 40 x 256 out :  40 x 40 x 512
      self.CBL_left_center_medium_2 = CBL(512,1,1,"same")

      #conv_pos_info_medium in : 40 x 40 x 512 out : 40 x 40 x 4
      self.conv_pos_info_medium = tf.keras.layers.Conv2D(filters=4,kernel_size=1,strides=1,padding="same",data_format="channels_last",activation=tf.keras.layers.LeakyReLU())

      #concat conv_prob_medium -- conv_pos_info_medium -- conv_class_medium , out: 40 x 40 x 25

      #$#$#$#$#$#$#$#$#$#$#$#$#$# medium output $#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$

      #CBL_connect_branch_6 in : 40 x 40 x 256 out : 20 x 20 x 512
      self.CBL_connect_branch_6 = CBL(512,2,2,"valid")

      #concat CBL_connect_branch_6 -- CBL_6_branch_3 , out: 20 x 20 x 1024

      #$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$

      #CBL_1 phase4 in : 20 x 20 x 1024 out : 20 x 20 x 512
      self.CBL_1_phase4 = CBL(512,3,1,"same")

      #CBL_2 phase4 in : 20 x 20 x 512 out : 20 x 20 x 1024
      self.CBL_2_phase4 = CBL(1024,1,1,"same")

      #CBL_3 phase4 in : 20 x 20 x 1024 out : 20 x 20 x 512
      self.CBL_3_phase4 = CBL(512,3,1,"same")

      #CBL_4 phase4 in : 20 x 20 x 512 out : 20 x 20 x 1024
      self.CBL_4_phase4 = CBL(1024,1,1,"same")

      #CBL_5 phase4 in : 20 x 20 x 1024 out : 20 x 20 x 512 
      self.CBL_5_phase4 = CBL(512,3,1,"same")

      
      #$#$#$#$#$#$#$#$#$#$#$#$#$# large output $#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$

      self.TCBL_connect_REX4 = TCBL(512,2,1,"valid")
      
      #CBL_large in : 20 x 20 x 512 out : 20 x 20 x 1024
      self.CBL_large = CBL(1024,1,1,"same")

      #prob info

      #CBL_prob_class_large_1 in : 20 x 20 x 1024 out : 20 x 20 x 512
      self.CBL_prob_class_large_1 = CBL(512,3,1,"same")

      #CBL_prob_class_large_2 in : 20 x 20 x 512 out : 20 x 20 x 1024
      self.CBL_prob_class_large_2 = CBL(1024,1,1,"same")

      #conv_prob_large in : 20 x 20 x 1024 out : 20 x 20 x 1
      self.conv_prob_large = tf.keras.layers.Conv2D(filters=1,kernel_size=3,strides=1,padding="same",data_format="channels_last",activation="sigmoid")

      #conv_class_large in : 20 x 20 x 1024 out : 20 x 20 x 20
      self.conv_class_large = tf.keras.layers.Conv2D(filters=20,kernel_size=3,strides=1,padding="same",data_format="channels_last",activation=tf.keras.layers.Softmax(axis=-1))


      #Reg info
      
      #CBL_left_center_large_1 in : 20 x 20 x 1024 out : 20 x 20 x 512
      self.CBL_left_center_large_1 = CBL(512,3,1,"same")

      #CBL_left_center_large_2 in :  20 x 20 x 512 out :  20 x 20 x 1024
      self.CBL_left_center_large_2 = CBL(1024,1,1,"same")

      #conv_pos_info_large in : 20 x 20 x 1024 out : 20 x 20 x 4
      self.conv_pos_info_large = tf.keras.layers.Conv2D(filters=4,kernel_size=3,strides=1,padding="same",data_format="channels_last",activation=tf.keras.layers.LeakyReLU())

      #concat conv_prob_large -- conv_pos_info_large -- conv_class_large , out: 20 x 20 x 25

      #$#$#$#$#$#$#$#$#$#$#$#$#$# large output $#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$
      
      
   def call(self,inputs,train_flag=True):

      #CBL1
      CBL_1 = self.CBL_1(inputs,train_flag)

      #resX_1
      resX_1 = self.resX_1(CBL_1,train_flag)

      #resX_2
      resX_2 = self.resX_2(resX_1,train_flag)

      #resX_8_branch_1 ----------------------------------------- branch 1
      resX_8_branch_1 = self.resX_8_branch_1(resX_2,train_flag)

      #resX_8_branch_2 ----------------------------------------- branch 2
      resX_8_branch_2 = self.resX_8_branch_2(resX_8_branch_1,train_flag)

      #resX_4
      resX_4 = self.resX_4(resX_8_branch_2,train_flag)

      #CBL_1_neck
      CBL_1_neck = self.CBL_1_neck(resX_4,train_flag)

      #CBL_2_neck
      CBL_2_neck = self.CBL_2_neck(CBL_1_neck,train_flag)

      #CBL_3_neck
      CBL_3_neck = self.CBL_3_neck(CBL_2_neck,train_flag)

      #SPP_neck
      SPP_neck = self.SPP_neck(CBL_3_neck)

      #CBL_5_neck
      CBL_5_neck = self.CBL_5_neck(SPP_neck,train_flag)

      #CBL_6_branch_3 ----------------------------------------- branch 3
      CBL_6_branch_3 = self.CBL_6_branch_3(CBL_5_neck,train_flag)

      #CBL_7_neck
      CBL_7_neck = self.CBL_7_neck(CBL_6_branch_3,train_flag)

      #TCBL1,2,3,4,5,6,7,8,9,10
      TCBL1 = self.TCBL1(CBL_7_neck,train_flag)

      TCBL2 = self.TCBL2(TCBL1,train_flag)

      TCBL3 = self.TCBL3(TCBL2,train_flag)

      TCBL4 = self.TCBL4(TCBL3,train_flag)

      TCBL5 = self.TCBL5(TCBL4,train_flag)

      TCBL6 = self.TCBL6(TCBL5,train_flag)

      TCBL7 = self.TCBL7(TCBL6,train_flag)

      TCBL8 = self.TCBL8(TCBL7,train_flag)

      TCBL9 = self.TCBL9(TCBL8,train_flag)

      TCBL10 = self.TCBL10(TCBL9,train_flag)

      #TCBL_connect_branch_2
      TCBL_connect_branch_2 = self.TCBL_connect_branch_2(resX_8_branch_2,train_flag)

      #concat TCBL10 -- TCBL_connect_branch_2
      concat_TCBL10_TCBL_connect_branch_2 = tf.keras.layers.concatenate(inputs=[TCBL10,TCBL_connect_branch_2],axis=-1)

      #CBL_1 phase1
      CBL_1_phase1 = self.CBL_1_phase1(concat_TCBL10_TCBL_connect_branch_2,train_flag)

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

      #TCBL 1 -20 a
      TCBL1a = self.TCBL1a(CBL_6_phase1,train_flag)

      TCBL2a = self.TCBL2a(TCBL1a,train_flag)

      TCBL3a = self.TCBL3a(TCBL2a,train_flag)

      TCBL4a = self.TCBL4a(TCBL3a,train_flag)

      TCBL5a = self.TCBL5a(TCBL4a,train_flag)

      TCBL6a = self.TCBL6a(TCBL5a,train_flag)

      TCBL7a = self.TCBL7a(TCBL6a,train_flag)

      TCBL8a = self.TCBL8a(TCBL7a,train_flag)

      TCBL9a = self.TCBL9a(TCBL8a,train_flag)

      TCBL10a = self.TCBL10a(TCBL9a,train_flag)

      TCBL11a = self.TCBL11a(TCBL10a,train_flag)

      TCBL12a = self.TCBL12a(TCBL11a,train_flag)

      TCBL13a = self.TCBL13a(TCBL12a,train_flag)

      TCBL14a = self.TCBL14a(TCBL13a,train_flag)

      TCBL15a = self.TCBL15a(TCBL14a,train_flag)

      TCBL16a = self.TCBL16a(TCBL15a,train_flag)

      TCBL17a = self.TCBL17a(TCBL16a,train_flag)

      TCBL18a = self.TCBL18a(TCBL17a,train_flag)

      TCBL19a = self.TCBL19a(TCBL18a,train_flag)

      TCBL20a = self.TCBL20a(TCBL19a,train_flag)

      #TCBL_connect_branch_1
      TCBL_connect_branch_1 = self.TCBL_connect_branch_1(resX_8_branch_1,train_flag)

      #concat TCBL20a -- TCBL_connect_branch_1 , out: 79 x 79 x 256
      concat_TCBL20a_TCBL_connect_branch_1 = tf.keras.layers.concatenate(inputs=[TCBL20a,TCBL_connect_branch_1],axis=-1)

      #CBL_1_phase2
      CBL_1_phase2 = self.CBL_1_phase2(concat_TCBL20a_TCBL_connect_branch_1,train_flag)

      #CBL_2_phase2
      CBL_2_phase2 = self.CBL_2_phase2(CBL_1_phase2,train_flag)

      #CBL_3_phase2
      CBL_3_phase2 = self.CBL_3_phase2(CBL_2_phase2,train_flag)

      #CBL_4_phase2
      CBL_4_phase2 = self.CBL_4_phase2(CBL_3_phase2,train_flag)

      #CBL_5_phase2_branch_5 ----------------------------------------- branch 5
      CBL_5_phase2_branch_5 = self.CBL_5_phase2_branch_5(CBL_4_phase2,train_flag)

      #$#$#$#$#$#$#$#$#$#$#$#$#$# small output $#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$

      #CBL_small
      CBL_small = self.CBL_small(CBL_5_phase2_branch_5,train_flag)

      ########################### auxiliary output small 0.3 #################################
      
      #auxiliary output 0.3
      CBL_small_03 = self.CBL_small(TCBL_connect_branch_1,train_flag)

      #prob info

      #CBL_prob_class_1_03 
      CBL_prob_class_small_1_03 = self.CBL_prob_class_small_1(CBL_small_03,train_flag)

      #CBL_prob_class_2 in : 80 x 80 x 256 out : 80 x 80 x 256
      CBL_prob_class_small_2_03 = self.CBL_prob_class_small_2(CBL_prob_class_small_1_03,train_flag)

      #CBL_prob_small 
      conv_prob_small_03 = self.conv_prob_small(CBL_prob_class_small_2_03)

      #CBL_class_small 
      conv_class_small_03 = self.conv_class_small(CBL_prob_class_small_2_03)

      #reg info
      
      #CBL_left_center_small_1_03
      CBL_left_center_small_1_03 = self.CBL_left_center_small_1(CBL_small_03,train_flag)

      #CBL_left_center_small_2_03
      CBL_left_center_small_2_03 = self.CBL_left_center_small_2(CBL_left_center_small_1_03,train_flag)

      #conv_pos_info_small
      conv_pos_info_small_03 = self.conv_pos_info_small(CBL_left_center_small_2_03)

      #concat conv_prob_small -- conv_pos_info_small -- conv_class_small , out: 80 x 80 x 85
      output_small_03 = tf.keras.layers.concatenate(inputs=[conv_prob_small_03,conv_pos_info_small_03,conv_class_small_03],axis=-1,name="output_small_03")

      
      ########################### auxiliary output small 0.3 #################################
      
      #prob info

      #CBL_prob_class_1 
      CBL_prob_class_small_1 = self.CBL_prob_class_small_1(CBL_small,train_flag)

      #CBL_prob_class_2 in : 80 x 80 x 256 out : 80 x 80 x 256
      CBL_prob_class_small_2 = self.CBL_prob_class_small_2(CBL_prob_class_small_1,train_flag)

      #CBL_prob_small 
      conv_prob_small = self.conv_prob_small(CBL_prob_class_small_2)

      #CBL_class_small 
      conv_class_small = self.conv_class_small(CBL_prob_class_small_2)

      #reg info
      
      #CBL_left_center_small_1
      CBL_left_center_small_1 = self.CBL_left_center_small_1(CBL_small,train_flag)

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
      
      #CBL_medium
      CBL_medium = self.CBL_medium(CBL_5_phase3_branch_6,train_flag)

      ########################### auxiliary output medium 0.3 #################################
      
      #auxiliary output 0.3
      CBL_medium_03 = self.CBL_medium(TCBL_connect_branch_2,train_flag)

      #CBL_prob_class_medium_1 
      CBL_prob_class_medium_1_03 = self.CBL_prob_class_medium_1(CBL_medium_03,train_flag)

      #CBL_prob_class_medium_2
      CBL_prob_class_medium_2_03 = self.CBL_prob_class_medium_2(CBL_prob_class_medium_1_03,train_flag)

      #conv_prob_medium 
      conv_prob_medium_03 = self.conv_prob_medium(CBL_prob_class_medium_2_03)

      #conv_class_medium 
      conv_class_medium_03 = self.conv_class_medium(CBL_prob_class_medium_2_03)


      #Reg info
      
      #CBL_left_center_medium_1 
      CBL_left_center_medium_1_03 = self.CBL_left_center_medium_1(CBL_medium_03,train_flag)

      #CBL_left_center_medium_2 
      CBL_left_center_medium_2_03 = self.CBL_left_center_medium_2(CBL_left_center_medium_1_03,train_flag)

      #conv_pos_info_medium 
      conv_pos_info_medium_03 = self.conv_pos_info_medium(CBL_left_center_medium_2_03)

      #concat conv_prob_medium -- conv_pos_info_medium -- conv_class_medium , out: 40 x 40 x 85
      output_medium_03 = tf.keras.layers.concatenate(inputs=[conv_prob_medium_03,conv_pos_info_medium_03,conv_class_medium_03],axis=-1,name="output_medium_03")

      
      ########################### auxiliary output medium 0.3 #################################

      #prob info

      #CBL_prob_class_medium_1 
      CBL_prob_class_medium_1 = self.CBL_prob_class_medium_1(CBL_medium,train_flag)

      #CBL_prob_class_medium_2
      CBL_prob_class_medium_2 = self.CBL_prob_class_medium_2(CBL_prob_class_medium_1,train_flag)

      #conv_prob_medium 
      conv_prob_medium = self.conv_prob_medium(CBL_prob_class_medium_2)

      #conv_class_medium 
      conv_class_medium = self.conv_class_medium(CBL_prob_class_medium_2)


      #Reg info
      
      #CBL_left_center_medium_1 
      CBL_left_center_medium_1 = self.CBL_left_center_medium_1(CBL_medium,train_flag)

      #CBL_left_center_medium_2 
      CBL_left_center_medium_2 = self.CBL_left_center_medium_2(CBL_left_center_medium_1,train_flag)

      #conv_pos_info_medium 
      conv_pos_info_medium = self.conv_pos_info_medium(CBL_left_center_medium_2)

      #concat conv_prob_medium -- conv_pos_info_medium -- conv_class_medium , out: 40 x 40 x 85
      output_medium = tf.keras.layers.concatenate(inputs=[conv_prob_medium,conv_pos_info_medium,conv_class_medium],axis=-1,name="output_medium")

      #$#$#$#$#$#$#$#$#$#$#$#$#$# medium output $#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$

      #TCBL_connect_branch_3
      TCBL_connect_branch_3 = self.TCBL_connect_branch_3(CBL_6_branch_3,train_flag)

      #CBL_connect_branch_6 
      CBL_connect_branch_6 = self.CBL_connect_branch_6(CBL_5_phase3_branch_6,train_flag)

      #concat CBL_connect_branch_6 -- TCBL_connect_branch_3
      concat_CBL_connect_branch_6_TCBL_connect_branch_3 = tf.keras.layers.concatenate(inputs=[CBL_connect_branch_6,TCBL_connect_branch_3],axis=-1)

      #CBL_1 phase4 
      CBL_1_phase4 = self.CBL_1_phase4(concat_CBL_connect_branch_6_TCBL_connect_branch_3,train_flag)

      #CBL_2 phase4 
      CBL_2_phase4 = self.CBL_2_phase4(CBL_1_phase4,train_flag)

      #CBL_3 phase4
      CBL_3_phase4 = self.CBL_3_phase4(CBL_2_phase4,train_flag)

      #CBL_4 phase4 
      CBL_4_phase4 = self.CBL_4_phase4(CBL_3_phase4,train_flag)

      #CBL_5 phase4 
      CBL_5_phase4 = self.CBL_5_phase4(CBL_4_phase4,train_flag)

      #$#$#$#$#$#$#$#$#$#$#$#$#$# large output $#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$
      
      #CBL_large 
      CBL_large = self.CBL_large(CBL_5_phase4,train_flag)

      ########################### auxiliary output large 0.3 #################################

      TCBL_connect_REX4 = self.TCBL_connect_REX4(resX_4,train_flag)

      #auxiliary output 0.3
      CBL_large_03 = self.CBL_large(TCBL_connect_REX4,train_flag)

      #CBL_prob_class_large_1 
      CBL_prob_class_large_1_03 = self.CBL_prob_class_large_1(CBL_large_03,train_flag)

      #CBL_prob_class_large_2
      CBL_prob_class_large_2_03 = self.CBL_prob_class_large_2(CBL_prob_class_large_1_03,train_flag)

      #conv_prob_large 
      conv_prob_large_03 = self.conv_prob_large(CBL_prob_class_large_2_03)

      #conv_class_large 
      conv_class_large_03 = self.conv_class_large(CBL_prob_class_large_2_03)

      #Reg info
      
      #CBL_left_center_large_1 
      CBL_left_center_large_1_03 = self.CBL_left_center_large_1(CBL_large_03,train_flag)

      #CBL_left_center_large_2 
      CBL_left_center_large_2_03 = self.CBL_left_center_large_2(CBL_left_center_large_1_03,train_flag)

      #conv_pos_info_large 
      conv_pos_info_large_03 = self.conv_pos_info_large(CBL_left_center_large_2_03) 

      #concat conv_prob_large -- conv_pos_info_large -- conv_class_large , out: 20 x 20 x 85
      output_large_03 = tf.keras.layers.concatenate(inputs=[conv_prob_large_03,conv_pos_info_large_03,conv_class_large_03],axis=-1,name="output_large_03")

      
      ########################### auxiliary output large 0.3 #################################

      #prob info

      #CBL_prob_class_large_1 
      CBL_prob_class_large_1 = self.CBL_prob_class_large_1(CBL_large,train_flag)

      #CBL_prob_class_large_2
      CBL_prob_class_large_2 = self.CBL_prob_class_large_2(CBL_prob_class_large_1,train_flag)

      #conv_prob_large 
      conv_prob_large = self.conv_prob_large(CBL_prob_class_large_2)

      #conv_class_large 
      conv_class_large = self.conv_class_large(CBL_prob_class_large_2)

      #Reg info
      
      #CBL_left_center_large_1 
      CBL_left_center_large_1 = self.CBL_left_center_large_1(CBL_large,train_flag)

      #CBL_left_center_large_2 
      CBL_left_center_large_2 = self.CBL_left_center_large_2(CBL_left_center_large_1,train_flag)

      #conv_pos_info_large 
      conv_pos_info_large = self.conv_pos_info_large(CBL_left_center_large_2) 

      #concat conv_prob_large -- conv_pos_info_large -- conv_class_large , out: 20 x 20 x 85
      output_large = tf.keras.layers.concatenate(inputs=[conv_prob_large,conv_pos_info_large,conv_class_large],axis=-1,name="output_large")

      #$#$#$#$#$#$#$#$#$#$#$#$#$# large output $#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$

      return [output_large,output_medium,output_small,output_large_03,output_medium_03,output_small_03]

   def graph_model(self,dim):

      x = tf.keras.layers.Input(shape=dim)
      
      return tf.keras.Model(inputs=x,outputs=self.call(x))
