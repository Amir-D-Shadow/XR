import tensorflow as tf
import tensorflow.keras.backend as K
import numpy  as np
from matplotlib import pyplot as plt
from AttentionModule_lambda_version import AttentionModule
from Fourier_Attention_Module import FourierAttentionModule

class alpha_loss(tf.keras.losses.Loss):

  def __init__(self,gamma = 2,**kwargs):

    super(alpha_loss,self).__init__(**kwargs)

    self.gamma = gamma

  def call(self,y_true,y_pred):

    """
    y_true -- (batch_size,H,W,info) -- info [prob,x_left,y_left,x_center,y_center,class]
    y_pred -- (batch_size,H,W,info) -- info [prob,x_left,y_left,x_center,y_center,class]

    """

    #get object mask
    object_mask = K.cast(y_true[:,:,:,0:1],K.dtype(y_pred))

    object_mask_bool = K.cast(object_mask,dtype=tf.bool)

    #get ignore mask
    ignore_mask  = tf.TensorArray(K.dtype(y_pred),size=1,dynamic_size=True)

    ignore_mask = get_ignore_mask(ignore_mask,y_true,y_pred,object_mask_bool)

    #get prob
    prob_true = y_true[:,:,:,0:1]
    prob_pred = y_pred[:,:,:,0:1]

    prob_true = K.cast(prob_true,K.dtype(prob_pred))

    #get class
    class_true = y_true[:,:,:,5:]
    class_pred = y_pred[:,:,:,5:]

    class_true = K.cast(class_true,K.dtype(class_pred))

    #****************** Focal loss ******************

    #get batch size
    m = K.cast(K.shape(y_pred)[0],K.dtype(y_pred))

    #clip the prediction
    #prob_pred = K.clip(prob_pred,min_value = 0.0, max_value = 1.0)
    #class_pred = K.clip(class_pred,min_value = 0.0, max_value = 1.0)

    #prob focal loss
    loss_tensor_prob =  - ( (1 - prob_pred[:,:,:,:])**self.gamma ) * prob_true[:,:,:,:] * tf.math.log( prob_pred[:,:,:,:] + 1e-18 ) - ( prob_pred[:,:,:,:] ** self.gamma ) * ( 1 - prob_true[:,:,:,:] ) * tf.math.log( 1 - prob_pred[:,:,:,:] + 1e-18 )
    
    pos_loss_tensor_prob = loss_tensor_prob[:,:,:,:] * object_mask[:,:,:,:]
    neg_loss_tensor_prob = loss_tensor_prob[:,:,:,:] * (1 - object_mask[:,:,:,:]) * ignore_mask[:,:,:,:]
    
    prob_focal_loss = K.sum( (pos_loss_tensor_prob[:,:,:,:] + neg_loss_tensor_prob[:,:,:,:]) )/ m

    #class focal loss
    loss_tensor_class =  - ( (1 - class_pred[:,:,:,:])**self.gamma ) * class_true[:,:,:,:] * tf.math.log( class_pred[:,:,:,:] + 1e-18 ) - ( class_pred[:,:,:,:] ** self.gamma) * ( 1 - class_true[:,:,:,:] ) * tf.math.log( 1 - class_pred[:,:,:,:] + 1e-18 )
    class_focal_loss = K.sum(loss_tensor_class[:,:,:,:]*object_mask[:,:,:,:]) / m


    #****************** Focal loss ******************

    #get reg left -- (x,y)
    reg_left_true = y_true[:,:,:,1:3] 
    reg_left_pred = y_pred[:,:,:,1:3]

    reg_left_true = K.cast(reg_left_true,K.dtype(reg_left_pred))

    #get reg center -- (x,y)
    reg_center_true = y_true[:,:,:,3:5] 
    reg_center_pred = y_pred[:,:,:,3:5]

    reg_center_true = K.cast(reg_center_true,K.dtype(reg_center_pred))

    #calculate width x height of anchor box
    reg_wh_true = (reg_center_true[:,:,:,:] - reg_left_true[:,:,:,:])*2
    reg_wh_pred = (reg_center_pred[:,:,:,:] - reg_left_pred[:,:,:,:])*2

    #get reg right
    reg_right_true = reg_left_true[:,:,:,:] + reg_wh_true[:,:,:,:]
    reg_right_pred = reg_left_pred[:,:,:,:] + reg_wh_pred[:,:,:,:]

    #get reg width
    reg_width_true = reg_wh_true[:,:,:,0:1] 
    reg_width_pred = reg_wh_pred[:,:,:,0:1]

    #get reg height
    reg_height_true = reg_wh_true[:,:,:,1:2]
    reg_height_pred = reg_wh_pred[:,:,:,1:2]

    
    #****************** Box Scale Entropy ******************
    
    #calculate mini width
    reg_width_mini = tf.math.minimum(reg_width_true,reg_width_pred)

    reg_width_mini = K.square(reg_width_mini)
    
    #calculate max width
    reg_width_max = tf.math.maximum(reg_width_true,reg_width_pred)

    reg_width_max = K.square(reg_width_max)
    

    #calculate width scale
    reg_width_scale = reg_width_mini[:,:,:,:] / ( reg_width_max[:,:,:,:] + 1e-10 ) 

    #----------------------------------------------------------------------
    
    #calculate mini height
    reg_height_mini = tf.math.minimum(reg_height_true,reg_height_pred)

    reg_height_mini = K.square(reg_height_mini)

    #calculate max height
    reg_height_max = tf.math.maximum(reg_height_true,reg_height_pred)

    reg_height_max = K.square(reg_height_max)


    #calculate height scale
    reg_height_scale =  reg_height_mini[:,:,:,:] / ( reg_height_max[:,:,:,:] + 1e-10 )

    #----------------------------------------------------------------------

    #Box Scale Entropy
    loss_tensor_Box_Scale_Entropy = - tf.math.log(reg_width_scale[:,:,:,:] + 1e-18)  -  tf.math.log(reg_height_scale[:,:,:,:] + 1e-18)

    box_scale_entropy_loss = K.sum(loss_tensor_Box_Scale_Entropy[:,:,:,:] * object_mask[:,:,:,:] ) / m
    

    #****************** Box Scale Entropy ******************
    
    #****************** IOU loss ******************
    
    #calculate IOU  

    #calculate intersection left
    reg_left_intersection = tf.math.maximum(reg_left_true,reg_left_pred)

    #calculate intersection right
    reg_right_intersection = tf.math.minimum(reg_right_true,reg_right_pred)

    #calibrate
    #reg_right_intersection = tf.where((reg_left_intersection>reg_right_intersection),reg_left_intersection,reg_right_intersection) #-- same meaning
    reg_right_intersection = tf.math.maximum(reg_left_intersection,reg_right_intersection) #-- same meaning

    #intersection width height
    intersection_wh = reg_right_intersection[:,:,:,:] - reg_left_intersection[:,:,:,:]

    #intersection area
    intersection_area = intersection_wh[:,:,:,0:1] * intersection_wh[:,:,:,1:2]

    #union area
    true_area = reg_wh_true[:,:,:,0:1] * reg_wh_true[:,:,:,1:2]
    pred_area = reg_wh_pred[:,:,:,0:1] * reg_wh_pred[:,:,:,1:2]

    union_area = true_area[:,:,:,:] + pred_area[:,:,:,:] - intersection_area[:,:,:,:]

    #calculate iou 
    iou_val = intersection_area[:,:,:,:] / ( union_area[:,:,:,:] +  1e-10 )

    #iou loss
    loss_tensor_iou = - tf.math.log( iou_val[:,:,:,:] + 1e-18 ) * object_mask[:,:,:,:] 

    iou_loss = K.sum( loss_tensor_iou ) / m

    #****************** IOU loss ******************
    
    #****************** Loc loss ******************
    loss_tensor_loc = ( K.square(reg_left_true[:,:,:,:] - reg_left_pred[:,:,:,:]) + K.square(reg_center_true[:,:,:,:] - reg_center_pred[:,:,:,:]) ) * object_mask[:,:,:,:] 

    #loc loss
    loc_loss = K.sum(loss_tensor_loc) / m

    #****************** Loc loss ******************

    #calculate reg loss
    reg_loss = loc_loss + iou_loss + box_scale_entropy_loss
 
    #calculate loss
    loss = prob_focal_loss + class_focal_loss + reg_loss
 
    return loss

def get_ignore_mask(ignore_mask,y_true,y_pred,object_mask_bool,ignore_threshold = 0.5):

  """
  y_true -- (batch_size,H,W,info) -- info [prob,x_left,y_left,x_center,y_center,class]
  y_pred -- (batch_size,H,W,info) -- info [prob,x_left,y_left,x_center,y_center,class]
  ignore_mask -- TensorArray
  """
  #set up
  y_true = K.cast(y_true,K.dtype(y_pred))

  m = K.cast(K.shape(y_true)[0],tf.int32)

  i = 0

  i = K.cast(i,K.dtype(m))

  #loop via each batch
  while i < m:

    #get true box -- shape (n,4)
    true_box = tf.boolean_mask(y_true[i,:,:,1:5],object_mask_bool[i,:,:,0])

    #get true box -- shape (1,n,4)
    true_box = true_box[tf.newaxis,:,:]

    #get y_pred_batch -- shape (h,w,4)
    y_pred_batch = y_pred[i,:,:,1:5]
    
    #get y_pred_batch -- shape (h,w,1,4)
    y_pred_batch = y_pred_batch[:,:,tf.newaxis,:]

    #****************** IOU ******************

    #batch pred -- (h,w,1,4)
    batch_pred_left_xy =  y_pred_batch[:,:,:,0:2]
    batch_pred_center_xy = y_pred_batch[:,:,:,2:4]
    
    batch_pred_wh = (batch_pred_center_xy[:,:,:,:] - batch_pred_left_xy[:,:,:,:])*2

    batch_pred_right_xy = batch_pred_left_xy[:,:,:,:] + batch_pred_wh[:,:,:,:] 

    #batch true -- (1,n,4)
    batch_true_left_xy = true_box[:,:,0:2]
    batch_true_center_xy = true_box[:,:,2:4]

    batch_true_wh =  (batch_true_center_xy[:,:,:] - batch_true_left_xy[:,:,:])*2

    batch_true_right_xy = batch_true_left_xy[:,:,:] + batch_true_wh[:,:,:]

    #intersection -- (h,w,n,2)
    batch_intersection_left = K.maximum(batch_pred_left_xy,batch_true_left_xy)
    batch_intersection_right = K.minimum(batch_pred_right_xy,batch_true_right_xy)

    #calibrate 
    batch_intersection_right = K.maximum(batch_intersection_right,batch_intersection_left)

    #batch intersection wh -- (h,w,n,2)
    batch_intersection_wh = batch_intersection_right[:,:,:,:] - batch_intersection_left[:,:,:,:]
    
    #batch intersection area -- (h,w,n)
    batch_intersection_area = batch_intersection_wh[:,:,:,0] * batch_intersection_wh[:,:,:,1]

    #batch union area -- (h,w,n)
    batch_true_union_area = batch_true_wh[:,:,0] * batch_true_wh[:,:,1]
    batch_pred_union_area = batch_pred_wh[:,:,:,0] * batch_pred_wh[:,:,:,1]

    batch_union_area = (batch_true_union_area + batch_pred_union_area) - batch_intersection_area

    #calculate iou -- (h,w,n)
    batch_iou = batch_intersection_area / (batch_union_area + 1e-10)

    #****************** IOU ******************

    #get best iou -- (h,w,1)
    best_iou = K.max(batch_iou,axis=-1,keepdims=True)

    #ignore value -- (h,w,1)
    ignore_val = K.cast(best_iou < ignore_threshold,K.dtype(y_pred))

    #update ignore mask
    ignore_mask = ignore_mask.write(i,ignore_val)

    #update i
    i = i + 1

  #stack to tensor -- (m,h,w,1)
  ignore_mask = ignore_mask.stack()

  return ignore_mask

if __name__ == "__main__":

   y_true = tf.constant(value = np.random.randn(10,38,38,89),dtype=np.float64)
   y_true = np.clip(y_true,0.0,1.0)
   data = tf.constant(value= np.random.randn(10,40,40,3),dtype=np.float64)
   """
   attention_info = {}
   attention_info["CBL_1"] = (20,3,1,"same")
   attention_info["conv_query"] = (20,3,1,"same")
   attention_info["conv_keys"] = (20,3,1,"same")
   attention_info["conv_values"] = (20,3,1,"same")
   """

   attention_info = {}
   attention_info["CBL_1"] = (64,3,1,"same")
   attention_info["CBL_Q"] = (64,3,1,"same")
   attention_info["CBL_K"] = (64,3,1,"same")
   attention_info["CBL_V"] = (64,3,1,"same")
   attention_info["CBL_TU"] = (64,3,1,"same")
   attention_info["CBL_TL"] = (64,3,1,"same")
   
   x = tf.keras.layers.Input(shape =(40,40,3))
   strategy = tf.distribute.MirroredStrategy(cross_device_ops=tf.distribute.HierarchicalCopyAllReduce())

   with strategy.scope():
     
     """
     k1 = tf.keras.layers.BatchNormalization(axis=-1)(x)
     h1 = tf.keras.layers.Conv2D(1024,3,1,"valid",data_format="channels_last",activation=tf.keras.layers.LeakyReLU())(k1)
     h2 = tf.keras.layers.BatchNormalization(axis=-1)(h1)
     h3 = tf.keras.layers.LeakyReLU()(h2)
     h4 = tf.keras.layers.Conv2D(1024,3,1,"same",data_format="channels_last")(h3)
     h5 = tf.keras.layers.BatchNormalization(axis=-1)(h4)
     h6 = tf.keras.layers.LeakyReLU()(h5)
     bat1 = tf.keras.layers.BatchNormalization(axis=-1)(h6)
     k2 = tf.keras.layers.Conv2D(89,3,1,"same",data_format="channels_last",activation="sigmoid")(bat1)

     """
     k1 = tf.keras.layers.Conv2D(512,3,1,"valid",data_format="channels_last",activation=tf.keras.layers.LeakyReLU())(x)
     h2 = tf.keras.layers.BatchNormalization(axis=-1)(k1)
     a1 = tf.keras.layers.LeakyReLU()(h2)
     #AttentionModule_1 = AttentionModule(attention_info)(a1)
     AttentionModule_1 = FourierAttentionModule(attention_info)(a1)
     p2 = tf.keras.layers.Conv2D(512,3,1,"same",data_format="channels_last")(AttentionModule_1)
     h3 = tf.keras.layers.BatchNormalization(axis=-1)(p2)
     a2 = tf.keras.layers.LeakyReLU()(h3)
     k2 = tf.keras.layers.Conv2D(89,3,1,"same",data_format="channels_last",activation="sigmoid")(a2)
     
     model = tf.keras.Model(inputs=x,outputs=k2)

     model.compile(optimizer="adam",loss=alpha_loss())

   b = model.fit(data,y_true,epochs=100)

   plt.plot(b.history["loss"])

   plt.show()
