import tensorflow as tf
import tensorflow.keras.backend as K
import numpy  as np
from matplotlib import pyplot as plt

class alpha_loss(tf.keras.losses.Loss):

  def __init__(self,gamma = 2,**kwargs):

    super(alpha_loss,self).__init__(**kwargs)

    self.gamma = gamma

  def call(self,y_true,y_pred):

    #get object mask
    object_mask = K.cast(y_true[:,:,:,0],K.dtype(y_pred[:,:,:,0]))
    
    #reg_scale_factor = tf.math.reduce_mean(object_mask)
    #reg_scale_factor =  reg_scale_factor / (1 - reg_scale_factor) 

    #get prob
    prob_true = y_true[:,:,:,0]
    prob_pred = y_pred[:,:,:,0]

    prob_true = K.cast(prob_true,K.dtype(prob_pred))

    #get class
    class_true = y_true[:,:,:,9:]
    class_pred = y_pred[:,:,:,9:]

    class_true = K.cast(class_true,K.dtype(class_pred))

    #****************** Focal loss ******************

    #get batch size
    m = K.cast(K.shape(y_pred)[0],K.dtype(y_pred))

    #clip the prediction
    #prob_pred = K.clip(prob_pred,min_value = 0.0, max_value = 1.0)
    #class_pred = K.clip(class_pred,min_value = 0.0, max_value = 1.0)

    #prob focal loss
    loss_tensor =  - ( (1 - prob_pred[:,:,:])**self.gamma ) * prob_true[:,:,:] * tf.math.log( prob_pred[:,:,:] + 1e-18 ) - ( (prob_pred[:,:,:]) ** self.gamma ) * ( 1 - prob_true[:,:,:] ) * tf.math.log( 1 - prob_pred[:,:,:] + 1e-18 ) 
    prob_focal_loss = K.sum(loss_tensor)/ m

    #class focal loss
    loss_tensor =  - ( (1 - class_pred[:,:,:,:])**self.gamma ) * class_true[:,:,:,:] * tf.math.log( class_pred[:,:,:,:] + 1e-18 ) - ( (class_pred[:,:,:,:]) ** self.gamma) * ( 1 - class_true[:,:,:,:] ) * tf.math.log( 1 - class_pred[:,:,:,:] + 1e-18 )
    
    class_focal_loss = K.sum(( loss_tensor[:,:,:,:] * object_mask[:,:,:,tf.newaxis] )) / m


    #****************** Focal loss ******************

    #get reg left -- (x,y)
    reg_left_true = y_true[:,:,:,1:3] 
    reg_left_pred = y_pred[:,:,:,1:3]

    reg_left_true = K.cast(reg_left_true,K.dtype(reg_left_pred))

    #get left ratio
    reg_left_true_ratio = y_true[:,:,:,5:7]
    reg_left_pred_ratio = y_pred[:,:,:,5:7]

    reg_left_true_ratio = K.cast(reg_left_true_ratio,K.dtype(reg_left_pred_ratio))

    #get reg left 1,2
    reg_left_true_1 = reg_left_true[:,:,:,:] * reg_left_true_ratio[:,:,:,:]
    reg_left_true_2 = reg_left_true[:,:,:,:] * (1 - reg_left_true_ratio[:,:,:,:])
    
    reg_left_pred_1 = reg_left_pred[:,:,:,:] * reg_left_pred_ratio[:,:,:,:]
    reg_left_pred_2 = reg_left_pred[:,:,:,:] * (1 - reg_left_pred_ratio[:,:,:,:])

    #mask the reg (because we consider the box with object only )
    #reg_left_pred = reg_left_pred[:,:,:,:] * object_mask[:,:,:,tf.newaxis]

    #get reg center -- (x,y)
    reg_center_true = y_true[:,:,:,3:5] 
    reg_center_pred = y_pred[:,:,:,3:5]

    reg_center_true = K.cast(reg_center_true,K.dtype(reg_center_pred))

    #reg center ratio
    reg_center_true_ratio = y_true[:,:,:,7:9]
    reg_center_pred_ratio = y_pred[:,:,:,7:9]

    reg_center_true_ratio = K.cast(reg_center_true_ratio,K.dtype(reg_center_pred_ratio))

    #get reg center 1,2
    reg_center_true_1 = reg_center_true[:,:,:,:] * reg_center_true_ratio[:,:,:,:]
    reg_center_true_2 = reg_center_true[:,:,:,:] * (1 - reg_center_true_ratio[:,:,:,:])

    reg_center_pred_1 = reg_center_pred[:,:,:,:] * reg_center_pred_ratio[:,:,:,:]
    reg_center_pred_2 = reg_center_pred[:,:,:,:] * (1 - reg_center_pred_ratio[:,:,:,:])

    #mask the reg (because we consider the box with object only )
    #reg_center_pred = reg_center_pred[:,:,:,:] * object_mask[:,:,:,tf.newaxis]

    #calculate width x height of anchor box
    reg_wh_true_1 = (reg_center_true_1[:,:,:,:] - reg_left_true_1[:,:,:,:])*2
    reg_wh_true_2 = (reg_center_true_2[:,:,:,:] - reg_left_true_2[:,:,:,:])*2
    
    reg_wh_pred_1 = (reg_center_pred_1[:,:,:,:] - reg_left_pred_1[:,:,:,:])*2
    reg_wh_pred_2 = (reg_center_pred_2[:,:,:,:] - reg_left_pred_2[:,:,:,:])*2 

    #calculate reg right
    reg_right_true_1 = reg_left_true_1[:,:,:,:] + reg_wh_true_1[:,:,:,:]
    reg_right_true_2 = reg_left_true_2[:,:,:,:] + reg_wh_true_2[:,:,:,:]
    
    reg_right_pred_1 = reg_left_pred_1[:,:,:,:] + reg_wh_pred_1[:,:,:,:]
    reg_right_pred_2 = reg_left_pred_2[:,:,:,:] + reg_wh_pred_2[:,:,:,:]

    #****************** Focal IOU loss ******************

    #----------------------------------------------------------------------
    #calculate IOU  1

    #calculate intersection left 1
    reg_left_intersection_1 = tf.math.maximum(reg_left_pred_1,reg_left_true_1)

    #calculate intersection right 1
    reg_right_intersection_1 = tf.math.minimum(reg_right_pred_1,reg_right_true_1)

    #calibrate
    #reg_right_intersection = tf.where((reg_left_intersection>reg_right_intersection),reg_left_intersection,reg_right_intersection) #-- same meaning
    reg_right_intersection_1 = tf.math.maximum(reg_left_intersection_1,reg_right_intersection_1) #-- same meaning

    #intersection width height 1
    intersection_wh_1 = reg_right_intersection_1[:,:,:,:] - reg_left_intersection_1[:,:,:,:]

    #intersection area 1
    intersection_area_1 = intersection_wh_1[:,:,:,0] * intersection_wh_1[:,:,:,1]

    #union area 1
    true_area_1 = reg_wh_true_1[:,:,:,0] * reg_wh_true_1[:,:,:,1]
    pred_area_1 = reg_wh_pred_1[:,:,:,0] * reg_wh_pred_1[:,:,:,1]

    union_area_1 = true_area_1[:,:,:] + pred_area_1[:,:,:] - intersection_area_1[:,:,:]

    #calculate iou 1
    iou_val_1 = intersection_area_1[:,:,:] / ( union_area_1[:,:,:] +  1e-10 )

    #calculate Focal IOU 1
    reg_loss_1 = - ((1 - iou_val_1[:,:,:])**self.gamma) * tf.math.log(iou_val_1 + 1e-18) * object_mask[:,:,:]
    
    #----------------------------------------------------------------------

    #calculate IOU  2

    #calculate intersection left 2
    reg_left_intersection_2 = tf.math.maximum(reg_left_pred_2,reg_left_true_2)

    #calculate intersection right 2
    reg_right_intersection_2 = tf.math.minimum(reg_right_pred_2,reg_right_true_2)

    #calibrate
    #reg_right_intersection = tf.where((reg_left_intersection>reg_right_intersection),reg_left_intersection,reg_right_intersection) #-- same meaning
    reg_right_intersection_2 = tf.math.maximum(reg_left_intersection_2,reg_right_intersection_2) #-- same meaning

    #intersection width height 2
    intersection_wh_2 = reg_right_intersection_2[:,:,:,:] - reg_left_intersection_2[:,:,:,:]

    #intersection area 2
    intersection_area_2 = intersection_wh_2[:,:,:,0] * intersection_wh_2[:,:,:,1]

    #union area 2
    true_area_2 = reg_wh_true_2[:,:,:,0] * reg_wh_true_2[:,:,:,1]
    pred_area_2 = reg_wh_pred_2[:,:,:,0] * reg_wh_pred_2[:,:,:,1]

    union_area_2 = true_area_2[:,:,:] + pred_area_2[:,:,:] - intersection_area_2[:,:,:]

    #calculate iou 2
    iou_val_2 = intersection_area_2[:,:,:] / ( union_area_2[:,:,:] +  1e-10 )

    #calculate Focal IOU 2
    reg_loss_2 = - ((1 - iou_val_2[:,:,:])**self.gamma) * tf.math.log(iou_val_2 + 1e-18) * object_mask[:,:,:]

    #----------------------------------------------------------------------
    #sum over total loss
    loss_tensor = reg_loss_1[:,:,:] + reg_loss_2[:,:,:]
    reg_loss = K.sum(loss_tensor) / m

    #****************** Focal IOU loss ******************
 
    #calculate loss
    loss = prob_focal_loss + class_focal_loss + 5 * reg_loss
 
    return loss


if __name__ == "__main__":

   y_true = tf.constant(value = np.random.randn(10,38,38,89),dtype=np.float64)
   y_true = np.clip(y_true,0.0,1.0)
   data = tf.constant(value= np.random.randn(10,40,40,3),dtype=np.float64)

   x = tf.keras.layers.Input(shape =(40,40,3))
   strategy = tf.distribute.MirroredStrategy(cross_device_ops=tf.distribute.HierarchicalCopyAllReduce())

   with strategy.scope():
     
     k1 = tf.keras.layers.BatchNormalization(axis=-1)(x)
     h1 = tf.keras.layers.Conv2D(1024,3,1,"valid",data_format="channels_last")(k1)
     h2 = tf.keras.layers.BatchNormalization(axis=-1)(h1)
     h3 = tf.keras.layers.LeakyReLU()(h2)
     drop_h = tf.keras.layers.SpatialDropout2D(0.5)(h3)
     h4 = tf.keras.layers.Conv2D(1024,3,1,"same",data_format="channels_last")(drop_h)
     h5 = tf.keras.layers.BatchNormalization(axis=-1)(h4)
     h6 = tf.keras.layers.LeakyReLU()(h5)
     bat1 = tf.keras.layers.BatchNormalization(axis=-1)(h6)
     k2 = tf.keras.layers.Conv2D(89,3,1,"same",data_format="channels_last",activation="sigmoid")(bat1)

     model = tf.keras.Model(inputs=x,outputs=k2)

     model.compile(optimizer="adam",loss=alpha_loss())

   b = model.fit(data,y_true,epochs=100)

   plt.plot(b.history["loss"])

   plt.show()
