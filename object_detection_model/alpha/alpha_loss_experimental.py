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
    object_mask = K.cast(y_true[:,:,:,0:1],K.dtype(y_pred))

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
    loss_tensor =  - ( (1 - prob_pred[:,:,:,:])**self.gamma ) * prob_true[:,:,:,:] * tf.math.log( prob_pred[:,:,:,:] + 1e-18 ) - ( prob_pred[:,:,:,:] ** self.gamma ) * ( 1 - prob_true[:,:,:,:] ) * tf.math.log( 1 - prob_pred[:,:,:,:] + 1e-18 ) 
    prob_focal_loss = K.sum(loss_tensor)/ m

    #class focal loss
    loss_tensor =  - ( (1 - class_pred[:,:,:,:])**self.gamma ) * class_true[:,:,:,:] * tf.math.log( class_pred[:,:,:,:] + 1e-18 ) - ( class_pred[:,:,:,:] ** self.gamma) * ( 1 - class_true[:,:,:,:] ) * tf.math.log( 1 - class_pred[:,:,:,:] + 1e-18 )
    class_focal_loss = K.sum(loss_tensor[:,:,:,:]*object_mask[:,:,:,:]) / m


    #****************** Focal loss ******************

    #get reg left -- (x,y)
    reg_left_true = y_true[:,:,:,1:3] 
    reg_left_pred = y_pred[:,:,:,1:3]

    reg_left_true = K.cast(reg_left_true,K.dtype(reg_left_pred))

    #mask the reg (because we consider the box with object only )
    #reg_left_pred = reg_left_pred[:,:,:,:] * object_mask[:,:,:,tf.newaxis]

    #get reg center -- (x,y)
    reg_center_true = y_true[:,:,:,3:5] 
    reg_center_pred = y_pred[:,:,:,3:5]

    reg_center_true = K.cast(reg_center_true,K.dtype(reg_center_pred))

    #mask the reg (because we consider the box with object only )
    #reg_center_pred = reg_center_pred[:,:,:,:] * object_mask[:,:,:,tf.newaxis]

    #calculate width x height of anchor box
    reg_wh_true = (reg_center_true[:,:,:,:] - reg_left_true[:,:,:,:])*2
    reg_wh_pred = (reg_center_pred[:,:,:,:] - reg_left_pred[:,:,:,:])*2 

    #calculate reg right
    reg_right_true = reg_left_true[:,:,:,:] + reg_wh_true[:,:,:,:]
    reg_right_pred = reg_left_pred[:,:,:,:] + reg_wh_pred[:,:,:,:]

    #****************** Focal IOU loss ******************

    #----------------------------------------------------------------------
    #calculate IOU  

    #calculate intersection left
    reg_left_intersection = tf.math.maximum(reg_left_pred,reg_left_true)

    #calculate intersection right
    reg_right_intersection = tf.math.minimum(reg_right_pred,reg_right_true)

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

    #----------------------------------------------------------------------

    #calculate reg loss
    #loss_tensor = - ( (1 - iou_val[:,:,:,:])**self.gamma ) * tf.math.log( iou_val[:,:,:,:] + 1e-18) * object_mask[:,:,:,:]
    loss_tensor = (1 - iou_val[:,:,:,:])*object_mask[:,:,:,:]
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
