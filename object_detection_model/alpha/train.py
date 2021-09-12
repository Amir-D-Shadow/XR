import tensorflow as tf
import numpy as np
from alpha import alpha_model
from alpha_loss import alpha_loss
import generate_data
import preprocess_data
import random
import os

#find current path
cur_path = os.getcwd()

#define strategy
strategy = tf.distribute.MirroredStrategy(cross_device_ops=tf.distribute.HierarchicalCopyAllReduce())

#define model,loss,optimizer
with strategy.scope():

   #define loss object
   loss_object = alpha_loss(reduction=tf.keras.losses.Reduction.NONE)

   #define compute loss
   def compute_loss(labels,predictions):

      #large
      large_obj_loss = loss_object(labels[0],predictions[0])

      #medium
      medium_obj_loss = loss_object(labels[1],predictions[1])

      #small
      small_obj_loss = loss_object(labels[2],predictions[2])

      #total loss
      total_loss = large_obj_loss + medium_obj_loss + small_obj_loss

      return total_loss 

   #define learning rate scheduler
   #lr_scheduler = tf.keras.optimizers.schedules.CosineDecay(initial_learning_rate=0.001,decay_steps=5,alpha=1e-5)

   #define optimizer
   #optimizer = tf.keras.optimizers.Adam(learning_rate=lr_scheduler)
   optimizer = tf.keras.optimizers.Adam()

   #define model
   model = alpha_model()
   model.load_weights(f"{cur_path}/gdrive/MyDrive/model_weights")
   #model = tf.keras.models.load_model(f"{cur_path}/gdrive/MyDrive/model")
   
#step function
@tf.function
def distributed_train_step(data_inputs):

   per_replica_losses = strategy.run(train_step,args=(data_inputs,))

   return strategy.reduce(tf.distribute.ReduceOp.SUM,per_replica_losses,axis=None)


def train_step(inputs):

   images,labels = inputs

   with tf.GradientTape() as tape:

      predictions = model(images,train_flag=True)
      
      loss = compute_loss(labels,predictions)

   gradients = tape.gradient(loss,model.trainable_weights)

   optimizer.apply_gradients(zip(gradients,model.trainable_weights))

   return loss

batch_size_per_replica = 4

EPOCHS = 20

#preprocessing class
input_path = f"{cur_path}/gdrive/MyDrive/annotations/train_annotations.csv"
save_path = f"{cur_path}/gdrive/MyDrive/data"

class_train_info = preprocess_class(input_path,save_path)

#preprocessing label
input_path = f"{cur_path}/gdrive/MyDrive/annotations/test_annotations.csv"
save_path = f"{cur_path}/gdrive/MyDrive/data"

img_train_info = preprocessing_label(input_path,save_path)

#train image path
img_train_path = f"{cur_path}/gdrive/MyDrive/img"

img_shape = (640,640)

#get number of sample m
m = len(list(img_train_info.keys()))

#dataset size
global_batch_size = batch_size_per_replica * strategy.num_replicas_in_sync
buffer_size = global_batch_size * 2

#aug flag
aug_flag = False

#train
for i  in range(EPOCHS):

   total_loss = 0.0

   if (i+1) < 295:

     aug_flag = True

   else:

     aug_flag = False

   #get data 
   for train_images, train_labels in get_gt_data(buffer_size,img_train_info,class_train_info,img_train_path,img_shape,aug_flag):

      #cast data
      train_images = train_images.astype(np.float64)

      # Create Datasets from the batches
      train_dataset = tf.data.Dataset.from_tensor_slices((train_images, train_labels)).shuffle(buffer_size).batch(global_batch_size)

      #create distributed dataset
      train_dist_dataset = strategy.experimental_distribute_dataset(train_dataset)

      #Do training
      for batch in train_dist_dataset:

         total_loss = total_loss + distributed_train_step(batch) 
   
   total_loss = total_loss 
   
   print(f"Epoch {i+1} , Loss: {total_loss}")

   #save model for each epoch
   model.save_weights(f"{cur_path}/gdrive/MyDrive/model_weights")
   """
   if ((i+1)%10) == 0:

     #model.save(f"{cur_path}/gdrive/MyDrive/model")
     model.save_weights(f"{cur_path}/gdrive/MyDrive/model_weights")

   """
      

     








