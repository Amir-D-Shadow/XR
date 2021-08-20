import numpy as np
import tensorflow as tf
import cv2
from numba import jit
import random

#data augmentation
def data_aug(img):

   idx = np.random.randint(0,4)

   if idx == 0:

      h  = tune_odd(np.random.randint(1,17))
      w  = tune_odd(np.random.randint(1,17))

      kernel_shape = (h,w)

      img = GaussianBlur(img,kernel_shape)

   elif idx == 1:

      img = GaussianNoise(img)

   elif idx == 2:

      color_aug_seq = [ i for i in range(4)]
      random.shuffle(color_aug_seq)

      for i in color_aug_seq:

         if i == 0:

            img = random_brightness(img)

         elif i == 1:

            img = random_saturation(img)

         elif i == 2:

            img = random_contrast(img)

         elif i == 3:

            img = random_hue(img)

   elif idx == 3:

      h = np.random.randint(8,81)
      w = np.random.randint(8,81)

      kernel_size = (h,w)

      for i in range(12):

         img = random_erase(img,kernel_size)

      
   return img

@jit(nopython=True)
def tune_odd(val):

   if (val % 2) == 0:

      val = val + 1

   return val

#Gaussian Blur
def GaussianBlur(img,kernel_shape,sigma=0):

   """
   img -- numpy array
   kernel_shape -- (int,int)
   sigma -- real number
   """

   return cv2.GaussianBlur(img,kernel_shape,sigma)


##Gaussian Noise
@jit(nopython=True)
def ext_operator(val):

  if val > 255 :

    val = 255

  elif val < 0:

    val = 0

  return val

@jit(nopython=True)
def GaussianNoise(img,low=8,high=64):

  nH,nW,nC = img.shape

  for h in range(nH):

    for w in range(nW):

      for c in range(nC):

        factor = np.random.randint(low,high)

        loc = factor * np.random.random()

        scale = factor * np.random.random()

        noise = np.random.normal(loc,scale)

        img[h,w,c] = ext_operator(img[h,w,c]+noise)

  return img


#brightness
def random_brightness(img):

   return tf.image.random_brightness(img,0.4).numpy()


#saturation
def random_saturation(img):

   return tf.image.random_saturation(img,1.5,8.0).numpy()

#contrast
def random_contrast(img):

   return tf.image.random_contrast(img,1.5,8.0).numpy()

#hue
def random_hue(img):

   return tf.image.random_hue(img,0.4).numpy()

#erase
@jit(nopython=True)
def random_erase(img,kernel_size=(64,64)):

   nH,nW,_ = img.shape
   fH,fW = kernel_size

   new_nH = nH - fH + 1
   new_nW = nW - fW + 1

   nH_start = np.random.randint(0,new_nH)
   nW_start = np.random.randint(0,new_nW)

   nH_end = nH_start + fH
   nW_end = nW_start + fW

   erase_region = img[nH_start:nH_end,nW_start:nW_end,:].copy()

   for h in range(nH_start,nH_end):

      for w in range(nW_start,nW_end):
         
         random_h = np.random.randint(0,fH)
         random_w = np.random.randint(0,fW)

         img[h,w,:] = erase_region[random_h,random_w,:].copy()


   return img


         
