import numpy as np
import tensorflow as tf

def psnr(x,y):
    return tf.image.psnr(x, y, max_val=255)

def normalize_01(image):
    return image/255.0

def normalize_m11(image):
    return image/127.5-1.0
    
def de_normalize_m11(image):
    return (image+1.0)*127.5

def denormalize_01(image):
    return image * 255.0

def single_inference(model, lr_image):
    lr_image = np.expand_dims(lr_image, axis=0) 
    lr_image = tf.cast(lr_image, tf.float32)

    sr_image = model(lr_image)
    sr_image = de_normalize_m11(sr_image)
    sr_image = tf.clip_by_value(sr_image, 0, 255)
    sr_image = tf.round(sr_image)
    sr_image = tf.cast(sr_image, tf.uint8)

    return sr_image