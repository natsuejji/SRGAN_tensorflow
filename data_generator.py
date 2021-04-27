import numpy as np
import tensorflow as tf
import cv2.cv2 as cv2
from scipy import signal
import random
class DataGenerator:
    def __init__(self, 
                batch_size, 
                image_size = [448,256],
                dataset_basepath='E:\\dataset\\vimeo_septuplet', 
                gau_std=1.6, 
                gau_kernel=21,
                scale = 2,
                mode=0):
        '''
            mode = 0 -> training
            mode = 1 -> testing
        '''
        self.batch_size = batch_size
        self.image_size = image_size
        self.scale = scale
        self.dataset_basepath = dataset_basepath
        self.gaussian_blur_kernel = self.__gaussian_blur_filter(gau_std,gau_kernel)
        if mode == 0:
            self.dataset_path = self.dataset_basepath + '\\sep_trainlist.txt'
            self.dataset = []
        elif mode == 1:
            self.dataset_path = self.dataset_basepath + '\\sep_testlist.txt'
            self.dataset = []
        
        
        with open(self.dataset_path, 'r') as f:
            self.dataset = [x.replace('\n','').replace('/','\\') for x in f.readlines()]
        random.shuffle(self.dataset)

    def __gaussian_blur_filter(self, std, k_size):
        '''
            given fixed standard deviation and kernel size
            return a gaussian 2d kernel
        '''
        axis = np.linspace(-(k_size-1)//2, (k_size-1)//2, k_size)
        x_axis,y_axis = np.meshgrid(axis, axis)
        g_kernel = np.exp(-0.5*( (np.square(x_axis) + np.square(y_axis)) / np.square(std) ) )

        g_kernel = tf.constant(g_kernel, tf.float32)
        g_kernel = tf.expand_dims(tf.expand_dims(g_kernel,axis=-1),axis=-1)
        g_kernel = tf.tile(g_kernel, [1,1,1,3])

        return g_kernel/np.sum(g_kernel)

    def gaussian_blur_2D(self, img):
        
        return tf.nn.conv2d(img, self.gaussian_blur_kernel , strides=1, padding='SAME').numpy()

    def downsampling(self, img, scale):

        _, old_h, old_w, _ = np.squeeze(np.shape(img)) 
        img = tf.image.resize(img, (old_h//scale , old_w//scale))
        return img

    def flip_img(self, img):

        return np.flip(img,axis=1)

    def __get_sample(self):
        return [random.randint(0,len(self.dataset)-1) for _ in range(self.batch_size)]

    def read(self):
        cur_batch = self.__get_sample()

        gt = tf.zeros([0, self.image_size[1], self.image_size[0], 3])
        samples = tf.zeros([0, self.image_size[1]//self.scale, self.image_size[0]//self.scale, 3])

        for sample in cur_batch:
            #讀取ground truth
            gt_image = cv2.imread(self.dataset_basepath + '\\sequences\\' + self.dataset[sample] + '\\im1.png')
            print(self.dataset_basepath + '\\sequences\\' + self.dataset[sample] + '\\im1.png')
            #下採樣
            #lr_image = self.flip_img(gt_image) if random.uniform(0,1) > 0.5 else gt_image

            gt_image = np.expand_dims(gt_image, axis=0)
            lr_image = np.expand_dims(gt_image, axis=0)

            lr_image = self.gaussian_blur_2D(gt_image) if random.uniform(0,1) > 0.5 else gt_image
            lr_image = self.downsampling(lr_image, self.scale)
            
            samples = tf.concat([samples, lr_image], axis=0)
            gt = tf.concat([gt, gt_image], axis=0)

        return samples, gt

    def __call__(self):
        return self.read()

    def generator(self):
        while True:
            yield self.read()

if __name__ == '__main__':
    g = DataGenerator(16)
    test_list = g.__call__()
    print(test_list[0])
    