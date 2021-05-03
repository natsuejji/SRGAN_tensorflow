import tensorflow as tf
import common
from tensorflow.keras.applications.vgg19 import VGG19
class D_ConvBlock(tf.Module):
    def __init__(self, filters=64, strides=1):
        super(D_ConvBlock, self).__init__(name ="D_B")
        self.filters = filters
        self.strides = strides
        self.__build()
    def __build(self):
        self.conv2d = tf.keras.layers.Conv2D(self.filters,3, self.strides, padding='SAME')
        self.bn = tf.keras.layers.BatchNormalization()
        
    def __call__(self, x):
        x = self.conv2d(x)
        x = self.bn(x)
        x = tf.nn.leaky_relu(x, alpha=0.2)
        return x

class G_UpsamplingBlock(tf.Module):
    def __init__(self, scale):
        super(G_UpsamplingBlock, self).__init__(name ="G_Block")
        self.scale = scale
        self.__build()
    def __build(self):
        self.conv2d = tf.keras.layers.Conv2D(256,3,1,padding='SAME')
        self.prelu = tf.keras.layers.PReLU(shared_axes=[1,2])
    def __call__(self, x):
        x = self.conv2d(x)
        x = tf.nn.depth_to_space(x, self.scale)
        x = self.prelu(x)
        return x

class G_ResidualBlock(tf.Module):
    def __init__(self, momentum=0.8):
        super(G_ResidualBlock, self).__init__(name ="G_RBlock")
        self.momentum = momentum
        self.__build()
    def __build(self):
        self.conv2d_1 = tf.keras.layers.Conv2D(64,3,1,padding='SAME')
        self.bn_1 = tf.keras.layers.BatchNormalization(momentum=self.momentum)
        self.prelu = tf.keras.layers.PReLU(shared_axes=[1,2])
        self.conv2d_2 = tf.keras.layers.Conv2D(64,3,1,padding='SAME')
        self.bn_2 = tf.keras.layers.BatchNormalization(momentum=self.momentum)
    def __call__(self, x):
        origin_x = x
        x = self.conv2d_1(x)
        x = self.bn_1(x)
        x = self.prelu(x)
        x = self.conv2d_2(x)
        x = self.bn_2(x)
        # skip connected
        x = x + origin_x
        return x

class SRGenerator(tf.Module):
    def __init__(self, num_res_blocks=16):
        super(SRGenerator, self).__init__(name ="SR_G")
        self.num_res_blocks = num_res_blocks
        self.__build()
    def __build(self):
        self.conv2d_1 = tf.keras.layers.Conv2D(64,9,1, padding='SAME')
        self.prelu = tf.keras.layers.PReLU(shared_axes=[1,2])
        self.g_blocks = [G_ResidualBlock() for x in range(self.num_res_blocks)]
        self.conv2d_2 = tf.keras.layers.Conv2D(64,3,1,padding='SAME')
        self.bn = tf.keras.layers.BatchNormalization(momentum=0.8)
        self.conv2d_3 = tf.keras.layers.Conv2D(256,3,1,padding='SAME')
        self.upsampling_blocks = [G_UpsamplingBlock(2) for i in range(2)]
        self.conv2d_4 = tf.keras.layers.Conv2D(3,9,1, padding='SAME')

    def __call__(self, x):
        #限縮到[-1~1]
        x = common.normalize_m11(x)
        x = self.conv2d_1(x)
        x = self.prelu(x)
        orig_x = x
        for gblock in self.g_blocks:
            x = gblock(x)
        x = self.conv2d_2(x)
        x = self.bn(x)
        x = orig_x + x
        for upblock in self.upsampling_blocks:
            x = upblock(x)
        x = self.conv2d_4(x)
        return x
        
class SRDiscriminator(tf.Module):
    def __init__(self):
        super(SRDiscriminator, self).__init__(name ="SR_D")
        self.__build()
    def __build(self):
        self.conv2d = tf.keras.layers.Conv2D(64,3,1,padding='SAME')
        self.d_block1 = D_ConvBlock(64,2)
        self.d_block2 = D_ConvBlock(128,1)
        self.d_block3 = D_ConvBlock(128,2)
        self.d_block4 = D_ConvBlock(256,1)
        self.d_block5 = D_ConvBlock(256,2)
        self.d_block6 = D_ConvBlock(512,1)
        self.d_block7 = D_ConvBlock(512,2)
        self.flatten = tf.keras.layers.Flatten()
        self.dense_1 = tf.keras.layers.Dense(1024)
        self.dense_2 = tf.keras.layers.Dense(1)

    def __call__(self,x):
        #限縮到[-1~1]
        x = common.normalize_m11(x)

        x = self.conv2d(x)
        x = tf.keras.layers.LeakyReLU(0.2)(x)

        x = self.d_block1(x)
        x = self.d_block2(x)
        x = self.d_block3(x)
        x = self.d_block4(x)
        x = self.d_block5(x)
        x = self.d_block6(x)
        x = self.d_block7(x)

        x = self.flatten(x)
        x = self.dense_1(x)
        x = tf.keras.layers.LeakyReLU(0.2)(x)
        x = self.dense_2(x)
        return x


def vgg22():

    return VGG19(5)

def vgg54():

    return VGG19(20)

def _vgg(target_layer, hr_image_size):

    vgg = VGG19(input_shape=(hr_image_size,hr_image_size,3), include_top=False)
    return tf.keras.Model(vgg.input, vgg.layers[target_layer].output)
