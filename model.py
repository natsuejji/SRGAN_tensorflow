import tensorflow as tf

@tf.Module
class D_ResidualBlock:
    def __init__(self):
        super(D_ResidualBlock, self).__init__(name ="R_B")
    def __build(self):
        self.conv2d = tf.keras.layers.Conv2D(64,3,2,padding='SAME')
        self.bn = tf.keras.layers.BatchNormalization()
        
    def call(self, x):
        x = self.conv2d()(x)
        x = self.bn()(x)
        x = tf.nn.leaky_relu(x, alpha=0.2)
        return x

@tf.Module
class G_ResidualBlock:
    def __init__(self):
        super(G_ResidualBlock, self).__init__(name ="R_B")
    def __build(self):
        self.conv2d_1 = tf.keras.layers.Conv2D(64,3,1,padding='SAME')
        self.bn_1 = tf.keras.layers.BatchNormalization()
        self.prelu = tf.keras.layers.PReLU()
        self.conv2d_2 = tf.keras.layers.Conv2D(64,3,1,padding='SAME')
        self.bn_2 = tf.keras.layers.BatchNormalization()
    def call(self, x):
        origin_x = x
        x = self.conv2d_1()(x)
        x = self.bn_1()(x)
        x = self.prelu(x)
        x = self.conv2d_2()(x)
        x = self.bn_2()(x)
        x = x + origin_x
        return x
@tf.Module
class SRGenerator:
    def __init__(self):
        super(SRGenerator, self).__init__(name ="SR_G")
        
    def __build(self):
        pass
    def call(self):
        pass

@tf.Module
class SRDiscriminator:
    def __init__(self):
        super(SRDiscriminator, self).__init__(name ="SR_D")
    def __build(self):
        pass
    def call(self):
        pass
