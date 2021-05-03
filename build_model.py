from model import SRGenerator, SRDiscriminator
import tensorflow as tf

def build_srresnet(input_shape=(24,24,3)):
    srresnet = SRGenerator()
    inputs_ts = tf.keras.Input(input_shape)
    output_ts = srresnet(inputs_ts)
    model = tf.keras.Model(inputs_ts, output_ts)
    return model

def build_disc(input_shape=(None,None,3)):
    disc = SRDiscriminator()
    inputs_ts = tf.keras.Input(input_shape)
    output_ts = srresnet(inputs_ts)
    model = tf.keras.Model(inputs_ts, output_ts)
    return model
