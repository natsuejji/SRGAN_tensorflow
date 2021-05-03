from model import vgg22, vgg54
import numpy as np
from common import psnr
from build_model import build_srresnet
import data_generator
import tensorflow as tf

class GPretrain:
    def __init__(self,
                 generator,
                 loss=tf.keras.losses.MeanSquaredError(),
                 epochs=100000,
                 eval_steps=100,
                 step_pre_epoch=1,
                 save_only_best=False,
                 val_steps=1,
                 ckpt_dir='./ckpt',
                 batch_size=16,
                 learning_rate=tf.keras.optimizers.schedules.PiecewiseConstantDecay(
                     boundaries=[100000], values=[1e-4, 1e-5])
                 ):
        self.lr = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs,
        self.step_pre_epoch = 1,
        self.save_only_best = save_only_best
        self.eval_steps = eval_steps
        self.val_steps = val_steps
        self.loss = loss
        self.check_point = tf.train.Checkpoint(model=generator,
                                               psnr=tf.Variable(-1.0),
                                               optimizer=tf.optimizers.Adam(
                                                   learning_rate),
                                               step=tf.Variable(0.0))
        self.checkpoint_manager = tf.train.CheckpointManager(checkpoint=self.check_point,
                                                             directory=ckpt_dir,
                                                             max_to_keep=3)

        self.train_set = data_generator.DataGenerator(
            batch_size=self.batch_size)
        self.test_set = data_generator.DataGenerator(
            batch_size=self.batch_size, mode=1)

    def train_step(self, lr_image, hr_image):
        with tf.GradientTape() as t:

            sr_image = self.check_point.model(lr_image)
            loss = self.loss(hr_image, sr_image)

        gradients = t.gradient(
            loss, self.check_point.model.trainable_variables)
        self.check_point.optimizer.apply_gradients(
            zip(gradients, self.check_point.model.trainable_variables))
        return loss

    def train(self):
        loss_mean = tf.metrics.Mean()

        ckpt_mgr = self.checkpoint_manager
        ckpt = self.check_point

        for _ in range(self.epochs[0]):
            for _ in range(self.step_pre_epoch[0]):
                ckpt.step.assign_add(1)
                lr, hr = self.train_set.__call__()
                loss = self.train_step(lr, hr)
                loss_mean(loss)

                if ckpt.step % self.eval_steps == 0:
                    loss_val = loss_mean.result()
                    loss_mean.reset_states()
                    val_psnr_mean = tf.keras.metrics.Mean()

                    # eval
                    for _ in range(self.val_steps):
                        lr, hr = self.test_set.__call__()
                        sr = ckpt.model(lr)
                        val_psnr_mean(psnr(hr, sr))

                    psnr_val = val_psnr_mean.result()
                    val_psnr_mean.reset_states()

                    print(
                        f'{ckpt.step.numpy().astype(np.int8)}/{self.epochs[0]*self.step_pre_epoch[0]} :  loss = {loss_val.numpy()}, psnr = {psnr_val.numpy()}.')
                    if self.save_only_best and psnr_value <= ckpt.psnr:
                        # skip saving checkpoint, no PSNR improvement
                        continue

                    ckpt.psnr = psnr_val
                    ckpt_mgr.save()


class SRGANTrainer:
    def __init__(self,
                 generator,
                 discriminator,
                 ckpt_dir='./ckpt',
                 vgg_loss='VGG54',
                 batch_size=16,
                 learning_rate=tf.keras.optimizers.schedules.PiecewiseConstantDecay(
                     boundaries=[200000], values=[1e-4, 1e-5])
                 ):
        self.lr = learning_rate
        self.batch_size = batch_size

        if vgg_loss == 'VGG54':
            self.vgg = vgg54
        elif vgg_loss == 'VGG22':
            self.vgg = vgg22
        else:
            raise ValueError(f'{vgg_loss} must be VGG54 or VGG22.')

        self.check_point = tf.train.Checkpoint(g=generator,
                                               d=discriminator,
                                               psnr=tf.Variable(-1.0),
                                               optimizer=tf.optimizers.Adam(
                                                   learning_rate),
                                               step=tf.Variable(0.0))
        self.checkpoint_manager = tf.train.CheckpointManager(checkpoint=self.check_point,
                                                             directory=ckpt_dir,
                                                             checkpoint_name='SRGAN',
                                                             max_to_keep=3)

        self.train_set = data_generator.DataGenerator(
            batch_size=self.batch_size)
        self.test_set = data_generator.DataGenerator(
            batch_size=self.batch_size, mode=1)

    def vgg_loss (self, lr_image, hr_image):

        lr_f = self.vgg(lr_image)/12.75
        hr_f = self.vgg(hr_image)/12.75

        return tf.keras.losses.MeanSquaredError(lr_f, hr_f)

    def train_step(self, lr_image, hr_image):
        with tf.GradientTape() as gen_t, tf.GradientTape() as disc_t:

            sr_image = self.check_point.g(lr_image)
            #advensial loss
            sr_out = self.check_point.d(sr_image)
            hr_out = self.check_point.d(hr_image)
            #Context loss
            context_loss = self.vgg_loss(lr_image, hr_image) 
            #generator loss
            g_loss = 0.001 * tf.keras.losses.binary_crossentropy(tf.ones_like(sr_out), sr_out) + context_loss 
            #discriminator loss
            d_loss = tf.keras.losses.binary_crossentropy(tf.ones_like(hr_out), hr_out) + tf.keras.losses.binary_crossentropy(tf.zeros_like(sr_out), sr_out)
            
        g_gradients = gen_t.gradient(g_loss,self.check_point.g.trainable_variables)
        d_gradients = disc_t.gradient(d_loss,self.check_point.d.trainable_variables)
        self.check_point.optimizer.apply_gradients(zip(g_gradients, self.check_point.g.trainable_variables))
        self.check_point.optimizer.apply_gradients(zip(d_gradients, self.check_point.d.trainable_variables))

        return g_loss, d_loss

    def train(self, epochs=300000, eval_steps=1000, val_steps=10, save_only_best=False):
        g_loss_mean = tf.metrics.Mean()
        d_loss_mean = tf.metrics.Mean()
        eval_psnr_mean = tf.keras.metrics.Mean()
        
        ckpt_mgr = self.checkpoint_manager
        ckpt = self.check_point

        for _ in range(epochs):

            ckpt.step.assign_add(1)
            lr, hr = self.train_set.__call__()
            g_loss,d_loss = self.train_step(lr, hr)
            g_loss_mean(loss)
            d_loss_mean(loss)

            if ckpt.step % eval_steps == 0:
                g_loss_val = g_loss_mean.result()
                d_loss_val = d_loss_mean.result()

                g_loss_mean.reset_states()
                d_loss_mean.reset_states()

                # evaluation
                for _ in range(val_steps):
                    lr, hr = self.test_set.__call__()
                    sr = ckpt.model(lr)
                    eval_psnr_mean(psnr(hr, sr))

                psnr_val = eval_psnr_mean.result()
                eval_psnr_mean.reset_states()

                print(f'{ckpt.step.numpy().astype(np.int8)}/{epochs} :'  \
                      f'g_loss = {g_loss_val.numpy()}, d_loss == {d_loss_val.numpy()} psnr = {psnr_val.numpy()}.')
                
                if save_only_best and psnr_val <= ckpt.psnr:
                    # skip saving checkpoint, no PSNR improvement
                    continue

                ckpt.psnr = psnr_val
                ckpt_mgr.save()
