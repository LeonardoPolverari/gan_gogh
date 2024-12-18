from utils.gan_functions import masked_mae_loss, masked_mse_loss
import keras
from tensorflow import keras as tfk
import tensorflow as tf


@keras.saving.register_keras_serializable()
class GAN(tfk.Model):

    # Initialise the GAN with a discriminator, generator, latent dimension, and discriminator update frequency
    def __init__(self, local_discriminator, generator, n_discriminator_updates=3, **kwargs):
        super().__init__(**kwargs)
        self.local_discriminator = local_discriminator
        self.generator = generator
        self.n_discriminator_updates = n_discriminator_updates

        # Initialise loss trackers for discriminator and generator
        self.mae_loss_tracker = tfk.metrics.Mean(name="mae_loss")
        self.local_loss_tracker = tfk.metrics.Mean(name="local_loss")
        self.g_loss_tracker = tfk.metrics.Mean(name="g_loss")
        self.local_adv_loss_tracker = tfk.metrics.Mean(name="local_adv_loss")

    # Compile the GAN with optimisers and an optional loss function
    def compile(self, local_optimizer, g_optimizer, loss_fn=None, mae_weight=0, local_weight = 1):
        super(GAN, self).compile()
        self.local_optimizer = local_optimizer
        self.g_optimizer = g_optimizer
        self.loss_fn = loss_fn or tfk.losses.BinaryCrossentropy(from_logits=True)
        self.mse_loss_fn = masked_mse_loss
        self.mae_loss_fn = masked_mae_loss
        self.mae_weight = mae_weight
        self.local_weight = local_weight

    def get_config(self):
        # Return all arguments passed to the __init__ method
        config = super(GAN, self).get_config()
        config.update({
            'local_discriminator': keras.saving.serialize_keras_object(self.local_discriminator),
            'generator' : keras.saving.serialize_keras_object(self.generator),
            'n_discriminator_updates' : self.n_discriminator_updates
        
        })
        return config

    @classmethod
    def from_config(cls, config):
        # Recreate the model from the config
        local_discriminator_config = config.pop("local_discriminator")
        local_discriminator = keras.saving.deserialize_keras_object(local_discriminator_config)
        generator_config = config.pop("generator")
        generator = keras.saving.deserialize_keras_object(generator_config)
        return cls(local_discriminator, generator, **config)

    # Define GAN metrics
    @property
    def metrics(self):
        return [self.local_loss_tracker, self.g_loss_tracker]

    def _adversarial_loss(self, fake_output, generated_images, real_images):
      adv_loss = self.loss_fn(tf.ones_like(fake_output), fake_output)
      return adv_loss

    def _mse_loss(self, generated_images, real_images, masks):
        mse_loss = self.mse_loss_fn(real_images, generated_images, masks)
        return mse_loss
        
    def _mae_loss(self, generated_images, real_images, masks):
        mse_loss = self.mae_loss_fn(real_images, generated_images, masks)
        return mse_loss
        

    # Compute discriminator loss
    def _discriminator_loss(self, real_output, fake_output):
        real_loss = self.loss_fn(tf.ones_like(real_output), real_output)
        fake_loss = self.loss_fn(tf.zeros_like(fake_output), fake_output)
        return real_loss + fake_loss

    # Define the training step
    @tf.function
    def train_step(self, masked_and_real_images):

        # Train discriminator multiple times
        global_loss = 0
        local_loss = 0
        for _ in range(self.n_discriminator_updates):

            real_images = masked_and_real_images[0]
            masks = masked_and_real_images[1]
            masked_images = masks * real_images
            inverse_masks = (1 - masks)
            masked_area = inverse_masks * real_images


            with tf.GradientTape() as tape:
                # local discriminator
                global_generated_images = self.generator(masked_images, training=True)
                local_generated_images = global_generated_images * inverse_masks
                local_real_output = self.local_discriminator(masked_area, training = True)
                local_fake_output = self.local_discriminator(local_generated_images, training = True)
                current_local_loss = self._discriminator_loss(local_real_output, local_fake_output)

            # Update local discriminator weights
            grads = tape.gradient(current_local_loss, self.local_discriminator.trainable_weights)
            self.local_optimizer.apply_gradients(zip(grads, self.local_discriminator.trainable_weights))
            local_loss += current_local_loss / self.n_discriminator_updates

        # Train generator
        with tf.GradientTape() as tape:
            global_generated_images = self.generator(masked_images, training=True)
            superimposed_images = real_images * masks + global_generated_images * inverse_masks
            local_generated_images = global_generated_images * inverse_masks
            local_fake_output = self.local_discriminator(local_generated_images, training=False)
            local_adv_loss = self._adversarial_loss(local_fake_output, local_generated_images, masked_area) * self.local_weight
            mae_loss = self._mae_loss(global_generated_images, real_images, inverse_masks) * self.mae_weight
            g_loss = local_adv_loss + mae_loss

        # Update generator weights
        grads = tape.gradient(g_loss, self.generator.trainable_weights)
        self.g_optimizer.apply_gradients(zip(grads, self.generator.trainable_weights))

        # Update loss trackers
        self.local_loss_tracker.update_state(local_loss)
        self.g_loss_tracker.update_state(g_loss)
        self.mae_loss_tracker.update_state(mae_loss)
        self.local_adv_loss_tracker.update_state(local_adv_loss)

        # Return loss metrics
        return {
            "local_loss": self.local_loss_tracker.result(),
            "g_loss": self.g_loss_tracker.result(),
            "mae_loss": self.mae_loss_tracker.result(),
            "local_adv_loss" : self.local_adv_loss_tracker.result()
        }