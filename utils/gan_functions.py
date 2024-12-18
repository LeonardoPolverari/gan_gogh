import tensorflow as tf
from tensorflow import keras as tfk
from utils.image_processing import plot_test_images

def masked_mse_loss(y_true, y_pred, mask):
    # Compute squared error
    squared_error = tf.square(y_true - y_pred)
    # Apply mask
    masked_error = squared_error * mask
    # Average over valid areas only
    return tf.reduce_sum(masked_error) / tf.reduce_sum(mask)

def masked_mae_loss(y_true, y_pred, mask):
    abs_error = tf.abs(y_true - y_pred)
    masked_error = abs_error * mask
    return tf.reduce_sum(masked_error) / tf.reduce_sum(mask)

# Define a GANMonitor callback to visualise generated images during training
class GANMonitor(tfk.callbacks.Callback):

    # Initialise the callback with the number of images, latent dimension, name, and colour mode
    def __init__(self, num_img=10, name='', gray=False, test_images=None):
        self.num_img = num_img
        self.name = name
        self.gray = gray
        self.test_images = test_images

    def on_epoch_end(self, epoch, logs=None):
        print(f'finished epoch {epoch}')
        plot_test_images(test_images=self.test_images, generator=self.model.generator)
        if epoch % 2 == 0:
            self.model.save_weights(f'latest.weights.h5')