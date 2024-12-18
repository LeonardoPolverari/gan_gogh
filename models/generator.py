import tensorflow as tf
from tensorflow.keras import layers, models
import keras

# Positional Encoding (simple implementation)
@keras.saving.register_keras_serializable(package="MyGAN")
class PositionEncoding(layers.Layer):
    def __init__(self, d_model, **kwargs):
        super(PositionEncoding, self).__init__(**kwargs)
        self.d_model = d_model

    def call(self, x):
        seq_len = tf.shape(x)[1]
        pos = tf.range(0, seq_len, dtype=tf.float32)[:, tf.newaxis]
        i = tf.range(0, self.d_model, 2, dtype=tf.float32)
        angle_rates = 1 / tf.pow(10000.0, (i / tf.cast(self.d_model, tf.float32)))
        angle_rads = pos * angle_rates
        pos_encoding = tf.concat([tf.sin(angle_rads), tf.cos(angle_rads)], axis=-1)
        return x + pos_encoding[tf.newaxis, ...]

    def get_config(self):
        # Correctly serialize the integer value of d_model
        base_config = super().get_config()
        config = {
            "d_model": self.d_model,
        }
        return {**base_config, **config}
        

    @classmethod
    def from_config(cls, config):
        d_model_config = config.pop('d_model')
        d_model = d_model_config
        return cls(d_model, **config)
    

# Attention Block (TSA - Transformer Style Attention)
@keras.saving.register_keras_serializable(package="MyGAN")
class TSA_Block(layers.Layer):
    def __init__(self, d_model, **kwargs):
        super(TSA_Block, self).__init__(**kwargs)
        self.d_model = d_model
        self.q_dense = layers.Dense(d_model)
        self.k_dense = layers.Dense(d_model)
        self.v_dense = layers.Dense(d_model)
        self.add = layers.Add()
        self.softmax = layers.Softmax(axis=-1)
    
    def call(self, inputs):
        # Positional Encoding
        pos_encoded = PositionEncoding(self.d_model)(inputs)
        
        # Linear projections
        q = self.q_dense(pos_encoded)
        k = self.k_dense(pos_encoded)
        v = self.v_dense(pos_encoded)
        
        # Scaled dot-product attention
        attention_logits = tf.matmul(q, k, transpose_b=True)
        attention_logits /= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        attention_weights = self.softmax(attention_logits)
        attention_output = tf.matmul(attention_weights, v)
        
        # Skip connection
        return self.add([inputs, attention_output])

    def get_config(self):
        # Correctly serialize the integer value of d_model
        base_config = super().get_config()
        config = {
            "d_model": self.d_model,
        }
        return {**base_config, **config}
        

    @classmethod
    def from_config(cls, config):
        d_model_config = config.pop('d_model')
        d_model = d_model_config
        return cls(d_model, **config)


# Grouped Convolution Block (GSA)
def GSA_Block(inputs, filters):
    conv_w = layers.Conv2D(filters, 3, padding="same", activation="relu")(inputs)
    conv_m = layers.Conv2D(filters, 3, padding="same", activation="relu")(inputs)
    conv_n = layers.Conv2D(filters, 3, padding="same", activation="relu")(inputs)
    combined = layers.Multiply()([conv_w, conv_m, conv_n])
    return layers.Add()([inputs, combined])

# Encoder Block
def EncoderBlock(inputs, filters):
    x = layers.Conv2D(filters, 3, padding="same", activation="relu")(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU()(x)
    pooled = layers.MaxPooling2D(pool_size=(2, 2))(x)
    return x, pooled

# Decoder Block
def DecoderBlock(inputs, skip, filters):
    x = layers.UpSampling2D(size=(2, 2), interpolation="bilinear")(inputs)
    x = layers.Concatenate()([x, skip])
    x = layers.Conv2D(filters, 3, padding="same", activation="relu")(x)
    x = layers.LeakyReLU()(x)
    x = layers.BatchNormalization()(x)
    return x

# Full Model
def build_model(input_shape=(256, 256, 1), channels=1):
    inputs = layers.Input(shape=input_shape)

    # Encoder
    enc1, pool1 = EncoderBlock(inputs, 64)
    enc2, pool2 = EncoderBlock(pool1, 128)
    enc3, pool3 = EncoderBlock(pool2, 256)

    # Transformer Attention (TSA) + Grouped Attention (GSA)
    tsa = TSA_Block(256)(pool3)
    gsa = GSA_Block(pool3, 256)
    combined = layers.Add()([tsa, gsa])

    # Decoder
    dec3 = DecoderBlock(combined, enc3, 256)
    dec2 = DecoderBlock(dec3, enc2, 128)
    dec1 = DecoderBlock(dec2, enc1, 64)

    # Output Layer
    outputs = layers.Conv2D(channels, 1, activation="tanh")(dec1)

    return models.Model(inputs, outputs)