import tensorflow as tf
import tensorflow.keras.layers as tfkl

def get_local_discriminator(input_shape, seed=42, name='local_discriminator_with_gap'):
    # Set random seed for reproducibility
    tf.random.set_seed(seed)

    # Define input layer
    input_layer = tfkl.Input(shape=input_shape, name='input_layer')

    # Mask 0 Pixels to only consider generated area
    masking_layer = tfkl.Masking()(input_layer)

    # First convolutional block
    x = tfkl.Conv2D(32, 3, padding='same', strides=2, name='conv1')(masking_layer)
    x = tfkl.LayerNormalization(name='ln1')(x)
    x = tfkl.LeakyReLU(alpha=0.2, name='activation1')(x)

    # Second convolutional block
    x = tfkl.Conv2D(64, 3, padding='same', strides=2, name='conv2')(x)
    x = tfkl.LayerNormalization(name='ln2')(x)
    x = tfkl.LeakyReLU(alpha=0.2, name='activation2')(x)

    # Third convolutional block
    x = tfkl.Conv2D(128, 3, padding='same', strides=2, name='conv3')(x)
    x = tfkl.LayerNormalization(name='ln3')(x)
    x = tfkl.LeakyReLU(alpha=0.2, name='activation3')(x)

    # Global Average Pooling instead of Flatten
    x = tfkl.GlobalAveragePooling2D(name='gap')(x)

    # Multi-Layer Perceptron (MLP) after GAP
    x = tfkl.Dense(512, name='dense1')(x)  # Increase the number of units to capture more complex features
    x = tfkl.LayerNormalization(name='ln4')(x)
    x = tfkl.LeakyReLU(alpha=0.2, name='activation4')(x)
    
    x = tfkl.Dense(256, name='dense2')(x)  # Add another dense layer for better representation
    x = tfkl.LayerNormalization(name='ln5')(x)
    x = tfkl.LeakyReLU(alpha=0.2, name='activation5')(x)
    
    # Final output layer
    output_layer = tfkl.Dense(1, name='dense_out')(x)

    # Return the discriminator model
    return tf.keras.Model(inputs=input_layer, outputs=output_layer, name=name)
