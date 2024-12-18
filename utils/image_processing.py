import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import tarfile

def display_image(image):
    # Denormalize the image to the range [0, 1] for visualization
    image = (image + 1) / 2.0
    plt.figure(figsize=(1.5, 1.5))
    plt.imshow(image, cmap = 'gray')
    plt.axis('off')
    plt.show()

def apply_random_patches(image, patch_shape = (64, 64, 1), return_coords = False, image_height = 256, image_width = 256):
    """
    Applies random masks (filled with zeros) to an image.

    Args:
        image: Tensor of shape (height, width, channels).
        patch_height: Height of the patch.
        patch_width: Width of the patch.
        num_patches: Number of patches to apply.

    Returns:
        Tensor with patches applied.
    """
    # batch_size, img_height, img_width, channels = batch_size, img_height, img_width, channels
    mask = tf.ones((image_height, image_width), dtype=tf.float32)
    patch_width, patch_height = patch_shape[:2]

    # Random top-left corner for the patch
    x_start = tf.random.uniform((), minval=0, maxval=image_width - patch_width, dtype=tf.int32)
    y_start = tf.random.uniform((), minval=0, maxval=image_height - patch_height, dtype=tf.int32)

    # Create patch mask
    patch_coords = tf.stack(tf.meshgrid(
        tf.range(y_start, y_start + patch_height),
        tf.range(x_start, x_start + patch_width),
        indexing='ij'
    ), axis=-1)
    patch_coords = tf.reshape(patch_coords, [-1, 2])  # Flatten the coordinates

    mask = tf.tensor_scatter_nd_update(
        mask,
        indices=patch_coords,
        updates=tf.zeros(patch_height * patch_width, dtype=tf.float32)
    )

    # Add channel dimension to the mask
    mask = tf.expand_dims(mask, axis=-1)

    # Apply the mask to the image
    patched_image = image * mask
    if return_coords:
      patched_image = patched_image, patch_coords
    return patched_image

# Function to extract, preprocess images, convert to grayscale, and apply patching
def extract_and_preprocess_images(tar_path, patch_shape, return_inverse = False, image_height = 256, image_width = 256, channels = 1):
    """Extracts images from a tar file, converts them to grayscale, resizes them,
    and applies random patches for inpainting."""
    with tarfile.open(tar_path, 'r') as tar:
        for member in tar.getmembers():
            if member.isfile() and member.name.lower().endswith(('.png', '.jpg', '.jpeg')):  # Check for image files
                # Extract the image file
                f = tar.extractfile(member)
                image_data = f.read()

                # Decode and preprocess the image using TensorFlow
                image = tf.image.decode_image(image_data, channels=3, expand_animations=False)
                image = tf.image.resize(image, [image_height, image_width])

                # Convert to grayscale
                image_gray = tf.image.rgb_to_grayscale(image)
                image_gray = (image_gray / 127.5) - 1  # Normalize to [-1, 1]

                mask = tf.ones((image_height, image_width, channels), dtype=tf.float32)
                mask = apply_random_patches(mask, patch_shape)
                inverse_mask = tf.ones((image_height, image_width, channels), dtype=tf.float32) - mask

                if return_inverse:
                    yield image_gray, mask, inverse_mask  # Yield the preprocessed image with patch
                else:
                    yield image_gray, mask

def plot_test_images(test_images, generator, titles=None, n_imgs = 4):
    """
    Plots the first four images from the test batch with their respective titles.

    Parameters:
        test_images (list): A list containing input images and masks for testing.
        generator (Model): The trained generator model for generating images.
        titles (list): List of titles for the images, defaulted to None.
    """
    if titles is None:
        titles = ['Original Image', 'Patched Image', 'Generated Image', 'Superimposed Image']


    fig, axes = plt.subplots(n_imgs, 4, figsize=(7, 7))
    for i in range(n_imgs):
        original_image = test_images[0][i]
        patched_image = test_images[0][i] * test_images[1][i]
        generated_image = generator(test_images[0] * test_images[1]).numpy()[i]
        inverse_mask = test_images[2][i]
        superimposed_image = patched_image + (inverse_mask * generated_image)

        # Collect images for the grid
        images = [original_image, patched_image, generated_image, superimposed_image]
        
        for j, img in enumerate(images):
            ax = axes[i, j]
            ax.imshow(img, cmap = 'gray', vmin=-1, vmax=1)
            ax.axis('off')
            ax.set_title(titles[j], fontsize=5)

    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # Apply random patches
    image = np.random.rand(3, 256, 256, 1)
    patched_image, coords = apply_random_patches(image, return_coords = True)

    # Display results
    display_image(tf.squeeze(image[0]))
    display_image(tf.squeeze(patched_image[0]))

