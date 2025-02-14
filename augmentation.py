import imgaug.augmenters as iaa
import numpy as np

# Data Augmentation Function
def augment_image(image):
    seq = iaa.Sequential([
        iaa.Fliplr(0.5),  # Horizontal flip 50% of the time
        iaa.Flipud(0.2),  # Vertical flip 20% of the time
        iaa.Affine(rotate=(-20, 20)),  # Random rotation between -20 and 20 degrees
        iaa.Affine(scale={"x": (0.8, 1.2), "y": (0.8, 1.2)}),  # Scale between 80% and 120%
        iaa.Sometimes(0.5, iaa.GaussianBlur(sigma=(0, 0.5))),  # Blur 50% of the time
        iaa.AdditiveGaussianNoise(scale=(0, 0.05 * 255)),  # Add noise
        iaa.ContrastNormalization((0.75, 1.5)),  # Change contrast
    ])

    # The imgaug library expects images in a specific format (uint8)
    image = np.clip(image, 0, 255).astype(np.uint8)  # Important
    seq_det = seq.to_deterministic()  # Call this once to determine the augmentation parameters
    augmented_image = seq_det.augment_images([image])[0]  # Apply augmentation
    return augmented_image
