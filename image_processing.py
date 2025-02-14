import cv2
import numpy as np

# Load an image (handles both grayscale and color)
def load_image(image_path):
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Image not found at {image_path}")
    if len(image.shape) == 2 or image.shape[2] == 1:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    else:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image

# Apply Gaussian Blur
def apply_gaussian_blur(image):
    return cv2.GaussianBlur(image, (5, 5), 0)

# Apply Median Blur
def apply_median_blur(image):
    return cv2.medianBlur(image, 5)

# Apply Bilateral Filter
def apply_bilateral_filter(image):
    return cv2.bilateralFilter(image, 9, 75, 75)

# Save enhanced image
def save_image(image, output_path):
    image = np.clip(image, 0, 255).astype(np.uint8)
    cv2.imwrite(output_path, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
