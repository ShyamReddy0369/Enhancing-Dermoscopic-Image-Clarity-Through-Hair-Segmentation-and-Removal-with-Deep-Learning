import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
import os
import imgaug.augmenters as iaa  # Import imgaug for augmentation

# Create output folder
os.makedirs("enhanced", exist_ok=True)

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


# Load an image (handles both grayscale and color)
def load_image(image_path):
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Image not found at {image_path}")
    # Convert grayscale (1 or 2 channels) to RGB
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


# Compare filters on the same image
def compare_filters(image_path, output_prefix):
    try:
        original_image = load_image(image_path)
        augmented_image = augment_image(original_image)  # Call the augmentation function

        height, width = original_image.shape[:2]  # For original image

        min_dim = min(height, width)  # For original image
        win_size = min(7, min_dim)  # For original image
        win_size = win_size if win_size % 2 == 1 else win_size - 1

        if win_size < 3:
            print(f"Skipping {output_prefix}: Image too small ({height}x{width})")
            return

        # Apply filters to original image
        gaussian_blurred = apply_gaussian_blur(original_image)
        median_blurred = apply_median_blur(original_image)
        bilateral_filtered = apply_bilateral_filter(original_image)

        # Apply filters to augmented image
        gaussian_blurred_aug = apply_gaussian_blur(augmented_image)
        median_blurred_aug = apply_median_blur(augmented_image)
        bilateral_filtered_aug = apply_bilateral_filter(augmented_image)

        # Save enhanced images (save both original and augmented versions)
        save_image(gaussian_blurred, f"enhanced/{output_prefix}_gaussian_blurred.jpg")
        save_image(gaussian_blurred_aug, f"enhanced/{output_prefix}_gaussian_blurred_aug.jpg")

        save_image(median_blurred, f"enhanced/{output_prefix}_median_blurred.jpg")
        save_image(median_blurred_aug, f"enhanced/{output_prefix}_median_blurred_aug.jpg")

        save_image(bilateral_filtered, f"enhanced/{output_prefix}_bilateral_filtered.jpg")
        save_image(bilateral_filtered_aug, f"enhanced/{output_prefix}_bilateral_filtered_aug.jpg")

        data_range = 255
        ssim_params = {"win_size": win_size, "channel_axis": -1 if original_image.shape[-1] == 3 else None, "data_range": data_range}

        # Calculate PSNR and SSIM for original image
        psnr_gaussian = psnr(original_image, gaussian_blurred, data_range=data_range)
        psnr_median = psnr(original_image, median_blurred, data_range=data_range)
        psnr_bilateral = psnr(original_image, bilateral_filtered, data_range=data_range)

        ssim_gaussian = ssim(original_image, gaussian_blurred, **ssim_params)
        ssim_median = ssim(original_image, median_blurred, **ssim_params)
        ssim_bilateral = ssim(original_image, bilateral_filtered, **ssim_params)

        # Calculate PSNR and SSIM for augmented image
        psnr_gaussian_aug = psnr(augmented_image, gaussian_blurred_aug, data_range=data_range)
        psnr_median_aug = psnr(augmented_image, median_blurred_aug, data_range=data_range)
        psnr_bilateral_aug = psnr(augmented_image, bilateral_filtered_aug, data_range=data_range)

        ssim_gaussian_aug = ssim(augmented_image, gaussian_blurred_aug, **ssim_params)
        ssim_median_aug = ssim(augmented_image, median_blurred_aug, **ssim_params)
        ssim_bilateral_aug = ssim(augmented_image, bilateral_filtered_aug, **ssim_params)

        # Print results for both original and augmented images
        print(f"Results for {output_prefix} ({height}x{width}):")
        print(f"Original Image")
        print(f"Gaussian Blur - PSNR: {psnr_gaussian:.2f}, SSIM: {ssim_gaussian:.2f}")
        print(f"Median Blur - PSNR: {psnr_median:.2f}, SSIM: {ssim_median:.2f}")
        print(f"Bilateral Filter - PSNR: {psnr_bilateral:.2f}, SSIM: {ssim_bilateral:.2f}")
        print(f"Augmented Image")
        print(f"Gaussian Blur - PSNR: {psnr_gaussian_aug:.2f}, SSIM: {ssim_gaussian_aug:.2f}")
        print(f"Median Blur - PSNR: {psnr_median_aug:.2f}, SSIM: {ssim_median_aug:.2f}")
        print(f"Bilateral Filter - PSNR: {psnr_bilateral_aug:.2f}, SSIM: {ssim_bilateral_aug:.2f}")
        print("-" * 40)

        # Plotting: Plot both original and augmented + filtered images
        plt.figure(figsize=(15, 10))
        titles = ["Original", "Gaussian Blur", "Median Blur", "Bilateral Filter"]
        images = [original_image, gaussian_blurred, median_blurred, bilateral_filtered]

        for i in range(4):
            plt.subplot(2, 2, i + 1)
            plt.title(titles[i])
            plt.imshow(images[i])
            plt.axis('off')

        plt.tight_layout()
        plt.show()

        plt.figure(figsize=(15, 10))
        titles = ["Augmented", "Gaussian Blur", "Median Blur", "Bilateral Filter"]
        images = [augmented_image, gaussian_blurred_aug, median_blurred_aug, bilateral_filtered_aug]

        for i in range(4):
            plt.subplot(2, 2, i + 1)
            plt.title(titles[i])
            plt.imshow(images[i])
            plt.axis('off')

        plt.tight_layout()
        plt.show()

    except Exception as e:
        print(f"Error processing {image_path}: {str(e)}")


# Main function
if __name__ == "__main__":
    image_paths = [
        "data/raw/image1.jpg",
        "data/raw/image2.jpg",
        "data/raw/image3.jpg"
    ]

    for i, image_path in enumerate(image_paths):
        print(f"Processing image {i + 1}: {image_path}")
        if not os.path.exists(image_path):
            print(f"Image not found: {image_path}")
            continue
        compare_filters(image_path, f"image{i + 1}")
