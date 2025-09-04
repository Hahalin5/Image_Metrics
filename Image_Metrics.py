import cv2
import math
import numpy as np
from skimage.metrics import structural_similarity as ssim

### PART1: Implement MSE, MAE, and PSNR. Use implementations to compare Lab 0 results mountains_quantized.png to the original image mountains.png
def calculate_mae(img1, img2):
    """MAE == Mean Absolute Error"""
    height, width = img1.shape
    sum = 0

    for i in range(height):
        for j in range(width):
            diff = abs(int(img1[i][j]) - int(img2[i][j]))
            sum += diff

    return sum / (height * width)

def calculate_mse(img1, img2):
    """MSE == Mean Squared Error"""
    height, width = img1.shape
    sum = 0

    for i in range(height):
        for j in range(width):
            diff = int(img1[i][j]) - int(img2[i][j])
            sum  += diff * diff

    return sum  / (height * width)

def calculate_psnr(img1, img2, max_pixel=255.0):
    """PSNR == Peak Signal-to-Noise Ratio"""
    mse = calculate_mse(img1, img2)
    if mse == 0:
        return float('inf')
    return 20 * math.log10(max_pixel) - 10 * math.log10(mse)

### PART2: Implement image perturbation of adding noise. Compare the original image to the perturbed image using MSE.
def add_noise(img, noise_intensity = 50):
    # to generate a random vlaue centered around 0 with a standard deviation of 50 and the same size as the original image (height * width)
    noise = np.random.normal(0, noise_intensity, img.shape)
    noisy_image = img + noise
    # Ensures pixel values stay within the valid range (0-255) and converts the image to an 8-bit unsigned integer format
    noisy_image = np.clip(noisy_image, 0, 255).astype(np.uint8)
    return noisy_image


# Load images
# Grayscale images have only one channel (intensity) instead of three (RGB), reducing computational complexity.
original_img = cv2.imread('mountains.png', cv2.IMREAD_GRAYSCALE)
quantized_img = cv2.imread('mountains_quantized.png', cv2.IMREAD_GRAYSCALE)

# PART1: Calculate metrics between the original image and the result from lab0
mse_value = calculate_mse(original_img, quantized_img)
mae_value = calculate_mae(original_img, quantized_img)
psnr_value = calculate_psnr(original_img, quantized_img)

# PART2: Image perturbation: adding noise for the original image
noisy_original = add_noise(original_img)
cv2.imwrite('mountains_noisy_original.png', noisy_original)

# Compare the original image to the perturbed image using MSE
mse_value_pertubed = calculate_mse(original_img, noisy_original)

# BONUS: Another image comparison metric: Structural Similarity Index Measure (SSIM)
# Calculate SSIM between the original and quantized image
ssim_quantized = ssim(original_img, quantized_img,)
# Calculate SSIM between the original and noisy image
ssim_noisy = ssim(original_img, noisy_original)


print(f"MAE between original image and the result image from Lab0: {mae_value:.4f}")
print(f"MSE between original image and the result image from Lab0: {mse_value:.4f}")
print(f"PSNR between original image and the result image from Lab0: {psnr_value:.4f}")
print(f" ")
print(f"MSE between original image and the perturbed image by adding noise: {mse_value_pertubed:.4f}")
print(f" ")
print(f"SSIM between original and quantized image: {ssim_quantized:.4f}")
print(f"SSIM between original and noisy image: {ssim_noisy:.4f}")