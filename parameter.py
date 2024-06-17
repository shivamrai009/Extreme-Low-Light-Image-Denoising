import cv2
import os
import numpy as np
from tqdm import tqdm
from skimage.metrics import structural_similarity as ssim

# Path to the folder containing images
folder_path = r"C:\Users\91981\Desktop\OnePlus_Photo\Night_images"
output_path = r'C:\Users\91981\Desktop\OnePlus_Photo\Night_images_enhanced_by_KS'

# Create output directory if it doesn't exist
if not os.path.exists(output_path):
    os.makedirs(output_path)

# Get a list of all the image files in the folder
image_files = [f for f in os.listdir(folder_path) if f.endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
total_images = len(image_files)

def Sharp_Img():
    """Enhance night images based on their average pixel value."""
    # Use tqdm to create a progress bar with message and percentage
    with tqdm(total=total_images, desc="Processing images") as pbar:
        # Loop through all the image files
        for i, file_name in enumerate(image_files):
            # Load the image
            img_path = os.path.join(folder_path, file_name)
            img1 = cv2.imread(img_path)
            img1 = cv2.resize(img1, (812, 612))

            # Apply bilateral filter
            img = cv2.bilateralFilter(img1, 2, 10, 10)

            # Sharpening of image
            gaussian_blur = cv2.GaussianBlur(img, (5, 5), 2)
            img = cv2.addWeighted(img, 1, gaussian_blur, -0.5, 0)

            # Calculate the average pixel value
            avg_pixel_value = np.mean(img1)

            # Apply different conditions for enhancing images based on lighting conditions
            if avg_pixel_value < 10:  # Very dark image
                matrix = np.ones(img.shape, dtype="uint8") * 5
                matrix1 = np.ones(img.shape) * 3
            elif 10 <= avg_pixel_value < 20:  # Medium dark image
                matrix = np.ones(img.shape, dtype="uint8") * 3
                matrix1 = np.ones(img.shape) * 2
            else:  # Rest images
                matrix = np.ones(img.shape, dtype="uint8") * 2
                matrix1 = np.ones(img.shape) * 2

            # Change brightness
            bright = cv2.add(img, matrix)
            dark = cv2.subtract(img, matrix)

            # Change contrast
            contrast_bright = np.uint8(np.clip(cv2.multiply(np.float64(bright), matrix1), 0, 255))

            # Sharpen image based on difference
            diff_img = cv2.absdiff(contrast_bright, img1)
            Sharp_img = cv2.add(3 * diff_img, img1)
            hsv_img = cv2.cvtColor(Sharp_img, cv2.COLOR_BGR2HSV)

            # Increase saturation
            hsv_img[..., 1] = np.uint8(np.clip(hsv_img[..., 1] * 1.3, 20, 150))

            # Convert back to original color space
            sat_img = cv2.cvtColor(hsv_img, cv2.COLOR_HSV2BGR)

            # Apply Non-Local Means Denoising
            denoised_img = cv2.fastNlMeansDenoisingColored(sat_img, None, 3, 3, 20, 15)

            # Save the enhanced image
            out_path = os.path.join(output_path, file_name)
            cv2.imwrite(out_path, denoised_img)

            # Update the progress bar
            pbar.update(1)
            pbar.set_postfix_str("{:.1f}%".format((i + 1) / total_images * 100))

# Calling the function
Sharp_Img()
