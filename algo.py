import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim
import os
from tqdm import tqdm

def enhance_image(image):
    """
    Apply the proposed decomposition and enhancement algorithm to an image.
    
    Args:
    image (np.array): Input image in BGR format.
    
    Returns:
    np.array: Enhanced image in BGR format.
    """
    # Convert the image to HSV space
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Normalize the V channel
    norm_V = hsv[:,:,2] / 255.0

    # Apply the proposed decomposition
    L = np.exp(-1 * np.log(norm_V + 0.001))
    R = norm_V / L

    # Adjust the illumination
    L_adjusted = cv2.normalize(L, None, alpha=0.5, beta=400, norm_type=cv2.NORM_MINMAX)

    # Generate the enhanced V channel image
    enhanced_V = L_adjusted * R
    enhanced_V = cv2.normalize(enhanced_V, None, alpha=0, beta=1500, norm_type=cv2.NORM_MINMAX)
    enhanced_V = enhanced_V.astype(np.uint8)

    # Merge the enhanced V channel with the H and S channels to get the enhanced HSV image
    enhanced_hsv = cv2.merge((hsv[:,:,0], hsv[:,:,1], enhanced_V))

    # Convert the enhanced HSV image to RGB space
    enhanced_image = cv2.cvtColor(enhanced_hsv, cv2.COLOR_HSV2BGR)
  
    return enhanced_image

def psnr(img1, img2):
    """
    Calculate the PSNR (Peak Signal-to-Noise Ratio) between two images.
    
    Args:
    img1 (np.array): First image.
    img2 (np.array): Second image.
    
    Returns:
    float: PSNR value.
    """
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')
    max_pixel = 255.0
    return 20 * np.log10(max_pixel) - 10 * np.log10(mse)

def calculate_ssim(img1, img2):
    """
    Calculate the SSIM (Structural Similarity Index) between two images.
    
    Args:
    img1 (np.array): First image in grayscale.
    img2 (np.array): Second image in grayscale.
    
    Returns:
    float: SSIM value.
    """
    return ssim(img1, img2, win_size=5)

# Path to the folder containing images
folder_path = r"C:\Users\91981\Desktop\OnePlus_Photo\Night_images"
output_path = r'C:\Users\91981\Desktop\OnePlus_Photo\Night_Enhanced_by_RP'

# Create output directory if it doesn't exist
if not os.path.exists(output_path):
    os.makedirs(output_path)

# Get a list of all the image files in the folder
image_files = [f for f in os.listdir(folder_path) if f.endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
total_images = len(image_files)

# Use tqdm to create a progress bar with message and percentage
with tqdm(total=total_images, desc="Processing images") as pbar:
    psnr_values = []
    ssim_values = []
    
    for i, file_name in enumerate(image_files):
        # Load the image
        img_path = os.path.join(folder_path, file_name)
        image = cv2.imread(img_path)
        if image is None:
            print(f"Warning: Unable to read {file_name}")
            continue
        
        image = cv2.resize(image, (612, 812))
        
        # Apply the enhancement algorithm
        enhanced_image = enhance_image(image)

        # Save the enhanced image
        out_path = os.path.join(output_path, file_name)
        cv2.imwrite(out_path, enhanced_image)

        # Calculate PSNR
        psnr_value = psnr(image, enhanced_image)
        psnr_values.append(psnr_value)

        # Calculate SSIM
        image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        enhanced_image_gray = cv2.cvtColor(enhanced_image, cv2.COLOR_BGR2GRAY)
        ssim_index = calculate_ssim(image_gray, enhanced_image_gray)
        ssim_values.append(ssim_index)

        # Update the progress bar
        pbar.update(1)
        pbar.set_postfix_str("{:.1f}%".format((i+1)/total_images*100))

# Output results
print("Average PSNR:", np.mean(psnr_values))
print("Average SSIM:", np.mean(ssim_values))
