import cv2
import numpy as np
from tqdm import tqdm
import os
from skimage.metrics import structural_similarity as ssim

# Path to the folder containing images
folder_path = r"C:\Users\91981\Desktop\OnePlus_Photo\Night_images"
output_path = r'C:\Users\91981\Desktop\OnePlus_Photo\Night_Enhanced_Nature_Inspired'

# Create output directory if it doesn't exist
if not os.path.exists(output_path):
    os.makedirs(output_path)

# Get a list of all the image files in the folder
image_files = [f for f in os.listdir(folder_path) if f.endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
total_images = len(image_files)

def psnr(img1, img2):
    """Calculate PSNR between two images."""
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')
    max_pixel = 255.0
    return 20 * np.log10(max_pixel) - 10 * np.log10(mse)

psnr_list = []
ssim_list = []

def enhance_night_images():
    """Enhance night images using a nature-inspired method and calculate PSNR and SSIM."""
    with tqdm(total=total_images, desc="Processing images") as pbar:
        for i, file_name in enumerate(image_files):
            # Load the image
            img_path = os.path.join(folder_path, file_name)
            img = cv2.imread(img_path)
            img = cv2.resize(img, (612, 812))

            # Split the image into its color channels
            b, g, r = cv2.split(img)

            # Perform nature-inspired low light enhancement on each channel
            clahe = cv2.createCLAHE(clipLimit=9.0, tileGridSize=(20, 20))
            enhanced_b = clahe.apply(b)
            enhanced_g = clahe.apply(g)
            enhanced_r = clahe.apply(r)

            # Merge the enhanced channels back into an RGB image
            enhanced_img = cv2.merge((enhanced_b, enhanced_g, enhanced_r))

            # Apply gamma correction for additional enhancement
            gamma = 1.2
            enhanced_img = np.clip((enhanced_img / 255.0) ** gamma * 255.0, 0, 255).astype(np.uint8)

            # Save the enhanced image
            out_path = os.path.join(output_path, file_name)
            cv2.imwrite(out_path, enhanced_img)

            # Calculate PSNR
            psnr_value = psnr(img, enhanced_img)
            psnr_list.append(psnr_value)
            
            # Calculate SSIM
            img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            enhanced_img_gray = cv2.cvtColor(enhanced_img, cv2.COLOR_BGR2GRAY)
            ssim_index = ssim(img_gray, enhanced_img_gray, win_size=5)
            ssim_list.append(ssim_index)

            # Update the progress bar
            pbar.update(1)
            pbar.set_postfix_str("{:.1f}%".format((i+1)/total_images*100))

    # Print the average PSNR and SSIM values
    print("Average PSNR:", np.mean(psnr_list))
    print("Average SSIM:", np.mean(ssim_list))

# Calling the function
enhance_night_images()
