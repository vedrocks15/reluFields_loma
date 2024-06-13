import numpy as np
import imageio
import os
import torch
import torchvision.transforms as transforms
from lpips import LPIPS
from PIL import Image
import warnings
warnings.filterwarnings("ignore")

curr_dir = os.getcwd()
save_dir = os.path.join(os.getcwd(), "reluField_results")

def psnr(img1_path, img2_path):
    # Load images
    img1 = imageio.imread(img1_path).astype(np.float32)
    img2 = imageio.imread(img2_path).astype(np.float32)

    # Check if images are of same shape
    if img1.shape != img2.shape:
        raise ValueError("Images must have the same dimensions")

    # Calculate MSE (Mean Squared Error)
    mse = np.mean((img1 - img2) ** 2)

    # If MSE is zero, PSNR is infinity (perfect similarity)
    if mse == 0:
        return float('inf')

    # Calculate PSNR
    max_pixel = 255.0
    psnr = 20 * np.log10(max_pixel / np.sqrt(mse))

    return psnr

def lpips_similarity(img1_path, img2_path):
    # Load images
    img1 = transforms.ToTensor()(Image.open(img1_path).convert('L')).unsqueeze(0)
    img2 = transforms.ToTensor()(Image.open(img2_path).convert('L')).unsqueeze(0)

    # Initialize LPIPS model
    lpips_model = LPIPS(net='vgg')

    # Compute LPIPS distance
    distance = lpips_model(img1, img2)

    return distance.item()



# Example usage:
if __name__ == "__main__":
    og_img_path = os.path.join(curr_dir,"demo_train.jpg")
    og_image = Image.open(og_img_path).convert('L')
    w,h = og_image.size
    og_shape = (h,w)
    grid_size = (200,200)
    reluFields_image = os.path.join(save_dir, "test_inference_shape_{}_{}_grid_size_{}_{}.jpg".format(og_shape[0], 
                                                                                                      og_shape[1],
                                                                                                      grid_size[0],
                                                                                                      grid_size[1]))

    value_psnr = psnr(og_img_path, reluFields_image)
    print(f"PSNR value is {value_psnr} dB")

    lpips_value = lpips_similarity(og_img_path, reluFields_image)
    print(f"LPIPS similarity is {lpips_value}")
