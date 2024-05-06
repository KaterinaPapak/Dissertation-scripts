import os
import cv2
import numpy as np
import torch

# Clear the GPU memory cache
torch.cuda.empty_cache()
    
import sys
sys.path.append("..")
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor

# Clear GPU memory
torch.cuda.empty_cache()


# Initialize an instance of the SamAutomaticMaskGenerator class with specified parameters
mask_generator_2 = SamAutomaticMaskGenerator(
    model=sam,  # SAM model for mask generation
    points_per_side=32,  # Number of points per side
    pred_iou_thresh=0.86,  # Threshold for predicted IOU value
    stability_score_thresh=0.50,  # Threshold for stability score
    crop_n_layers=1,  # Number of layers to crop
    crop_n_points_downscale_factor=2,  # Downscale factor for cropping
    min_mask_region_area=100,  # Minimum area for mask region (requires open-cv for post-processing)
)


# Original images directory
original_images_directory = "/vol/research/ap01718/segment-anything/sam_test_images/"

# Output directory for saving images
output_directory = "/vol/research/ap01718/segment-anything/output_images"
os.makedirs(output_directory, exist_ok=True)

# Specify batch size for mask generation
batch_size = 4  # Adjust as needed

for image_number in range(25,31):
    # Input image path
    input_image_path = os.path.join(original_images_directory, f"image_{image_number}.jpg")

    # Read the original image
    original_image = cv2.imread(input_image_path)

    # Generate masks using mask_generator_2
    masks2 = mask_generator_2.generate(original_image)

    # Create a directory for each image
    image_output_directory = os.path.join(output_directory, f"image_{image_number}")
    os.makedirs(image_output_directory, exist_ok=True)

    # Create a directory to store the original image
    original_image_directory = os.path.join(image_output_directory, "image")
    os.makedirs(original_image_directory, exist_ok=True)

    # Copy the original image to the new directory with a modified name
    original_image_output_path = os.path.join(original_image_directory, f"image_{image_number}.jpg")
    cv2.imwrite(original_image_output_path, original_image)

    # Output directories for masks, overlayed masks, and spectrum
    masks_directory = os.path.join(image_output_directory, "masks")
    overlayed_mask_directory = os.path.join(image_output_directory, "overlays")
    spectrum_directory = os.path.join(image_output_directory, "spectrum")

    os.makedirs(masks_directory, exist_ok=True)
    os.makedirs(overlayed_mask_directory, exist_ok=True)
    os.makedirs(spectrum_directory, exist_ok=True)

    # Loop through each mask and create binary masks, overlayed masks, and masked images
    for i, mask in enumerate(masks2):
        # Create a binary mask
        binary_mask = (mask["segmentation"] > 0).astype(np.uint8)

        # Save the binary mask
        binary_mask_output_path = os.path.join(masks_directory, f"image_{image_number}_mask_{i+1}.png")
        cv2.imwrite(binary_mask_output_path, binary_mask * 255)  # Save as binary (0 or 255)

        # Create overlayed mask
        overlayed_mask = original_image.copy()
        color_mask = np.ones((overlayed_mask.shape[0], overlayed_mask.shape[1], 3)) * [128, 0, 128]

        # Ensure color_mask has the same shape as binary_mask
        color_mask = color_mask.reshape((1, overlayed_mask.shape[0], overlayed_mask.shape[1], 3))

        # Repeat the color_mask along dimensions to match the shape of binary_mask
        color_mask = np.tile(color_mask, (1, 1, 1, 1))

        # Create a binary mask with the same shape as overlayed_mask
        binary_mask_resized = cv2.resize(binary_mask, (overlayed_mask.shape[1], overlayed_mask.shape[0]))
        # Resize the binary mask to match the dimensions of original_image
        binary_mask_resized = cv2.resize(binary_mask.astype(np.uint8), (original_image.shape[1], original_image.shape[0]))
        # Apply the color_mask to the overlayed_mask using the resized binary_mask
        overlayed_mask[binary_mask_resized > 0] = color_mask[0, binary_mask_resized > 0]
        # Save the overlayed_mask
        overlayed_mask_output_path = os.path.join(overlayed_mask_directory, f"image_{image_number}_overlay_{i+1}.png")
        cv2.imwrite(overlayed_mask_output_path, overlayed_mask)
        # Free up GPU memory after processing each mask
        torch.cuda.empty_cache() 
