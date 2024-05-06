import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from scipy.io import loadmat
from scipy.io import savemat
from tqdm import tqdm
import tifffile
import h5py


# Define the dimensions of the matrix
width = 1480
height = 1974

# Define the number of spectra and the step size for selection
channel = 3468
step_size = 8

# Initialize an empty array to store the combined masks
mat = np.zeros((height, width, 456))

# Define the main directory containing the image folders
main_directory = r"C:\Users\Lenovo ThinkPad\OneDrive - University of Surrey\Desktop\disso_datacubes\Dataset_test" #add the correct path

# Function to convert mask image to binary format
def convert_to_binary_mask(mask):
    threshold = 128  # Threshold for converting grayscale to binary
    binary_mask = np.zeros_like(mask)
    binary_mask[mask > threshold] = 1
    return binary_mask

global_min =  -34808.76953125
global_max = 53336.6171875

# Iterate over each image folder
for folder_name in sorted(os.listdir(main_directory)):
    if folder_name.startswith("image_") and os.path.isdir(os.path.join(main_directory, folder_name)):
        image_folder = os.path.join(main_directory, folder_name)
        masks_directory = os.path.join(image_folder, "new_masks")
        spectrum_directory = os.path.join(image_folder, "new_spectrum")

        # Initialize an empty list to store loaded masks and spectra
        mask_files = sorted(os.listdir(masks_directory))
        spectrum_files = sorted(os.listdir(spectrum_directory))

        # Iterate over mask and spectrum files simultaneously
        for mask_file, spectrum_file in tqdm(zip(mask_files, spectrum_files)):
            # Extract image number from filenames
            image_num_mask = mask_file.split("_")[1]
            image_num_spectrum = spectrum_file.split("_")[1]

            # Ensure filenames match and belong to the same image
            if image_num_mask == image_num_spectrum and mask_file.endswith(".png") and spectrum_file.endswith(".mat"):
                # Load mask
                mask = np.array(Image.open(os.path.join(masks_directory, mask_file)))
                # Convert mask to binary format
                mask_binary = convert_to_binary_mask(mask)

                # Load spectrum using scipy.io.loadmat()
                spectrum_data = loadmat(os.path.join(spectrum_directory, spectrum_file))
                print("Keys in spectrum data:", spectrum_data.keys())

                # Check if 'spectrum' key exists and has data
                if 'spectrum' in spectrum_data:
                    # Retrieve spectrum data
                    spectrum = spectrum_data['spectrum']
                    # Check if spectrum is a string 'N/A'
                    if type(spectrum[0]) == str:
                        print(f"Spectrum data for {spectrum_file} is not available.")
                        spectrum = [-1 for i in range(3648)]
                        spectrum = np.array(spectrum)
                       # breakpoint()
                        continue  # Skip further processing for this spectrum file

                    # Convert spectrum to numpy array and handle any additional preprocessing
                    spectrum = np.array(spectrum).squeeze(0)

                    # Check if spectrum has more than one dimension
                    if len(spectrum.shape) != 1:
                        print(f"Spectrum data for {spectrum_file} is not available.")
                        spectrum = [-1 for i in range(3648)]
                        spectrum = np.array(spectrum)
                       # breakpoint()
                        # continue  # Skip further processing for this spectrum file
                else:
                    print("No spectrum data found.")

                spectrum = spectrum[::8]
                spectrum = (spectrum - global_min) / (global_max - global_min)


                # Iterate over each pixel in the binary mask
                for x in range(height):
                    for y in range(width):
                        # Check if mask value is 1 in the binary mask
                         print(f'x: {x}, y:{y} mask.shape: {mask.shape}')
                         if mask_binary[x, y] == 1:
                            # Assign spectrum to corresponding position in mat
                            mat[x, y, :] = spectrum
                            print(spectrum.max())
 
    with h5py.File(os.path.join(main_directory, f"{folder_name}_matrix.mat"), 'w') as file:file.create_dataset('datacube', data=mat)
    print("datacube saved sucessfully")