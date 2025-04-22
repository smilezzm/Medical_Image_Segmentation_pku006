import os
import matplotlib.pyplot as plt
import numpy as np
import nibabel as nib   # Used to load image from .nii.gz
from monai.transforms import LoadImaged   # Another method to load image from .nii.gz
from Fseg import _felzenszwalb_python # Import the function from a self-defined script, which segment the image by voxel intensity.

'''
This is a script testing the LoadImaged function from MONAI and the segmentation function from Fseg.
'''

# Define the path to the .nii.gz file
image_path = os.path.join("./FLARE22Train/images/", "FLARE22_Tr_0002_0000.nii.gz")

# 1. Create the input data dictionary
#    LoadImaged expects a dictionary where keys map to file paths.
data_dict = {'image': image_path}

# 2. Define the keys to load
#    Specify which keys in the dictionary correspond to images to be loaded.
keys_to_load = ['image']

# 3. Instantiate the LoadImaged transform
#    ensure_channel_first=True adds a channel dimension (C,H,W,D), common in MONAI.
#    Set image_only=True if you don't need metadata, False otherwise.
loader = LoadImaged(keys=keys_to_load, ensure_channel_first=True, image_only=True)

# 4. Apply the transform
loaded_data_dict = loader(data_dict)

# 5. Access the loaded data
#    and convert it to a NumPy array
num_image_monai = loaded_data_dict['image']  # an (C,H,W,D) NumPy array

# Now 'num_image_monai' is the NumPy array representation of your image
print(f"Loaded image type with MONAI: {type(num_image_monai)}")
print(f"Loaded image shape with MONAI: {num_image_monai.shape}") # Shape will include channel dim

# --- Original nibabel loading for comparison ---
image_nib = nib.load(image_path)
num_image_nib = image_nib.get_fdata()
print(f"\nOriginal nibabel loaded shape: {num_image_nib.shape}")

# # --- Visualization part (using the MONAI loaded array) ---

# # Note: Slicing needs to account for the channel dimension added by ensure_channel_first=True

# # 获得图像数据形状 (using MONAI loaded array)
# print("图像数据形状为{}".format(num_image_monai.shape))

# def show_slices(slices):
#    """ 显示一行图像切片 """
#    fig, axes = plt.subplots(1, len(slices))
#    for i, slice_data in enumerate(slices): # Renamed slice to slice_data
#        # Display the slice (cmap='gray' is typical for CT, 'jet' was used before)
#        axes[i].imshow(slice_data.T, cmap="gray", origin="lower")
#        axes[i].set_title(f"Slice {i}")
#        axes[i].axis('off')

# # 获得三个维度的切片, slicing the first channel [0]
# # Adjust indices based on the actual shape if needed
# slice_0 = num_image_monai[0, 120, :, :] # Slice along height
# slice_1 = num_image_monai[0, :, 120, :] # Slice along width
# slice_2 = num_image_monai[0, :, :, 70]  # Slice along depth

# show_slices([slice_0, slice_1, slice_2])
# plt.suptitle("Center slices loaded via MONAI")


# --- Apply Fseg ---

print("\nApplying Fseg...")
segmented_image = _felzenszwalb_python(num_image_monai.squeeze())   # Remove channel dim for processing
print("Segmentation complete.")

print(f"Segmented image shape: {segmented_image.shape}")
unique_segment_ids = np.unique(segmented_image)
print(f"Number of unique segments found: {len(unique_segment_ids)}")
print(f"Unique segment IDs (sample): {unique_segment_ids}") # Print all the ids

# --- Visualization of Segmentation ---

def show_Fseg_slice(original_slice, segmented_slice, slice_index, axis_name):
    """ Displays original and segmented slices side-by-side """
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(original_slice.T, cmap="gray", origin="lower")
    axes[0].set_title(f"Original Slice ({axis_name}={slice_index})")
    axes[0].axis('off')

    # Use a qualitative colormap suitable for segmentation maps
    axes[1].imshow(segmented_slice.T, cmap="tab20", origin="lower")
    axes[1].set_title(f"Segmented Slice ({axis_name}={slice_index})")
    axes[1].axis('off')
    plt.tight_layout()

slice_idx_0 = 120
slice_idx_1 = 120
slice_idx_2 = 70

original_slice_0 = num_image_monai.squeeze()[slice_idx_0, :, :]
segmented_slice_0 = segmented_image[slice_idx_0, :, :]

original_slice_1 = num_image_monai.squeeze()[:, slice_idx_1, :]
segmented_slice_1 = segmented_image[:, slice_idx_1, :]

original_slice_2 = num_image_monai.squeeze()[:, :, slice_idx_2]
segmented_slice_2 = segmented_image[:, :, slice_idx_2]

# Show the slices
show_Fseg_slice(original_slice_0, segmented_slice_0, slice_idx_0, "Axis 0")
show_Fseg_slice(original_slice_1, segmented_slice_1, slice_idx_1, "Axis 1")
show_Fseg_slice(original_slice_2, segmented_slice_2, slice_idx_2, "Axis 2")

plt.show()