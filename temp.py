import os
import matplotlib.pyplot as plt
import numpy as np
import nibabel as nib

path=os.path.join("./pku_train_dataset/ct","1.nii.gz")
image = nib.load(path)
num_image = image.get_fdata()
print(image.header)
##########
unique_labels = np.unique(num_image)
print("Unique label values:", unique_labels)

# 获得图像数据形状
print("图像数据形状为{}".format(num_image.shape))

def show_slices(slices):
   """ 显示一行图像切片 """
   fig, axes = plt.subplots(1, len(slices))
   for i, slice in enumerate(slices):
       axes[i].imshow(slice.T, cmap="jet", origin="lower")
# 获得三个维度的切片
slice_0 = num_image[120, :, :]
slice_1 = num_image[:, 120, :]
slice_2 = num_image[:, :, 13]
show_slices([slice_0, slice_1, slice_2])
plt.suptitle("Center slices for brain tumor image")
plt.show()