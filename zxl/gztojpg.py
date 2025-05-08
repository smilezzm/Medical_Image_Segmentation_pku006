import nibabel as nib
import numpy as np
from PIL import Image

import os
print("Current working directory:", os.getcwd())

# 加载NIfTI图像
img = nib.load('./1.nii.gz')
image_data = img.get_fdata()

# 查看图像数据的形状
print("Image shape:", image_data.shape)

# 选择一个切片进行显示（例如，中间切片）
slice_index = image_data.shape[2] // 2  # 选择中间切片
selected_slice = image_data[:, :, 46]

# 将数据转换为图像格式并保存为.jpg
# 将数据缩放到0-255的范围（假设数据是16位的，常见于医学图像）
min_val = np.min(selected_slice)
max_val = np.max(selected_slice)
scaled_slice = ((selected_slice - min_val) / (max_val - min_val) * 255).astype(np.uint8)

# 使用PIL保存为JPG
print("Scaled slice shape:", scaled_slice.shape)
print("Scaled slice dtype:", scaled_slice.dtype)

img = Image.fromarray(scaled_slice)
img.save('./slice_{}.jpg'.format(slice_index))

print(f"Saved slice {slice_index} as JPG")
