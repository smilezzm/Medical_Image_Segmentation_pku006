import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt

img = nib.load('1.nii.gz')
image_data = img.get_fdata()
header = img.header

# 方法1：从header直接获取参数（NIfTI）
#slope = header['scl_slope'] if 'scl_slope' in header else 1.0
#intercept = header['scl_inter'] if 'scl_inter' in header else -1024.0

slope = 1
intercept = -1024.0

# 方法2：手动指定（已知参数时）
hu_data = data * slope + intercept

# 查看图像数据的形状
print("Image shape:", image_data.shape)
# 选择一个切片进行显示
# 假设图像数据是三维的 (depth, height, width)，我们选择中间的深度切片
slice_index = image_data.shape[2] // 2  # 选择中间切片
selected_slice = image_data[:, :, 16]

# 显示图像
plt.figure(figsize=(8, 8))
plt.imshow(selected_slice, cmap='gray')  # 使用灰度图显示
plt.title(f'Slice {slice_index}')
plt.axis('off')  # 关闭坐标轴
plt.show()

window_level = 1024
window_width = 40

windowed = np.clip((hu_data - window_level + window_width/2) / window_width, 0, 1)
normalized = (hu_data - np.mean(hu_data)) / np.std(hu_data)
