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
#hu_data = data * slope + intercept


# 查看图像数据的形状
print("Image shape:", image_data.shape)
# 选择一个切片进行显示
# 假设图像数据是三维的 (depth, height, width)，我们选择中间的深度切片
slice_index = image_data.shape[2] // 2  # 选择中间切片
selected_slice = image_data[:, :, 20]

print(selected_slice)
# 显示图像
plt.figure(figsize=(10, 8))
ax = plt.subplot(1, 2, 1)  # 创建一个子图，用于显示灰度图
im = ax.imshow(selected_slice, cmap='gray')  # 使用灰度图显示
ax.set_title(f'Slice {slice_index}')
ax.axis('off')  # 关闭坐标轴

# 添加颜色条（显示数值范围）
plt.colorbar(im, ax=ax, orientation='vertical')  # 将颜色条与图像关联
ax2 = plt.subplot(1, 2, 2)  # 创建另一个子图，用于显示颜色条
ax2.axis('off')  # 关闭坐标轴

plt.tight_layout()
plt.show()
