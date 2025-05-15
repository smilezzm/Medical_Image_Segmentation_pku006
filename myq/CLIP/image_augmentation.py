import random
from typing import Union, Sequence
import torchvision.transforms.functional as TF
import numpy as np
from PIL import Image
import cv2
from scipy.ndimage import gaussian_filter,map_coordinates

'''
定义了一个名为SpatialTransform的类，用于执行空间变换操作，包括随机裁剪、旋转变换、翻转、弹性形变。
输入为一个图像列表，输出为变换后的图像列表(格式为np.ndarray),列表中的图像采取同种变换，目的是图像和掩码同步空间变换。
个人认为旋转、翻转对CT图是不必要的
'''
class SpatialTransform:
    def __init__(self,
                 random_crop: bool = False,  # 是否启用随机裁剪
                 crop_size: Union[int, float] = 0.95,  # 裁剪大小或比例
                 rotate: bool = False,  # 是否启用旋转变换
                 angle: Union[int, Sequence[int]] = 5,  # 旋转角度
                 flip_prob: float = 0,  # 翻转概率
                 elastic_transform: bool = False,  # 是否启用弹性形变
                 elastic_alpha: float = 10,  # 弹性形变强度
                 elastic_sigma: float = 5,  # 弹性形变平滑度
                 alpha_affine: float = 10,  # 仿射变换强度
                ):

        # 随机裁剪参数
        self.random_crop = random_crop
        if isinstance(crop_size, int) and crop_size>1:
            self.crop_size = crop_size#整数，表示裁剪的具体尺寸
        else:
            self.crop_size = float(crop_size)#浮点数，表示裁剪的比例

        # 旋转角度
        self.rotate =rotate
        if isinstance(angle, int):
            self.angles = (-angle, angle)
            self.range_mode = True
        else:
            self.angles = angle
            self.range_mode = False

        # 翻转概率
        self.flip_prob = flip_prob

        # 弹性形变参数
        self.elastic_transform = elastic_transform
        self.elastic_alpha = elastic_alpha
        self.elastic_sigma = elastic_sigma
        self.alpha_affine = alpha_affine


    def __call__(self, images:list) -> list[np.ndarray]:

        # 确保输入是PIL格式或Tensor，如果是ndarray格式则转换为PIL
        for i, image in enumerate(images):
            if isinstance(image, np.ndarray):
                images[i] = Image.fromarray(image.astype(np.uint8))

        # 旋转变换
        if self.rotate:
            if self.range_mode:
                angle = random.uniform(self.angles[0], self.angles[1])
            else:
                angle = random.choice(self.angles)

            for i, image in enumerate(images):
                images[i] = TF.rotate(image, angle=angle)

        # 随机裁剪
        if self.random_crop:
            width, height = images[0].size
            crop_width, crop_height = (self.crop_size,self.crop_size) if isinstance(self.crop_size, int) else (int(width * self.crop_size), int(height * self.crop_size))
            crop_width=crop_height = max(crop_width,crop_height)
            left = random.randint(0, width - crop_width)
            top = random.randint(0, height - crop_height)
            right = left + crop_width
            bottom = top + crop_height
            # 对所有图像应用相同的裁剪
            for i, image in enumerate(images):
                images[i] = image.crop((left, top, right, bottom))

        # 翻转变换
        if random.random() < self.flip_prob:
            # 水平翻转
            for i, image in enumerate(images):
                images[i] = TF.hflip(image)

        # 垂直翻转
        if random.random() < self.flip_prob:
            for i, image in enumerate(images):
                images[i] = TF.vflip(image)

        # 弹性形变
        if self.elastic_transform:
            images=self._elastic_transform(images)

        # 转换回np.ndarray格式
        for i, image in enumerate(images):
            images[i] =np.array(image)

        return images

    def _elastic_transform(self, images, random_state=None):
        """
        源自https://zhuanlan.zhihu.com/p/342274228
        """
        for i, image in enumerate(images):
            if isinstance(image,Image.Image):
                images[i] = np.array(image)

        if random_state is None:
            random_state = np.random.RandomState(None)

        shape = images[0].shape
        shape_size = shape[:2]
        # Random affine
        center_square = np.float32(shape_size) // 2
        square_size = min(shape_size) // 3
        # pts1: 仿射变换前的点(3个点)
        pts1 = np.float32([center_square + square_size,
                           [center_square[0] + square_size,
                            center_square[1] - square_size],
                           center_square - square_size])
        # pts2: 仿射变换后的点
        pts2 = pts1 + random_state.uniform(-self.alpha_affine, self.alpha_affine,
                                           size=pts1.shape).astype(np.float32)
        # 仿射变换矩阵
        M = cv2.getAffineTransform(pts1, pts2)
        # 对image进行仿射变换.
        for i, image in enumerate(images):
            images[i] = cv2.warpAffine(image, M, shape_size[::-1], borderMode=cv2.BORDER_REFLECT_101)

        # generate random displacement fields
        # random_state.rand(*shape)会产生一个和shape一样打的服从[0,1]均匀分布的矩阵
        # *2-1是为了将分布平移到[-1, 1]的区间, alpha是控制变形强度的变形因子
        dx = gaussian_filter((random_state.rand(*shape) * 2 - 1), self.elastic_sigma) * self.elastic_alpha
        dy = gaussian_filter((random_state.rand(*shape) * 2 - 1), self.elastic_sigma) * self.elastic_alpha
        # generate meshgrid
        x, y = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]))
        # x+dx,y+dy
        indices = np.reshape(y + dy, (-1, 1)), np.reshape(x + dx, (-1, 1))
        # bilinear interpolation
        for i, image in enumerate(images):
            images[i] = map_coordinates(image, indices, order=1, mode='reflect').reshape(shape)

        return images

'''
定义了一个名为IntensityTransform的类，用于执行强度变换操作，包括添加高斯噪声、调整亮度、对比度、锐度。
输入为一个图像列表，输出为变换后的图像列表。
有个思考：是否可以将窗宽限制和HU的线性变换作为图像增强的一环添加进去？
'''
class IntensityTransform:
    def __init__(self,
                 noise: bool = False,  # 是否添加高斯噪声
                 noise_std: float = 0.05,  # 高斯噪声的标准差
                 brightness: bool = False,  # 是否调整亮度
                 brightness_range: tuple[float, float] = (0.95, 1.05),  # 亮度调整范围
                 contrast: bool = False,  # 是否调整对比度
                 contrast_range: tuple[float, float] = (0.95, 1.05),  # 对比度调整范围
                 sharpness: bool = False,  # 是否调整锐度
                 sharpness_range: tuple[float, float] = (0.95, 1.05),  # 锐度调整范围
                 ):

        # 噪声参数
        self.noise = noise
        self.noise_std = noise_std

        # 亮度调整参数
        self.brightness = brightness
        self.brightness_range = brightness_range

        # 对比度调整参数
        self.contrast = contrast
        self.contrast_range = contrast_range

        # 锐度调整参数
        self.sharpness = sharpness
        self.sharpness_range = sharpness_range

    def __call__(self, images: list) -> list[np.ndarray]:
        # 将numpy数组转换为PIL图像格式
        for i, image in enumerate(images):
            if isinstance(image, np.ndarray):
                images[i] = Image.fromarray(image.astype(np.uint8))

        # 定义变换顺序并随机打乱，以增加数据多样性
        transform_order = [
            ('brightness', self._adjust_brightness),  # 亮度调整
            ('contrast', self._adjust_contrast),  # 对比度调整
            ('sharpness', self._adjust_sharpness),  # 锐度调整
            ('noise', self._add_gaussian_noise)  # 添加噪声
        ]

        # 随机打乱变换顺序
        random.shuffle(transform_order)

        # 应用每个启用的变换
        for name, transform in transform_order:
            # 检查变换是否启用（注意'noise'对应的是add_noise属性）
            if getattr(self, name, False):
                images = transform(images)

        # 转换回np.ndarray格式
        for i, image in enumerate(images):
            images[i] =np.array(image)

        return images

    def _adjust_brightness(self, images):
        factor = random.uniform(*self.brightness_range)  # 在给定范围内随机生成亮度因子
        for i, image in enumerate(images):
            images[i] = TF.adjust_brightness(image, factor)  # 使用torchvision的功能调整亮度
        return images

    def _adjust_contrast(self, images):
        factor = random.uniform(*self.contrast_range)  # 在给定范围内随机生成对比度因子
        for i, image in enumerate(images):
            images[i] = TF.adjust_contrast(image, factor)  # 使用torchvision的功能调整对比度
        return images

    def _adjust_sharpness(self, images) :
        factor = random.uniform(*self.sharpness_range)  # 在给定范围内随机生成锐度因子
        for i, image in enumerate(images):
            images[i] = TF.adjust_sharpness(image, factor)  # 使用torchvision的功能调整锐度
        return images

    def _add_gaussian_noise(self, images) :
        for i, image in enumerate(images):
            # 将PIL图像转换为numpy数组并归一化到[0,1]范围
            img_array = np.array(image).astype(np.float32) / 255.0

            # 生成高斯噪声并添加到图像
            noise = np.random.normal(0, self.noise_std, img_array.shape)
            noisy_img = img_array + noise

            # 将像素值裁剪到有效范围[0,1]，然后转换回[0,255]范围
            noisy_img = np.clip(noisy_img, 0, 1)
            noisy_img = (noisy_img * 255).astype(np.uint8)

            # 将numpy数组转换回PIL图像
            images[i] = Image.fromarray(noisy_img)

        return images

#测试代码
if __name__ == '__main__':
    lena=lena_=Image.open('lena.png').convert('L')
    # lena.show()
    intensityTransform=IntensityTransform(noise=True,
                                              noise_std=0.05,
                                              brightness=True,
                                              brightness_range=(0.95, 1.05),
                                              contrast=True,
                                              contrast_range=(0.95, 1.05),
                                              sharpness=True,
                                              sharpness_range=(0.95, 1.05),
                                              )
    spatialTransform=SpatialTransform(random_crop= True,
                               crop_size =0.9,
                               rotate=True,
                               angle=5,
                               flip_prob=0,
                               elastic_transform=True,
                               elastic_alpha=20,
                               elastic_sigma=5,
                               alpha_affine=20,
                               )

    lena_transformed=spatialTransform([lena,lena_])
    lena_transformed=intensityTransform(lena_transformed)

    if isinstance(lena_transformed[0], np.ndarray):
        lena_transformed[0] = Image.fromarray(lena_transformed[0].astype(np.uint8))
    lena_transformed[0].show()
