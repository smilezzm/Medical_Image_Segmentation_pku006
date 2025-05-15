import nibabel as nib
from glob import glob
import numpy as np
import os
import json
from transformers import CLIPSegProcessor
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from tqdm import tqdm
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from typing import Union, Sequence
import random
from image_augmentation import SpatialTransform, IntensityTransform

'''
做了大量修改。

需要注意的细节：processor处理text和images时，不能单独对每一条数据进行process最后整合，而应将所有数据的text,images合成为
各自的list，然后一并传入processor进行处理。
换言之，processor(text, images, ...)中text是一个list，包含所有数据的organ名称，images也是一个list，包含所有数据的图像信息。

另外，原始每个数据的labels是512*512 ndarray的binary mask，需要先转换成张量，再加入channel dim，再把所有数据的labels整合成一个(N,1,512,512)的tensor
最后再resize成(N,1,352,352)的tensor，然后squeeze成(N,352,352)的tensor。
先加入channel dim是为了后面resize时能够使用interpolate函数。
最后再squeeze掉channel dim，是为了之后传入模型时能顺利计算loss

pku的数据集已经是HU值，且slope和intercept都是nan，因此不需要进行转换！

在进行processor时，images里的图像需要是uint8类型的ndarray，且值在[0,255]之间，而且channel需要为3（这里直接复制3次，造出了3个channel）

processor 输入输出的说明：
processor(text=[str,str,...], images=[(512,512,3),(512,512,3),...], padding=True, truncation=True, return_tensors="pt")
-> input_ids: tensor(N, embedding_dim), pixel_values: tensor(N, 3, 352, 352), attention_mask: tensor(N, embedding_dim)

最后数据集的说明：
'input_ids': tensor(N, embedding_dim), 
'pixel_values': tensor(N, 3, 352, 352), 
'attention_mask': tensor(N, embedding_dim), 
'labels': tensor(N, 352, 352)
'''

image_dir = "./pku_train_dataset/ct"
mask_dir = "./pku_train_dataset/label"
json_dir = "./pku_train_dataset/json"
N_slices = 30
N_neg_slices = 5
seed = 42


class MyRotateTransform:
    def __init__(self, angle: Union[int, Sequence[int]]):
        """
        初始化旋转变换

        参数:
            angle: 可以是一个整数(表示正负范围)或一个序列(表示具体角度集合)
                  例如:
                  - 15 表示在[-15, 15]度范围内随机旋转
                  - [0, 90, 180, 270] 表示从这些固定角度中随机选择
        """
        if isinstance(angle, int):
            self.angles = (-angle, angle)
            self.range_mode = True
        else:
            self.angles = angle
            self.range_mode = False

    def __call__(self, x):
        if self.range_mode:
            angle = random.uniform(self.angles[0], self.angles[1])
        else:
            angle = random.choice(self.angles)
        return TF.rotate(x, angle)


def build_examples(image_dir, mask_dir, json_dir, N_slices, N_neg_slices, seed,
                   slope, intercept, hu_clip_min, hu_clip_max):
    np.random.seed(seed)
    processor = CLIPSegProcessor.from_pretrained("CIDAS/clipseg-rd64-refined", cache_dir="./CLIP/hf_cache")
    # 如果输入图像被归一化，需要修改processor的参数，避免对图像进行rescale
    # processor.image_processor.do_rescale = False

    raw_data = {'organs': [], 'images': [], 'labels': []}
    image_paths = sorted(glob(f"{image_dir}/*.nii.gz"))
    mask_paths = sorted(glob(f"{mask_dir}/*.nii.gz"))

    intensityTransform = IntensityTransform(noise=True,
                                            noise_std=0.02,
                                            brightness=True,
                                            brightness_range=(0.95, 1.05),
                                            contrast=True,
                                            contrast_range=(0.95, 1.05),
                                            sharpness=True,
                                            sharpness_range=(0.95, 1.05),
                                            )
    spatialTransform = SpatialTransform(random_crop=True,
                                        crop_size=0.9,
                                        rotate=False,
                                        angle=5,
                                        flip_prob=0,
                                        elastic_transform=True,
                                        elastic_alpha=5,
                                        elastic_sigma=5,
                                        alpha_affine=5,
                                        )

    assert len(image_paths) == len(mask_paths)
    for img_path, msk_path in tqdm(zip(image_paths, mask_paths), total=len(image_paths), desc="cases"):
        mask_3d = nib.load(msk_path).get_fdata().astype(int)
        image_3d = nib.load(img_path).get_fdata()
        with open(os.path.join(json_dir, os.path.basename(img_path).replace(".nii.gz", ".json"))) as f:
            info = json.load(f)

        for organ, idx in info["label_mapping"].items():
            pos = np.where(np.any(mask_3d == idx, axis=(0, 1)))[0]
            if len(pos) == 0: continue

            sel = np.random.choice(pos, size=min(N_slices, len(pos)), replace=False)
            neg = np.setdiff1d(np.arange(mask_3d.shape[-1]), pos)
            sel_neg = np.random.choice(neg, size=min(N_neg_slices, len(neg)), replace=False)

            for slice_i in np.concatenate((sel, sel_neg)):
                # pku_dataset里原始图像就是HU值，不用转换
                # hu_slice = image_3d[:,:,slice_i] * slope + intercept
                img_slice = image_3d[:, :, slice_i]  # (512, 512) ndarray
                img_slice = np.clip(img_slice, hu_clip_min, hu_clip_max)
                img_slice = ((img_slice - hu_clip_min) / (hu_clip_max - hu_clip_min)) * 255  # all values in [0, 255]
                img_slice = img_slice.astype(np.uint8)  # (512, 512) ndarray, all values are integers in [0, 255]

                msk_slice = mask_3d[:, :, slice_i]
                mask_tensor = (msk_slice == idx).astype(np.uint8)  # (512, 512) ndarray (uint8)

                img_slice_=img_slice.copy()
                mask_tensor_=mask_tensor.copy()
                img_slice_=intensityTransform([img_slice_])[0] #图像进行强度增强
                img_slice_,mask_tensor_=spatialTransform([img_slice_,mask_tensor_]) #图像和掩码同步进行空间增强

                img_slice_ = np.stack([img_slice_] * 3, axis=-1)  # (512,512,3) nparray (uint8)

                mask_tensor_ = torch.from_numpy(mask_tensor_.astype(np.float32))  # float32 为了后面模型能够计算BCEloss
                mask_tensor_ = mask_tensor_.unsqueeze(0)  # (1, H, W) tensor (为了应用interpolate)

                raw_data["images"].append(img_slice_)
                raw_data["labels"].append(mask_tensor_)
                raw_data["organs"].append(organ)

    # 注意一定要把包含所有数据集的列表一起放入processor里处理，不能分开处理
    inp = processor(text=raw_data["organs"], images=raw_data["images"], padding=True, truncation=True,
                    return_tensors="pt")

    # 还需要考虑的问题是：mask_tensor可能和处理后得到的pixel_values的shape不一致
    # processor将(N,3,512,512)的图像转成(N,3,352,352)的pixel_values
    # 而mask_tensor的shape是(N,1,512,512)

    labels = torch.stack(raw_data['labels'], dim=0)  # (N,H,W)
    labels = F.interpolate(labels, size=inp.pixel_values.shape[-2:], mode="nearest")
    labels = labels.squeeze(1)  # (N,H,W) tensor, all values being 0 or 1
    processed_data = {'input_ids': inp.input_ids, 'pixel_values': inp.pixel_values,
                      'attention_mask': inp.attention_mask, 'labels': labels}
    # Here inp.input_ids is a tensor of shape (N, embedding_dim), N is the number of images and embedding_dim is the size of the input embedding.
    # inp.pixel_values is a tensor of shape (N, 3, 352, 352),
    # inp.attention_mask is a tensor of shape (N, embedding_dim)
    # 'labels' is a tensor of shape (N, 352, 352)
    return processed_data


class ClipSegDataset(Dataset):
    def __init__(self, data_dict):
        self.input_ids = data_dict['input_ids']
        self.pixel_values = data_dict['pixel_values']
        self.attention_mask = data_dict['attention_mask']
        self.labels = data_dict['labels']

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return {"input_ids": self.input_ids[idx],
                "pixel_values": self.pixel_values[idx],
                "attention_mask": self.attention_mask[idx],
                "labels": self.labels[idx]}


def get_train_val_dataloader(
        pt_path="./CLIP/pku_dataset",
        image_dir="./pku_train_dataset/ct",
        mask_dir="./pku_train_dataset/label",
        json_dir="./pku_train_dataset/json",
        N_slices=30,
        N_neg_slices=5,
        seed=42,
        val_ratio=0.2,
        batch_size=4,
        slope=1.0,
        intercept=-1024.0,
        hu_clip_min=0.0,
        hu_clip_max=500.0
):
    # pt_path exists, then we don't need run build_examples every time
    if pt_path and os.path.exists(pt_path):
        examples = torch.load(pt_path)
    else:
        examples = build_examples(image_dir, mask_dir, json_dir,
                                  N_slices, N_neg_slices, seed,
                                  slope, intercept, hu_clip_min, hu_clip_max)
        if pt_path:
            torch.save(examples, pt_path)

    ds = ClipSegDataset(examples)

    n_total = len(ds)
    n_val = int(n_total * val_ratio)
    n_train = n_total - n_val
    train_ds, val_ds = random_split(ds, [n_train, n_val], generator=torch.Generator().manual_seed(seed))
    print(f"Train dataset size: {n_train}")
    print(f"Validation dataset size: {n_val}")

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader


#测试
if __name__ == "__main__":
    get_train_val_dataloader(
            pt_path="./CLIP/pku_dataset_trial",
            image_dir="./pku_train_dataset/ct",
            mask_dir="./pku_train_dataset/label",
            json_dir="./pku_train_dataset/json",
            N_slices=30,
            N_neg_slices=5,
            seed=42,
            val_ratio=0.2,
            batch_size=4,
            slope=1.0,
            intercept=-1024.0,
            hu_clip_min=0.0,
            hu_clip_max=500.0,
    )
