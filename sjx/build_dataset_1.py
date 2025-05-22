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


image_dir = "home/sjx/clip/pku_train_dataset/ct"
mask_dir = "home/sjx/clip/pku_train_dataset/label"
json_dir = "home/sjx/clip/pku_train_dataset/json"
N_slices = 30
N_neg_slices = 5
seed = 42


def build_examples(image_dir, mask_dir, json_dir, N_slices, N_neg_slices, seed,
    slope, intercept, hu_clip_min, hu_clip_max, batch_size=16):
    np.random.seed(seed)
    processor = CLIPSegProcessor.from_pretrained("/home/sjx/clip/premodel", cache_dir="/home/sjx/clip/CLIP/hf_cache")#使用本地模型
    
    image_paths = sorted(glob(f"{image_dir}/*.nii.gz"))
    mask_paths  = sorted(glob(f"{mask_dir}/*.nii.gz"))

    assert len(image_paths) == len(mask_paths)
    processed_data = {'input_ids': [], 'pixel_values': [], 'attention_mask': [], 'labels': []}
    
    for img_path, msk_path in tqdm(zip(image_paths, mask_paths), total=len(image_paths), desc="cases"):
        mask_3d = nib.load(msk_path).get_fdata().astype(int)
        image_3d = nib.load(img_path).get_fdata()
        with open(os.path.join(json_dir, os.path.basename(img_path).replace(".nii.gz", ".json"))) as f:
            info = json.load(f)

        raw_data = {'organs': [], 'images': [], 'labels': []}
        for organ, idx in info["label_mapping"].items():
            pos = np.where(np.any(mask_3d == idx, axis=(0, 1)))[0]
            if len(pos) == 0: continue

            sel = np.random.choice(pos, size=min(N_slices, len(pos)), replace=False)
            neg = np.setdiff1d(np.arange(mask_3d.shape[-1]), pos)
            sel_neg = np.random.choice(neg, size=min(N_neg_slices, len(neg)), replace=False)
            
            for slice_i in np.concatenate((sel, sel_neg)):
                img_slice = image_3d[:, :, slice_i] 
                img_slice = np.clip(img_slice, hu_clip_min[organ], hu_clip_max[organ])
                img_slice = ((img_slice - hu_clip_min[organ]) / (hu_clip_max[organ] - hu_clip_min[organ])) * 255
                img_slice = img_slice.astype(np.uint8)
                img_slice = np.stack([img_slice] * 3, axis=-1)

                msk_slice = mask_3d[:,:,slice_i]
                mask_tensor  = (msk_slice==idx).astype(np.uint8)    # (512, 512) ndarray (uint8)
                mask_tensor = torch.from_numpy(mask_tensor.astype(np.float32))   # float32 为了后面模型能够计算BCEloss
                mask_tensor = mask_tensor.unsqueeze(0)  # (1, H, W) tensor (为了应用interpolate)

                raw_data["images"].append(img_slice)
                raw_data["labels"].append(mask_tensor)
                raw_data["organs"].append(organ)
                #Process in batches to reduce memory usage
                if len(raw_data["images"]) >= batch_size:
                    inp = processor(text=raw_data["organs"], images=raw_data["images"], padding=True, truncation=True, return_tensors="pt")
                    labels = torch.stack(raw_data['labels'], dim=0)
                    labels = F.interpolate(labels, size=inp.pixel_values.shape[-2:], mode="nearest")
                    labels = labels.squeeze(1)

                    processed_data['input_ids'].append(inp.input_ids)
                    processed_data['pixel_values'].append(inp.pixel_values)
                    processed_data['attention_mask'].append(inp.attention_mask)
                    processed_data['labels'].append(labels)

                    raw_data = {'organs': [], 'images': [], 'labels': []}

    # Process remaining data
        if raw_data["images"]:
            inp = processor(text=raw_data["organs"], images=raw_data["images"], padding=True, truncation=True, return_tensors="pt")
            labels = torch.stack(raw_data['labels'], dim=0)
            labels = F.interpolate(labels, size=inp.pixel_values.shape[-2:], mode="nearest")
            labels = labels.squeeze(1)

            processed_data['input_ids'].append(inp.input_ids)
            processed_data['pixel_values'].append(inp.pixel_values)
            processed_data['attention_mask'].append(inp.attention_mask)
            processed_data['labels'].append(labels)

    # 先 pad 到最大长度，再 cat
    def pad_and_cat(tensor_list, pad_value=0):
        # 找到最大长度
        max_len = max(t.shape[1] for t in tensor_list)
        padded = []
        for t in tensor_list:
            if t.shape[1] < max_len:
                pad_width = (0, max_len - t.shape[1])
                t = torch.nn.functional.pad(t, pad_width, value=pad_value)
            padded.append(t)
        return torch.cat(padded, dim=0)

    processed_data['input_ids'] = pad_and_cat(processed_data['input_ids'])
    processed_data['pixel_values'] = torch.cat(processed_data['pixel_values'], dim=0)
    processed_data['attention_mask'] = pad_and_cat(processed_data['attention_mask'])
    processed_data['labels'] = torch.cat(processed_data['labels'], dim=0)

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
    pt_path="home/sjx/clip/CLIP/pku_dataset_1", 
    image_dir="home/sjx/clip/pku_train_dataset/ct",
    mask_dir="home/sjx/clip/pku_train_dataset/label",
    json_dir="home/sjx/clip/pku_train_dataset/json",
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

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,pin_memory=True,num_workers=4)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,pin_memory=True,num_workers=4)

    return train_loader, val_loader
