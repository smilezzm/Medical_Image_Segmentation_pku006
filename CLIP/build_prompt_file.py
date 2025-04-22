import os
import json
import pandas as pd
from glob import glob
import nibabel as nib
import numpy as np

'''
This script process the data located at ./pku_train_dataset, which is provided by the school.

最终生成了名为prompt.csv的文件，其中每一项包含以下字段：
- image_path: CT图像的路径
- mask_path: 分割掩码（label）的路径
- slice_index: 该图像的切片索引（3D ct的第几层）
- organ: 器官名称
- label: 器官对应的标签（整数）

把每一个ct图像根据slice、organ分成了若干条数据，这些数据对应相同的image_path
其中，N_slices表示每个器官的正样本切片数量（随机的N_slices个包含某个器官掩码的切片），N_neg_slices表示负样本切片数量
'''


image_dir = "./pku_train_dataset/ct"
mask_dir = "./pku_train_dataset/label"
json_dir = "./pku_train_dataset/json"
N_slices = 30
N_neg_slices = 5

SEED = 42
np.random.seed(SEED)

image_paths = sorted(glob(f"{image_dir}/*.nii.gz"))
mask_paths = sorted(glob(f"{mask_dir}/*.nii.gz"))
csv_data = []

for img_path, msk_path in zip(image_paths, mask_paths):
    mask = nib.load(msk_path).get_fdata().astype(int)  # 加载3D分割掩码
    info = json.load(open(os.path.join(json_dir, os.path.basename(img_path).replace(".nii.gz", ".json")), 'r'))
    
    label_mapping = info["label_mapping"]
    organ_labels = list(label_mapping.keys())
    organ_indices = list(label_mapping.values())

    for organ, idx in label_mapping.items():
        pos = np.where(np.any(mask==idx, axis=(0,1)))[0] #  gives a 1D array containing the slice index which has the organ label
        if len(pos) == 0:
            continue
        sel = np.random.choice(pos, size=min(N_slices,len(pos)), replace=False)
        
        neg = np.setdiff1d(np.arange(mask.shape[-1]),pos)   # gives a 1D array containing the slice index which does not have the organ label
        sel_neg = np.random.choice(neg, size=min(N_neg_slices,len(neg)), replace=False)
        
        all_sel = np.concatenate((sel, sel_neg))
        np.random.shuffle(all_sel)

        for slice_i in all_sel:
            csv_data.append({
                "image_path": img_path,
                "mask_path": msk_path,
                "slice_index": int(slice_i),
                "organ": organ,
                "label": int(idx)
            })

df = pd.DataFrame(csv_data)
df.to_csv("prompt.csv", index=False)
print(f"Saved {len(df)} entries to prompt.csv")