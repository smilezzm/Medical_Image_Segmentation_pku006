import torch
from transformers import CLIPSegForImageSegmentation, CLIPSegProcessor
from PIL import Image
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
import os
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CLIPSegForImageSegmentation.from_pretrained("./CLIP/cliseg-medical").to(device)
processor = CLIPSegProcessor.from_pretrained("./CIDAS_clipseg-rd64-refined")
test_image_dir = "./test_dataset/ct"
output_label_dir = "./test_dataset/label"
os.makedirs(output_label_dir, exist_ok=True)
organ_name = "colon"  #器官名称
prompt="colon"#提示词    
test_images=os.listdir(test_image_dir)
organ_dir={"Bladder":1,"Colon":2,"Femur_Head_L":3,"Femur_Head_R":4,"Kidney_L":5,"Kidney_R":6,"Liver":7,"Rectum":8,"SmallIntestine":9,"SpinalCord":10,"Stomach":11}
# 新建器官名称的子文件夹
organ_label_dir = os.path.join(output_label_dir, organ_name)
os.makedirs(organ_label_dir, exist_ok=True)
for img_name in test_images:
    img_path = os.path.join(test_image_dir, img_name)
    # 读取nii.gz文件
    nii_img = nib.load(img_path)
    img_data = nii_img.get_fdata()
    pred_volume = []
    for i in range(img_data.shape[2]):
        image_slice = img_data[:, :, i]
        #归一化并转为3通道
        image_slice=(255*(image_slice-np.min(image_slice))/(np.max(image_slice)-np.min(image_slice))).astype(np.uint8)
        image_rgb=np.stack((image_slice,)*3, axis=-1)
        #推理
        inputs=processor(text=[prompt], images=image_rgb, return_tensors="pt", padding=True).to(device)
        with torch.no_grad():
            outputs = model(**inputs)
            pred_mask = outputs.logits.sigmoid().squeeze().cpu().numpy()
        organ_id=organ_dir[organ_name]
        #二值化
        pred_mask_bin = (pred_mask > 0.5).astype(np.uint8)*organ_id
        # 如果模型输出尺寸与原始slice尺寸不同，需要resize
        if pred_mask_bin.shape != image_slice.shape:
            import torch.nn.functional as F
            pred_mask_bin = torch.from_numpy(pred_mask_bin).unsqueeze(0).unsqueeze(0).float()
            pred_mask_bin = F.interpolate(pred_mask_bin, size=image_slice.shape, mode="nearest").squeeze().numpy().astype(np.uint8)
        pred_volume.append(pred_mask_bin)
    pred_volume = np.stack(pred_volume, axis=-1)  # (H, W, D)
    # 保存为nii.gz
    pred_nii = nib.Nifti1Image(pred_volume, affine=nii_img.affine)
    save_path = os.path.join(organ_label_dir, img_name.replace('.nii.gz', '_pred.nii.gz'))
    nib.save(pred_nii, save_path)
    print(f"Saved: {save_path}")
    