import os
import torch
import numpy as np
import SimpleITK as sitk


def compute_multiclass_dice(gt_path, pred_path, num_classes, device="cuda"):
    """计算多分类 Dice（支持 CUDA）"""
    # 读取数据并转为 PyTorch Tensor
    gt = torch.from_numpy(sitk.GetArrayFromImage(sitk.ReadImage(gt_path))).to(device)
    pred = torch.from_numpy(sitk.GetArrayFromImage(sitk.ReadImage(pred_path))).to(device)

    assert gt.shape == pred.shape, f"Shape mismatch: GT {gt.shape} vs Pred {pred.shape}"

    dice_scores = []
    for class_label in range(1, num_classes + 1):  # 假设类别是 1, 2, ..., num_classes
        gt_mask = (gt == class_label)
        pred_mask = (pred == class_label)

        intersection = (gt_mask & pred_mask).sum().float()
        union = gt_mask.sum() + pred_mask.sum()

        dice = (2.0 * intersection) / (union + 1e-7)  # 避免除以0
        dice_scores.append(dice.item())  # 转回 CPU float

    return dice_scores


def compute_dice_across_folders(gt_dir, pred_dir, num_classes, device="cuda"):
    """计算整个文件夹的多分类 Dice"""
    dice_results = {}  # {filename: [dice_class1, dice_class2, ...]}

    gt_files = sorted([f for f in os.listdir(gt_dir) if f.endswith(".nii.gz")])
    pred_files = sorted([f for f in os.listdir(pred_dir) if f.endswith(".nii.gz")])

    assert gt_files == pred_files, "GT and Pred files do not match!"

    for filename in gt_files:
        gt_path = os.path.join(gt_dir, filename)
        pred_path = os.path.join(pred_dir, filename)

        dice_scores = compute_multiclass_dice(gt_path, pred_path, num_classes, device)
        dice_results[filename] = dice_scores

    # 计算每个类别的平均 Dice
    avg_dice_per_class = np.mean([dices for dices in dice_results.values()], axis=0)

    print("\n=== Dice Scores per Case ===")
    for filename, dices in dice_results.items():
        print(f"{filename}: {[f'{d:.4f}' for d in dices]}")

    print("\n=== Average Dice per Class ===")
    for class_idx, avg_dice in enumerate(avg_dice_per_class, start=1):
        print(f"Class {class_idx}: {avg_dice:.4f}")

    return dice_results, avg_dice_per_class


# 示例调用
if __name__ == "__main__":
    gt_dir = ".\\pku_train_dataset\\label"  # Ground Truth 文件夹
    pred_dir = ".\\pku_train_dataset\\SAM3D_output"  # Prediction 文件夹
    num_classes = 11  # 类别数（如 3 类：1, 2, 3）
    device = "cuda" if torch.cuda.is_available() else "cpu"  # 自动检测 GPU

    print(f"Using device: {device}")
    dice_results, avg_dice_per_class = compute_dice_across_folders(gt_dir, pred_dir, num_classes, device)