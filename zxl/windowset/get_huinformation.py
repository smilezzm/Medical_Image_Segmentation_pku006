import os
import json
import nibabel as nib
import numpy as np
from glob import glob
from collections import defaultdict

def load_label_mapping(json_path):
    """从JSON文件加载并反转标签映射 (ID -> 器官名)"""
    with open(json_path) as f:
        data = json.load(f)
    original_map = data.get("label_mapping", {})
    return {v: k for k, v in original_map.items()}

def process_case(ct_path, label_path, label_map, output_dir):
    """处理单个病例"""
    # 加载数据
    ct_img = nib.load(ct_path)
    label_img = nib.load(label_path)
    ct_data = ct_img.get_fdata().astype(np.float32)
    label_data = label_img.get_fdata().astype(np.uint8)
    
    # 提取器官统计
    organ_stats = {}
    present_organs = np.unique(label_data)
    
    for organ_id in present_organs:
        if organ_id == 0:  # 跳过背景
            continue
        
        mask = (label_data == organ_id)
        hu_values = ct_data[mask]
        
        if hu_values.size == 0:
            continue
            
        stats = {
            "min": np.min(hu_values),
            "max": np.max(hu_values),
            "mean": np.mean(hu_values),
            "std": np.std(hu_values),
            "p10": np.percentile(hu_values, 10),  # 新增10%分位
            "p90": np.percentile(hu_values, 90)   # 新增90%分位
        }
        organ_stats[organ_id] = stats
    
    # 生成病例级报告
    case_id = os.path.basename(ct_path).split(".")[0]
    output_path = os.path.join(output_dir, f"{case_id}_hu_stats.txt")
    
    with open(output_path, "w") as f:
        f.write(f"Case ID: {case_id}\n")
        # 扩展表头
        f.write("Organ ID | Name      | Min  | P10   | Mean  | P90   | Max  | Std\n")
        f.write("-"*75 + "\n")
        
        for org_id, stats in organ_stats.items():
            org_name = label_map.get(org_id, "Unknown")
            f.write(
                f"{org_id:8} | {org_name:9} | "
                f"{stats['min']:4.1f} | {stats['p10']:5.1f} | "
                f"{stats['mean']:5.1f} | {stats['p90']:5.1f} | "
                f"{stats['max']:4.1f} | {stats['std']:4.1f}\n"
            )
    
    return organ_stats

def generate_global_stats(all_stats, label_map, output_path):
    """生成全局统计报告"""
    global_data = defaultdict(list)
    
    # 聚合所有病例数据
    for case_stats in all_stats:
        for org_id, stats in case_stats.items():
            global_data[org_id].append({
                "min": stats["min"],
                "max": stats["max"],
                "mean": stats["mean"],
                "std": stats["std"],
                "p10": stats["p10"],  # 收集分位数
                "p90": stats["p90"]
            })
    
    # 计算全局统计
    with open(output_path, "w") as f:
        f.write("Global HU Statistics\n")
        # 扩展表头
        f.write("Organ ID | Name      | Global Min | Global P10 | Global Mean | Global P90 | Global Max | Global Std\n")
        f.write("-"*95 + "\n")
        
        for org_id, all_values in global_data.items():
            org_name = label_map.get(org_id, "Unknown")
            
            # 计算极值
            global_min = min([v["min"] for v in all_values])
            global_max = max([v["max"] for v in all_values])
            
            # 计算分位数均值
            global_p10 = np.mean([v["p10"] for v in all_values])
            global_p90 = np.mean([v["p90"] for v in all_values])
            
            # 计算均值和标准差
            all_means = [v["mean"] for v in all_values]
            global_mean = np.mean(all_means)
            global_std = np.std(all_means)
            
            f.write(
                f"{org_id:8} | {org_name:9} | "
                f"{global_min:10.1f} | {global_p10:10.1f} | "
                f"{global_mean:10.1f} | {global_p90:10.1f} | "
                f"{global_max:10.1f} | {global_std:8.1f}\n"
            )

def main(data_root, output_root):
    """主处理流程"""
    os.makedirs(output_root, exist_ok=True)
    
    # 输入路径
    ct_dir = os.path.join(data_root, "ct")
    label_dir = os.path.join(data_root, "label")
    
    # 获取匹配病例
    ct_files = sorted(glob(os.path.join(ct_dir, "*.nii.gz")))
    label_files = sorted(glob(os.path.join(label_dir, "*.nii.gz")))
    
    ct_basenames = {os.path.basename(f).split(".")[0] for f in ct_files}
    label_basenames = {os.path.basename(f).split(".")[0] for f in label_files}
    common_cases = ct_basenames & label_basenames
    
    # 加载标签映射
    label_map = load_label_mapping("pre/label_mapping.json")
    
    all_stats = []
    for case_id in common_cases:
        ct_path = os.path.join(ct_dir, f"{case_id}.nii.gz")
        label_path = os.path.join(label_dir, f"{case_id}.nii.gz")
        
        try:
            case_stats = process_case(ct_path, label_path, label_map, output_root)
            all_stats.append(case_stats)
        except Exception as e:
            print(f"Error processing {case_id}: {str(e)}")
    
    # 输出全局报告
    global_output = os.path.join(output_root, "global_hu_stats.txt")
    generate_global_stats(all_stats, label_map, global_output)

if __name__ == "__main__":
    DATA_ROOT = "CLAPATPS"
    OUTPUT_ROOT = "pre/huinformations"
    main(DATA_ROOT, OUTPUT_ROOT)
