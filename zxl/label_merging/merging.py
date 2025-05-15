import nibabel as nib
import numpy as np
import os
from glob import glob

def merge_labels_per_patient(label_dir, output_dir, patient_id):
    """
    合并单个病人的多器官标签文件
    :param label_dir: 标签文件目录（包含类似"001_1.nii.gz"的文件）
    :param output_dir: 合并后文件的输出目录
    :param patient_id: 病人编号（如"001"）
    """
    # 按器官编号排序获取该病人的所有标签文件
    label_files = sorted(
        glob(os.path.join(label_dir, f"{patient_id}_*.nii.gz")),
        key=lambda x: int(os.path.basename(x).split("_")[1].split(".")[0])
    
    # 预期应有11个器官文件
    if len(label_files) != 11:
        missing = set(range(1,12)) - {int(os.path.basename(f).split("_")[1].split(".")[0]) for f in label_files}
        raise FileNotFoundError(f"病人 {patient_id} 缺少器官文件: {missing}")

    # 初始化参考图像和合并矩阵
    ref_img = nib.load(label_files[0])
    ref_affine = ref_img.affine
    ref_shape = ref_img.shape
    combined = np.zeros(ref_shape, dtype=np.int16)
    overlap_map = np.zeros(ref_shape, dtype=np.int16)  # 记录重叠次数

    # 遍历每个器官文件
    for file_path in label_files:
        organ_id = int(os.path.basename(file_path).split("_")[1].split(".")[0])
        organ_img = nib.load(file_path)
        
        # 验证空间一致性
        if organ_img.shape != ref_shape:
            raise ValueError(f"文件 {os.path.basename(file_path)} 维度不一致: 预期{ref_shape}, 实际{organ_img.shape}")
        if not np.allclose(organ_img.affine, ref_affine, atol=1e-3):
            raise ValueError(f"文件 {os.path.basename(file_path)} 仿射矩阵不一致")
        
        data = organ_img.get_fdata().astype(np.int16)
        
        # 标记器官区域
        combined = np.where(data > 0, organ_id, combined)
        overlap_map += (data > 0)  # 重叠计数+1

    # 处理冲突区域（重叠次数>1则标记为-1）
    combined[overlap_map > 1] = -1

    # 保存合并后的文件
    output_path = os.path.join(output_dir, f"{patient_id}_merged.nii.gz")
    new_img = nib.Nifti1Image(combined, affine=ref_affine, header=ref_img.header)
    new_img.set_data_dtype(np.int16)
    nib.save(new_img, output_path)
    print(f"病人 {patient_id} 合并完成 -> {output_path}")

if __name__ == "__main__":
    # 使用示例
    label_directory = "/path/to/labels"  # 包含类似 "001_1.nii.gz" 的目录
    output_directory = "/path/to/merged"
    os.makedirs(output_directory, exist_ok=True)

    # 处理所有病人
    all_patients = set(
        os.path.basename(f).split("_")[0]
        for f in glob(os.path.join(label_directory, "*.nii.gz"))
    )
    
    for patient_id in sorted(all_patients):
        try:
            merge_labels_per_patient(label_directory, output_directory, patient_id)
        except Exception as e:
            print(f"处理病人 {patient_id} 失败: {str(e)}")
