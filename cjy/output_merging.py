import nibabel as nib
import numpy as np
import os
from glob import glob

'''
对于001号病人，有001_1.nii.gz、001_2.nii.gz、001_3.nii.gz……等多个器官标签文件，
按照器官优先级，优先级高的覆盖级别高。
合并单个病人的多器官标签文件，并按照器官优先级进行覆盖。
输出：001.nii.gz
'''

def merge_labels_per_patient(label_dir, output_dir, patient_id):
    """
    合并单个病人的多器官标签文件（带优先级覆盖）
    :param label_dir: 标签文件目录（包含类似"001_1.nii.gz"的文件）
    :param output_dir: 合并后文件的输出目录
    :param patient_id: 病人编号（如"001"）
    """
    # 定义器官优先级（值越大优先级越高）
    priority = {
        10: 10,  # SpinalCord
        7: 9,  # Liver
        5: 8,  # Kidney_L
        6: 8,  # Kidney_R
        1: 7,  # Bladder
        11: 6,  # Stomach
        2: 5,  # Colon
        9: 4,  # SmallIntestine
        8: 3,  # Rectum
        3: 2,  # Femur_Head_L
        4: 1  # Femur_Head_R
    }

    # 按优先级从低到高排序文件（高优先级最后处理）
    label_files = sorted(
        glob(os.path.join(label_dir, f"{patient_id}_*.nii.gz")),
        key=lambda x: priority[int(os.path.basename(x).split("_")[1].split(".")[0])]
    )

    # 初始化参考图像和合并矩阵
    ref_img = nib.load(label_files[0])
    ref_affine = ref_img.affine
    ref_shape = ref_img.shape
    combined = np.zeros(ref_shape, dtype=np.int16)

    # 遍历每个器官文件（从低优先级到高优先级）
    for file_path in label_files:
        organ_id = int(os.path.basename(file_path).split("_")[1].split(".")[0])
        organ_img = nib.load(file_path)

        # 验证空间一致性
        if organ_img.shape != ref_shape:
            raise ValueError(f"文件 {os.path.basename(file_path)} 维度不一致: 预期{ref_shape}, 实际{organ_img.shape}")
        if not np.allclose(organ_img.affine, ref_affine, atol=1e-3):
            raise ValueError(f"文件 {os.path.basename(file_path)} 仿射矩阵不一致")

        data = organ_img.get_fdata().astype(np.int16)

        # 优先级覆盖：高优先级器官覆盖低优先级
        combined = np.where(data > 0, organ_id, combined)

    # 保存合并后的文件
    output_path = os.path.join(output_dir, f"{patient_id}.nii.gz")
    new_img = nib.Nifti1Image(combined, affine=ref_affine, header=ref_img.header)
    new_img.set_data_dtype(np.int16)
    nib.save(new_img, output_path)
    print(f"病人 {patient_id} 合并完成 -> {output_path}")

if __name__ == "__main__":
    # 使用示例
    label_directory = "./test_output/label"  # 包含类似 "001_1.nii.gz" 的目录
    output_directory = "./test_output/output"
    os.makedirs(output_directory, exist_ok=True)

    # 处理所有病人
    all_patients = set(
        os.path.basename(f).split("_")[0]
        for f in glob(os.path.join(label_directory, "*.nii.gz"))
    )
    print(all_patients)
    for patient_id in sorted(all_patients):
        try:
            merge_labels_per_patient(label_directory, output_directory, patient_id)
        except Exception as e:
            print(f"处理病人 {patient_id} 失败: {str(e)}")
