# Medical_Image_Segmentation_pku006


尝试用CLIPSeg model，数据处理部分见[CLIP/build_prompt_file.py](./CLIP/build_prompt_file.py)和[/CLIP/build_dataset.py](./CLIP/build_dataset.py). 未经验证，可能有bug.

[SegVol](https://github.com/BAAI-DCAI/SegVol)中有一个基于voxel强度差异进行3D segmentation的脚本，被拿出并放在了[Fseg.py](./Fseg.py). [check_Fseg.py](./check_Fseg.py)想看看这个分割如何，但是跑不出来（不报错也没结果，一直在跑，跑了一夜了...）