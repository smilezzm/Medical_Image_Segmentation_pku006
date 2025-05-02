# Medical_Image_Segmentation_pku006

前面传的数据处理和模型训练代码有些bug，需要一些时间来一点点改。  --孟尧

已经上传了新版数据处理和微调代码，主要调整：
1. 彻底分离了数据处理和训练，但是要求有较大的内存才能跑
2. 修正了各种细节问题，（见[build_dataset.py](./CLIP/build_dataset.py)注释）
3. 舍弃了Trainer，手动设置了训练程序，流程更加明晰
