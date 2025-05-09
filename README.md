# Medical_Image_Segmentation_pku006

前面传的数据处理和模型训练代码有些bug，需要一些时间来一点点改。  --孟尧

已经上传了新版数据处理和微调代码，主要调整：
1. 彻底分离了数据处理和训练，但是要求有较大的内存才能跑
2. 修正了各种细节问题，（见[build_dataset.py](./CLIP/build_dataset.py)注释）
3. 舍弃了Trainer，手动设置了训练程序，流程更加明晰
4. 用PEFT遇到了loss.requires_grad=False的问题，没法进行反向传播，所以把PEFT的代码注释掉了，暂时先微调所有参数
随便设了一些超参数，结果大概这样，dice score不是很高：
![image](https://github.com/user-attachments/assets/f3ef740c-4654-4e2c-9e6f-34bd81f85c3b)
--孟尧

SAM模型所需要的ResNet网络及其训练放在了./SAM文件夹中，
使用方法为：
在./下运行
```shell
python SAM/temp.py，
```
确保./下有名为pku_train_dataset的数据集（这个数据集就是学校提供的原始数据集，不是CLIPSeg预处理后的数据集）.
