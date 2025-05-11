# Medical_Image_Segmentation_pku006

# ClipSeg

已经上传了新版数据处理和微调代码，主要调整：
1. 彻底分离了数据处理和训练，但是要求有较大的内存才能跑
2. 修正了各种细节问题，（见[build_dataset.py](./CLIP/build_dataset.py)注释）
3. 舍弃了Trainer，手动设置了训练程序，流程更加明晰
4. 用PEFT遇到了loss.requires_grad=False的问题，没法进行反向传播，所以把PEFT的代码注释掉了，暂时先微调所有参数
随便设了一些超参数，结果大概这样，dice score不是很高：
![image](https://github.com/user-attachments/assets/f3ef740c-4654-4e2c-9e6f-34bd81f85c3b)
--孟尧
修改了finetuning.py的代码，发现只要lora里因为冻结了编码器的原因导致无法实现反向传播，然后计划是现解放视觉编码器的q和v参数看一下dice score的效果
修改后的代码见sjx的文件夹
## 数据重采样
见信哥的文件夹[./zxl/preprocess/resample.py](./zxl/preprocess/resample.py)

# SAM
## ResNet训练
SAM模型所需要的ResNet网络及其训练放在了./SAM文件夹中，
使用方法为：
在./SAM下运行
```shell
python temp.py，
```
确保./SAM下有名为pku_train_dataset的数据集（这个数据集就是学校提供的原始数据集，不是CLIPSeg预处理后的数据集）.

ResNet网络容易训练，只需几个epoch，取测试集上loss最小（约0.0065）的模型参数储存在./SAM/best_model.pth中，就完成了ResNet的训练了. 
但是参数文件超过了25MB，没法上传，只好放微信群里了.

![image](https://github.com/user-attachments/assets/3ae0364f-18bb-40fa-abea-614d22181a86)
![image](https://github.com/user-attachments/assets/5739383f-ed6a-4137-9798-c91a9b3e1a8a)

## SAM分割演示
直接用SAM模型，不对SAM进行微调，也能取得很好效果. 
因此，直接做成演示文件[./SAM/SAMseg.py](./SAM/SAMseg.py). 
只需要保证./SAM中有best_model.pth和名为"pku_train_dataset"的原始数据集，就可以在./SAM中运行SAMseg.py，得到图像窗口。

![f914205c53612c63670f4b963950f37](https://github.com/user-attachments/assets/530ce3f4-457b-4d75-bc21-0185b9e7c7d4)

