import numpy as np
from transformers import CLIPSegProcessor, CLIPSegForImageSegmentation
from build_dataset import MedicalDataset
import torch
from torch.utils.data import random_split, DataLoader
from peft import LoraConfig, get_peft_model
from transformers import TrainingArguments, Trainer
import evaluate
from monai.metrics import DiceMetric

"""
有个bug：
process_input无法接收到正确的batch，打印出来是空的，然而train_dataset本身没问题，包含‘image’、‘prompt‘、‘mask’键值对，且可以被dataloader正确处理。
诡异的是如果在training_args中设置label_names=["mask"]，那么process_input中batch打印出来只含‘mask’键值对，‘image’和‘prompt‘键值对丢失。
可能是Trainer内部调用的问题，需要进一步研究
"""

dataset = MedicalDataset(prompt_csv="prompt.csv")  # prompt.csv is obtained by build_prompt_file.py
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
SEED = 42
np.random.seed(SEED)# 固定随机种子以确保训练集验证集划分一致性
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
print("Dataset splitting done")
print("Test dataset size:", len(test_dataset))
print("Training dataset size:", len(train_dataset))


def process_input(batch):
    """
    Collects image and text prompts from a batch of samples. Also return the ground truth mask as "labels".
    This function is used in Trainer.

    batch: A batch of samples from the dataset.
    return: A tuple containing a list of images and a list of text prompts.
    """
    print(batch)
    images = [sample["image"] for sample in batch]  # Convert numpy arrays to PIL images
    texts = [sample["prompt"] for sample in batch]  # Collect text prompts
    inputs = processor(
        text=texts,  # 文本列表
        images=images,  # 一个batch内的多张图像
        padding=True,  # 自动填充文本到相同长度
        truncation=True,  # 截断超长文本
        return_tensors="pt")  # 返回 PyTorch 张量"]
    inputs["labels"] = torch.tensor([sample["mask"] for sample in batch])  # Collect ground truth masks as labels
    return inputs


processor = CLIPSegProcessor.from_pretrained("CIDAS/clipseg-rd64-refined")#需要挂梯子，如果服务器不方便挂可能需要提前下载预训练的模型
model = CLIPSegForImageSegmentation.from_pretrained("CIDAS/clipseg-rd64-refined")
print("Model and processor loaded successfully")
lora_config = LoraConfig(
    r=8,
    lora_alpha=32,
    target_modules=["visual_projection", "text_projection"],  # 修改CLIPSeg关键层
    lora_dropout=0.1
)
model = get_peft_model(model, lora_config)
print("LoRA configuration applied to the model:")
model.print_trainable_parameters()  # 应显示可训练参数占比约1-5%

#从monai库导入DiceMetric，用于评估分割模型的性能。
dice_metric = DiceMetric(include_background=False, reduction="mean")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = torch.sigmoid(torch.tensor(logits))
    dice_metric(preds, torch.tensor(labels))
    return {"dice_score": dice_metric.aggregate().item()}


training_args = TrainingArguments(
    output_dir="clipseg-medical",
    per_device_train_batch_size=4,
    learning_rate=3e-5,
    num_train_epochs=20,
    #evaluation_strategy="steps",  #比赛官方说明文档中有，但是实际无这个参数
    eval_steps=200,  # 500,
    save_steps=1000,  # 1000,
    fp16=True,  # 启用混合精度训练
    # report_to="wandb",  # 使用Weights & Biases进行实验跟踪
    report_to="none", #暂时禁用wandb，需要wandb账号
    disable_tqdm=False,  # I added a process bar here

    label_names=["mask"],  # 根据实际标签字段名调整
)

trainer = Trainer(  # 训练器，用于训练模型
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    compute_metrics=compute_metrics,
    data_collator=process_input
)


# loader = DataLoader(train_dataset, batch_size=4)
# for batch in loader:
#     print(batch)
#     break
# print(train_dataset[0].keys())


trainer.train()
