import os
os.environ["WANDB_DIR"] = "./CLIP/wandb_logs"
from transformers import CLIPSegForImageSegmentation
from build_dataset import get_train_val_dataloader
import torch
from peft import LoraConfig, get_peft_model
import wandb
from tqdm import tqdm
import monai

'''
5.9改动：
训练阶段的dice_score也计算并记录
定义组合损失函数DiceBCELoss，但是训练结果并不理想
'''


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
checkpoint_dir = "./CLIP/cliseg-medical"
num_epochs = 20
lr = 1e-3
N_slices = 30
N_neg_slices = 5
batch_size = 4
hu_clip_min = 0.0
hu_clip_max = 500.0
val_ratio = 0.2
seed = 42


# train_loader, valid_loader = get_train_val_dataloader(
#     pt_path="./CLIP/pku_dataset",
#     image_dir="./pku_train_dataset/ct",
#     mask_dir="./pku_train_dataset/label",
#     json_dir="./pku_train_dataset/json",
#     N_slices=N_slices,
#     N_neg_slices=N_neg_slices,
#     seed=seed,
#     val_ratio=val_ratio,
#     batch_size=batch_size,
#     slope=1.0,
#     intercept=-1024.0,   #虽然输入了-1024，实际上没有用到这个
#     hu_clip_min=hu_clip_min,
#     hu_clip_max=hu_clip_max)
# print("dataloader loaded successfully")
model = CLIPSegForImageSegmentation.from_pretrained("CIDAS/clipseg-rd64-refined", cache_dir="./CLIP/hf_cache")
print("Model loaded successfully")
# lora_config = LoraConfig(
#     r=8, 
#     lora_alpha=32,
#     # target_modules=["visual_projection", "text_projection"],  # 修改CLIPSeg关键层
#     target_modules=['Linear'],
#     lora_dropout=0.1
# )
# model = get_peft_model(model, lora_config)
# print("LoRA configuration applied to the model:")
# model.print_trainable_parameters()  # 应显示可训练参数占比约1-5%
model.to(device)


# 直接把dice_metric放入compute_metrics函数中了
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    probs = logits.sigmoid()
    preds = (probs > 0.5).float()  # Convert probabilities to binary predictions
    intersection = (preds * labels).sum((1, 2))    # notice the dimension here, (B, H, W) tensor
    union = preds.sum((1, 2)) + labels.sum((1, 2))
    dice = (2 * intersection) / (union + 1e-6)
    return dice.mean().item()



optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=4, gamma=0.3)
wandb.init(
    project="CLIPSeg",
    name="aug",
    config={
        "learning_rate": lr,
        "num_epochs": num_epochs,
        # "lora_r": lora_config.r,
        # "lora_alpha": lora_config.lora_alpha,
        # "lora_dropout": lora_config.lora_dropout,
        "N_slices": N_slices,
        "N_neg_slices": N_neg_slices,
        "seed": seed,
        "val_ratio": val_ratio,
        "batch_size": batch_size,
        "hu_clip_min": hu_clip_min,
        "hu_clip_max": hu_clip_max,
        "optimizer": optimizer.__class__.__name__,
        "scheduler": scheduler.__class__.__name__,
        "step_size": scheduler.step_size,
        "gamma": scheduler.gamma
    }
)
print("WandB initialized successfully")

train_loader, valid_loader = get_train_val_dataloader(
    pt_path="./CLIP/pku_dataset_trial",
    image_dir="./pku_train_dataset/ct",
    mask_dir="./pku_train_dataset/label",
    json_dir="./pku_train_dataset/json",
    N_slices=N_slices,
    N_neg_slices=N_neg_slices,
    seed=seed,
    val_ratio=val_ratio,
    batch_size=batch_size,
    slope=1.0,
    intercept=-1024.0,   #虽然输入了-1024，实际上没有用到这个
    hu_clip_min=hu_clip_min,
    hu_clip_max=hu_clip_max)
print("dataloader loaded successfully")

def DiceBCELoss(output, target, alpha=1,beta=1):# BCEweighted + DiceLoss
    return alpha*torch.nn.BCEWithLogitsLoss()(output, target) + beta*monai.losses.DiceLoss()(output, target)

best_dice = 0.0

# for epoch in tqdm(range(num_epochs), desc="Epochs"):
for epoch in range(num_epochs):
    print(f"Start Training Epoch {epoch + 1}/{num_epochs}")
    model.train()
    running_loss = 0.0
    dice_score_train = 0.0
    for batch in tqdm(train_loader, desc="Training Batches"):
        optimizer.zero_grad()
        outputs = model(input_ids=batch["input_ids"].to(device),
                        pixel_values=batch["pixel_values"].to(device),
                        attention_mask=batch["attention_mask"].to(device),
                        labels=batch["labels"].to(device))
        # loss = DiceBCELoss(outputs.logits, batch["labels"].to(device),beta=0)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        dice_score_train += compute_metrics((outputs.logits, batch["labels"].to(device)))
        avg_dice_score_train = dice_score_train / len(train_loader)
        wandb.log({"train_loss": loss.item()})

    avg_train_loss = running_loss / len(train_loader)
    wandb.log({"avg_train_loss": avg_train_loss,"dice_score_train": avg_dice_score_train})
    print(f"Epoch {epoch + 1}/{num_epochs}, Average Training Loss: {avg_train_loss:.6f}, Dice Score Train: {avg_dice_score_train:.6f}")
    scheduler.step()

    model.eval()
    dice_score_val = 0.0
    val_loss = 0.0
    with torch.no_grad():
        for batch in tqdm(valid_loader, desc="Validation Batches"):
            outputs = model(input_ids=batch["input_ids"].to(device),
                            pixel_values=batch["pixel_values"].to(device),
                            attention_mask=batch["attention_mask"].to(device),
                            labels=batch["labels"].to(device))
            # val_loss += DiceBCELoss(outputs.logits, batch["labels"].to(device),beta=0).item()
            val_loss += outputs.loss.item()
            dice_score_val += compute_metrics((outputs.logits, batch["labels"].to(device)))
        avg_val_loss = val_loss / len(valid_loader)
        avg_dice_score_val = dice_score_val / len(valid_loader)
        wandb.log({"val_loss": avg_val_loss, "dice_score_val": avg_dice_score_val, "epoch": epoch + 1})
        print(f"Validation Loss: {avg_val_loss:.6f}, Dice Score Val: {avg_dice_score_val:.6f}")
        
        if avg_dice_score_val > best_dice:
            best_dice = avg_dice_score_val
            model.save_pretrained(checkpoint_dir)
wandb.finish()
