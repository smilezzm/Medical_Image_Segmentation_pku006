import os
os.environ["WANDB_DIR"] = "./CLIP/wandb_logs"
from transformers import CLIPSegForImageSegmentation
from CLIP.build_dataset import get_train_val_dataloader
import torch
from peft import LoraConfig, get_peft_model
import wandb
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
checkpoint_dir = "./CLIP/cliseg-medical"
num_epochs = 20
lr = 3e-4
N_slices = 30
N_neg_slices = 5
batch_size = 4
hu_clip_min = 0.0
hu_clip_max = 500.0
val_ratio = 0.2
seed = 42


train_loader, valid_loader = get_train_val_dataloader(
    pt_path="./CLIP/pku_dataset", 
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
model = CLIPSegForImageSegmentation.from_pretrained("CIDAS/clipseg-rd64-refined", cache_dir="./CLIP/hf_cache")
print("Model loaded successfully")
#print(model)
lora_config = LoraConfig(
     r=8, 
     lora_alpha=16,
     target_modules=[
                    "vision_model.encoder.layers.0.self_attn.q_proj",
                    "vision_model.encoder.layers.0.self_attn.v_proj",
                    "vision_model.encoder.layers.1.self_attn.q_proj",
                    "vision_model.encoder.layers.1.self_attn.v_proj",
                    "vision_model.encoder.layers.2.self_attn.q_proj",
                    "vision_model.encoder.layers.2.self_attn.v_proj",
                    "vision_model.encoder.layers.3.self_attn.q_proj",
                    "vision_model.encoder.layers.3.self_attn.v_proj",
                    "vision_model.encoder.layers.4.self_attn.q_proj",
                    "vision_model.encoder.layers.4.self_attn.v_proj",   
                    "vision_model.encoder.layers.5.self_attn.q_proj",
                    "vision_model.encoder.layers.5.self_attn.v_proj",
                    "vision_model.encoder.layers.6.self_attn.q_proj",
                    "vision_model.encoder.layers.6.self_attn.v_proj",
                    "vision_model.encoder.layers.7.self_attn.q_proj",
                    "vision_model.encoder.layers.7.self_attn.v_proj",
                    "vision_model.encoder.layers.8.self_attn.q_proj",
                    "vision_model.encoder.layers.8.self_attn.v_proj",
                    "vision_model.encoder.layers.9.self_attn.q_proj",
                    "vision_model.encoder.layers.9.self_attn.v_proj",
                    "vision_model.encoder.layers.10.self_attn.q_proj",
                    "vision_model.encoder.layers.10.self_attn.v_proj",
                    "vision_model.encoder.layers.11.self_attn.q_proj",
                    "vision_model.encoder.layers.11.self_attn.v_proj",
                     ],#视觉编码器
     lora_dropout=0.2,
     modules_to_save=["decoder"]#保证解码器可修改
 )
model = get_peft_model(model, lora_config)
print("LoRA configuration applied to the model:")
model.print_trainable_parameters()  # 应显示可训练参数占比约1-5%
#for name, param in model.named_parameters():
#    print(f"Parameter {name}, requires_grad: {param.requires_grad}")
#print(model)
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



optimizer = torch.optim.AdamW(
    filter(lambda p:p.requires_grad,model.parameters()), lr=lr,weight_decay=0.01,eps=1e-8)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=4, gamma=0.3)
wandb.init(
    project="CLIPSeg",
    config={
        "learning_rate": lr,
        "num_epochs": num_epochs,
        "lora_r": lora_config.r,
        "lora_alpha": lora_config.lora_alpha,
        "lora_dropout": lora_config.lora_dropout,
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


best_dice = 0.0

for epoch in tqdm(range(num_epochs), desc="Epochs"):
    model.train()
    running_loss = 0.0
    for batch in tqdm(train_loader, desc="Training Batches"):
        optimizer.zero_grad()
        outputs = model(input_ids=batch["input_ids"].to(device),
                        pixel_values=batch["pixel_values"].to(device),
                        attention_mask=batch["attention_mask"].to(device),
                        labels=batch["labels"].to(device))
        loss = outputs.loss
        
        #print(f"Loss:{loss},requires_grad:{loss.requires_grad},grad_fn:{loss.grad_fn}")
        # 检查损失值是否需要梯度
        #for name,param in model.named_parameters():
        #    print(f"Parameter {name}, requires_grad: {param.requires_grad}")
        # 检查损失值依赖的参数
        #for name, param in model.named_parameters():
        #    if "lora" in name and param.requires_grad:
        #        print(f"LoRA Parameter {name}, Grad: {param.grad}")
        #    else:
        #        print(f"Parameter {name}, Grad: {param.grad}")


        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        wandb.log({"train_loss": loss.item()})

    avg_train_loss = running_loss / len(train_loader)
    wandb.log({"avg_train_loss": avg_train_loss})
    print(f"Epoch {epoch + 1}/{num_epochs}, Average Training Loss: {avg_train_loss:.4f}")
    scheduler.step()

    model.eval()
    dice_score = 0.0
    val_loss = 0.0
    with torch.no_grad():
        for batch in tqdm(valid_loader, desc="Validation Batches"):
            # 打印 labels 的形状和设备
            #print(f"Validation Labels shape: {batch['labels'].shape}, Device: {batch['labels'].device}")
            outputs = model(input_ids=batch["input_ids"].to(device),
                            pixel_values=batch["pixel_values"].to(device),
                            attention_mask=batch["attention_mask"].to(device),
                            labels=batch["labels"].to(device))
            val_loss += outputs.loss.item()
            dice_score += compute_metrics((outputs.logits, batch["labels"].to(device)))
        avg_val_loss = val_loss / len(valid_loader)
        avg_dice_score = dice_score / len(valid_loader)
        wandb.log({"val_loss": avg_val_loss, "dice_score": avg_dice_score, "epoch": epoch + 1})
        print(f"Validation Loss: {avg_val_loss:.4f}, Dice Score: {avg_dice_score:.4f}")
        
        if avg_dice_score > best_dice:
            best_dice = avg_dice_score
            model.save_pretrained(checkpoint_dir)
wandb.finish()