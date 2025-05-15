import os
import time

import torch
import torch.nn
from torch.nn import BCELoss
from torch.optim import Adam
from torch import Tensor
from torchvision.models import ResNet
from torchvision.models.resnet import Bottleneck, conv1x1
import nibabel as nib
import numpy as np
from torch.utils.data import Dataset, DataLoader
from glob import glob
from pathlib import Path
import wandb



organ = 1

class ClsNet(ResNet):
    '''
    input: tensor(B, Channels, H, W), (clipped) HU values
    output: tensor(B, 1, H/32, W/32), values between 0 and 1
    '''
    def __init__(self):
        super().__init__(Bottleneck, [3, 4, 6, 3])
        self.linear=conv1x1(2048,1)
        self.sigmoid=torch.nn.Sigmoid()

    def forward(self, x: Tensor) -> Tensor:
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x=self.linear(x)
        x=self.sigmoid(x)
        return x


train_device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
train_epoch=30#训练轮次
train_batch_size=16
ct_dir = "./pku_train_dataset/ct"
mask_dir = "./pku_train_dataset/label"
# json_dir = "./pku_train_dataset/json"
image_dir = "./pku_train_dataset/img"
seed = 42
organ_dict = {
    1: "Bladder",
    2: "Colon",
    3: "Femur_Head_L",
    4: "Femur_Head_R",
    5: "Kidney_L",
    6: "Kidney_R",
    7: "Liver",
    8: "Rectum",
    9: "SmallIntestine",
    10: "SpinalCord",
    11: "Stomach"
}
save_dir = "./saved_models"
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

class SAMDataset(Dataset):#包含划窗问题
    def __init__(self,
                 ct_dir: str,
                 mask_dir: str,
                 image_dir: str,
                 hu_min: float = 0.0,
                 hu_max: float = 500.0,
                 organ_idx: int = 1
                 ):
        np.random.seed(seed)
        self.hu_min = hu_min
        self.hu_max = hu_max
        self.organ_idx = organ_idx

        # 初始化存储CT和mask体积的字典
        self.ct_volumes = {}
        self.mask_volumes = {}
        examples = []

        # 获取所有CT文件路径
        ct_paths = sorted(glob(f"{ct_dir}/*.nii.gz"))

        for ct_path in ct_paths:
            ct_idx = int(Path(ct_path).name.replace('.nii.gz', ''))
            mask_path = f"{mask_dir}/{ct_idx}.nii.gz"

            # 加载并预处理CT数据
            ct_data = nib.load(ct_path).get_fdata().astype(np.float32)
            ct_data = np.clip(ct_data, self.hu_min, self.hu_max)#

            # 调整维度并转换为三通道
            ct_tensor = torch.from_numpy(ct_data).permute(2, 0, 1)  # [D, H, W]
            ct_tensor = ct_tensor.unsqueeze(1).repeat(1, 3, 1, 1)  # [D, 3, H, W]
            self.ct_volumes[ct_idx] = ct_tensor

            # 加载并预处理mask数据
            mask_data = nib.load(mask_path).get_fdata().astype(np.uint8)
            mask_data = (mask_data == self.organ_idx).astype(np.float32)

            mask_tensor = torch.from_numpy(mask_data).permute(2, 0, 1)  # [D, H, W]
            mask_tensor = mask_tensor.unsqueeze(1)  # [D, 1, H, W]
            self.mask_volumes[ct_idx] = mask_tensor

            # 获取有效切片编号
            image_paths = sorted(glob(f"{image_dir}/{ct_idx}/slice_*.png"))
            for image_path in image_paths:
                stem = Path(image_path).stem
                slice_num = int(stem.split("_")[-1])

                # 验证切片号的有效性
                if slice_num < ct_tensor.shape[0]:
                    examples.append((ct_idx, slice_num))

        self.examples = examples
        self.length = len(examples)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        ct_idx, slice_num = self.examples[idx]
        x = self.ct_volumes[ct_idx][slice_num]  # [3, H, W]
        y = self.mask_volumes[ct_idx][slice_num]  # [1, H, W]
        return x, y

if __name__ == "__main__":

    dataset = SAMDataset(ct_dir, mask_dir, image_dir, hu_min=0.0, hu_max=500.0, organ_idx=organ)
    val_ratio = 0.2
    n_total = len(dataset)
    n_val = int(n_total * val_ratio)
    n_train = n_total - n_val
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [n_train, n_val], generator=torch.Generator().manual_seed(seed))
    train_loader = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True,num_workers=10, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=train_batch_size, shuffle=False,num_workers=10, pin_memory=True)


    net=ClsNet().to(train_device)
    optimizer=Adam(net.parameters(),lr=3e-3,weight_decay=1e-4)
    scheduler=torch.optim.lr_scheduler.StepLR(optimizer,step_size=5,gamma=0.3)
    loss_module=BCELoss()

    wandb.init(
        project='SAM',
        name=f"{organ}_{organ_dict[organ]}",
        config={
            'learning_rate': optimizer.defaults['lr'],
            'batch_size': train_batch_size,
            'train_epoch': train_epoch,
            'seed': seed,
            'hu_min': dataset.hu_min,
            'hu_max': dataset.hu_max,
            'organ_idx': dataset.organ_idx,
            'scheduler': scheduler.__class__.__name__,
            'optimizer': optimizer.__class__.__name__,
            'step_size': scheduler.step_size,
            'gamma': scheduler.gamma,
            'weight_decay': optimizer.defaults['weight_decay']
        }
    )

    import tqdm
    net.train()
    best_val_loss = float('inf')
    patience = 5  # Number of epochs to wait before stopping if no improvement
    early_stop_counter = 0  # Counter for epochs without improvement
    for i in range(train_epoch):
        net.train()
        average_loss=0.
        with tqdm.tqdm(train_loader) as titer:
            titer.set_description(f'epoch {i}')
            for batch_i,(_x,_y) in enumerate(titer):
                x,y=_x.to(train_device),_y.to(train_device)
                x=net(x)
                y = torch.nn.functional.interpolate(y, scale_factor=1 / 32, mode='nearest')
                optimizer.zero_grad()
                loss_value=loss_module(x,y)
                loss_value.backward()
                optimizer.step()

                average_loss=(average_loss*batch_i+loss_value.item())/(batch_i+1)
                titer.set_postfix_str(f'{average_loss:.3}')

                wandb.log({"ave_loss": average_loss})
        net.eval()
        val_loss = 0.0
        with torch.no_grad():
            for _x, _y in val_loader:
                x, y = _x.to(train_device), _y.to(train_device)
                output = net(x)
                y = torch.nn.functional.interpolate(y, scale_factor=1 / 32, mode='nearest')
                val_loss += loss_module(output, y).item()
        val_loss /= len(val_loader)
        wandb.log({"val_loss": val_loss, "epoch": i})
        print(f"Epoch {i+1}/{train_epoch}, Validation Loss: {val_loss:.4f}")

        # Early stopping logic
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            early_stop_counter = 0  # Reset counter

            torch.save(net.state_dict(), f"{save_dir}/best_model_{organ}.pth")
        else:
            early_stop_counter += 1
            print(f"Validation loss did not improve. Patience: {early_stop_counter}/{patience}")
            if early_stop_counter >= patience:
                print(f"Early stopping triggered at epoch {i + 1}!")
                break

        scheduler.step()  # Step the scheduler after each epoch

    # **每次循环结束后，彻底清理当前 organ 训练的所有缓存**
    del net, optimizer, scheduler, loss_module
    torch.cuda.empty_cache()
    time.sleep(100)

