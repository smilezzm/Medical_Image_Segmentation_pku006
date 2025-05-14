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
import argparse  # 添加用于解析命令行参数的模块

class ClsNet(ResNet):
    '''
    针对Bladder的ResNet粗分类器
    input: tensor(B, Channels=3, H, W), (clipped) HU values
    output: tensor(B, 1, H/32, W/32), values between 0 and 1
    purpose: 将某个ct截面图片分块，根据每一块是否包含Bladder给出0到1的值（0表示没有，1表示很有）
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
train_epoch=30 # 针对Bladder的resnet的训练轮次
# 一般不用跑到30轮

train_batch_size=16
ct_dir = "./CLAPATPS/ct"
mask_dir = "./CLAPATPS/label"
# json_dir = "./pku_train_dataset/json"
image_dir = "./CLAPATPS/img"
seed = 42

class SAMDataset(Dataset):
    def __init__(self,
                 ct_dir: str,
                 mask_dir: str,
                 image_dir: str,
                 hu_min: float=0.0,
                 hu_max: float=500.0,
                 organ_idx: int=1
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
            ct_data = np.clip(ct_data, self.hu_min, self.hu_max)

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
    # 添加命令行参数解析
    parser = argparse.ArgumentParser(description='Train a ResNet model for medical image segmentation.')
    parser.add_argument('--organ', type=int, default=1, help='Organ index for segmentation.')
    parser.add_argument('--organ-name', type=str, default="colon", help='Organ name for the model filename.')
    parser.add_argument('--gpu-id', type=int, default=0, help='GPU ID to use.')
    parser.add_argument('--hu-min', type=int, default=0, help='GPU ID to use.')
    parser.add_argument('--hu-max', type=int, default=500, help='GPU ID to use.')
    args = parser.parse_args()

    # 设置GPU
    if torch.cuda.is_available():
        train_device = torch.device(f'cuda:{args.gpu_id}')
    else:
        train_device = torch.device('cpu')

    # 使用命令行参数
    organ_idx = args.organ
    organ_name = args.organ_name

    dataset = SAMDataset(ct_dir, mask_dir, image_dir, hu_min=args.hu_min, hu_max=args.hu_max, organ_idx=organ_idx)
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
        name = f"{organ_idx}_{organ_name}",
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
            torch.save(net.state_dict(), f"./sambase/resnetmodel/best_model_{organ_idx}_{organ_name}_{dataset.hu_min}to{dataset.hu_max}.pth")  # 储存针对Bladder的粗分类ResNet模型参数
        else:
            early_stop_counter += 1
            print(f"Validation loss did not improve. Patience: {early_stop_counter}/{patience}")
            if early_stop_counter >= patience:
                print(f"Early stopping triggered at epoch {i + 1}!")
                break

        scheduler.step()  # Step the scheduler after each epoch

    # pick some slices to visualize
    # unneccessary
    # import matplotlib.pyplot as plt
    # import matplotlib.patches as patches

    # def show_ct_with_patches(ct_slice, pred_map, patch_size=32, threshold=0.5):
    #     """
    #     ct_slice: 2D numpy array (H, W)
    #     pred_map: 2D numpy array (H/patch_size, W/patch_size) of scores in [0,1]
    #     """
    #     fig, ax = plt.subplots(figsize=(6,6))
    #     ax.imshow(ct_slice, cmap='gray')
    #     h_p, w_p = pred_map.shape
    #     for i in range(h_p):
    #         for j in range(w_p):
    #             if pred_map[i, j] > threshold:
    #                 rect = patches.Rectangle(
    #                     (j * patch_size, i * patch_size),
    #                     patch_size, patch_size,
    #                     linewidth=1, edgecolor='r', facecolor='none'
    #                 )
    #                 ax.add_patch(rect)
    #     plt.axis('off')
    #     plt.show()
    #     plt.savefig("./SAM/ct_with_patches.png", dpi=300, bbox_inches='tight')

    # ct_path, _, slice_idx = dataset.examples[0]
    # ct_np = nib.load(ct_path).get_fdata().astype(np.float32)[:,:,slice_idx]
    # ct_np_clipped = np.clip(ct_np, dataset.hu_min, dataset.hu_max)
    # x_gray = torch.from_numpy(ct_np_clipped)
    # x_rgb = x_gray.unsqueeze(0).repeat(3,1,1).unsqueeze(0).to(train_device)  # (1, 3, H, W)
    # with torch.no_grad():
    #     pred_map = net(x_rgb).squeeze(0).squeeze(0)
    # show_ct_with_patches(ct_np_clipped, pred_map.cpu().numpy(), patch_size=32, threshold=0.5)
