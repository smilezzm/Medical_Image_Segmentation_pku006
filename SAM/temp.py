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
ct_dir = "./pku_train_dataset/ct"
mask_dir = "./pku_train_dataset/label"
# json_dir = "./pku_train_dataset/json"
image_dir = "./pku_train_dataset/img"
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
        ct_paths = sorted(glob(f"{ct_dir}/*.nii.gz"))   # 完整的ct路径的列表 (str形式)
        
        count = 0
        examples = []

        # 遍历ct_paths，获取每个立体ct中所有有标记的slice_num
        # 其中image_paths是展示标签的png图的路径，就是从这获取slice_num的
        # 得到的examples是一个list，每个元素是一个tuple，包含ct路径、mask路径和slice_num
        # for example:
        # [('./pku_train_datset/ct/1.nii.gz', './pku_train_dataset/label/1.nii.gz', 15),(...),...]
        # 仅储存path，防止内存过大 （但是ClipSeg不得不储存所有nparray，占用很多内存。因为processor里有padding等，最好一次性处理所有raw data。）
        for ct_path in ct_paths:
            ct_idx = int(Path(ct_path).name.replace('.nii.gz', ''))
            mask_path = f"{mask_dir}/{ct_idx}.nii.gz"
            image_paths = sorted(glob(f"{image_dir}/{ct_idx}/slice_*.png"))
            for image_path in image_paths:
                count += 1
                stem = Path(image_path).stem
                slice_num = int(stem.split("_")[-1])
                examples.append((ct_path, mask_path, slice_num))
        self.length = count
        self.examples = examples # list，每个元素是tuple，包含ct路径、mask路径和有标记的slice_num

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        x = nib.load(self.examples[idx][0]).get_fdata().astype(np.float32)
        x = x[:,:,self.examples[idx][2]]
        x = np.clip(x, self.hu_min, self.hu_max)
        x = torch.from_numpy(x)
        x = torch.stack([x,x,x],dim=0)    # (3, H=512, W=512)
        y = nib.load(self.examples[idx][1]).get_fdata().astype(np.uint8)
        y = y[:,:,self.examples[idx][2]]
        y = (y == self.organ_idx).astype(np.float32)
        y = torch.from_numpy(y)
        y = y.unsqueeze(0)  # (1, H=512, W=512)
        return x, y   # x is 2D image, y is 2D mask

if __name__ == "__main__":
    # 开始训练Bladder的Resnet
    dataset = SAMDataset(ct_dir, mask_dir, image_dir, hu_min=0.0, hu_max=500.0, organ_idx=1)
    val_ratio = 0.2
    n_total = len(dataset)
    n_val = int(n_total * val_ratio)
    n_train = n_total - n_val
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [n_train, n_val], generator=torch.Generator().manual_seed(seed))
    train_loader = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=train_batch_size, shuffle=False)


    net=ClsNet().to(train_device)
    optimizer=Adam(net.parameters(),lr=3e-3,weight_decay=1e-4)
    scheduler=torch.optim.lr_scheduler.StepLR(optimizer,step_size=5,gamma=0.3)
    loss_module=BCELoss()   

    wandb.init(
        project='SAM',
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
    for i in range(train_epoch):
        net.train()
        average_loss=0.
        with tqdm.tqdm(train_loader) as titer:
            titer.set_description(f'epoch {i}')
            for batch_i,(_x,_y) in enumerate(titer):
                x,y=_x.to(train_device),_y.to(train_device)
                x=net(x)
                y = torch.nn.functional.interpolate(y, scale_factor=1 / 32, mode='nearest')   #原本y(B,1,512,512)，需要弄成和x一样大(B,1,16,16)
                optimizer.zero_grad()
                loss_value=loss_module(x,y)
                loss_value.backward()
                optimizer.step()

                average_loss=(average_loss*batch_i+loss_value.item())/(batch_i+1)
                titer.set_postfix_str(f'{average_loss:.3}')

                wandb.log({"ave_loss": average_loss})
        scheduler.step()

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
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(net.state_dict(), "./SAM/best_model_bladder.pth")  # 储存针对Bladder的粗分类ResNet模型参数


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
