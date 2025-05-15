from temp import SAMDataset, ClsNet
import torch
from torch import Tensor
from torch.utils.data import DataLoader

'''
需要先将github上SAM的全部文件下载到本地./下，
同时，保证temp.py和pku_train_dataset也在./下，
在终端运行python SAMseg.py即可看到所有的可视化结果.
'''


train_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def get_prompt_points(x_cls,pred_threshold,max_points):
    #x_cls是分类网络预测的类别信息
    sample_set = torch.where(x_cls > pred_threshold) #tuple(tensor(indices1),tensor(indices2),tensor(indices3),tensor(indices4)),(points_num,points_num,points_num,points_num)
    points_num = sample_set[0].__len__()
    batch_i_start = 0
    batch_i_end = 0
    sample_points = torch.zeros((x_cls.shape[0], 1, 6, 2))
    sample_points_label = torch.zeros((x_cls.shape[0], 1, 6))
    while batch_i_start < points_num:
        while batch_i_end < points_num and sample_set[0][batch_i_end] == sample_set[0][
            batch_i_start]: batch_i_end = batch_i_end + 1
        batch_points_num = batch_i_end - batch_i_start
        batch_sample_num = min(max_points, batch_points_num)
        batch_sample_interval = batch_points_num // batch_sample_num
        for batch_sample_i in range(batch_sample_num):
            sample_points_label[sample_set[0][batch_i_start]][0][batch_sample_i] = 1
            sample_points[sample_set[0][batch_i_start]][0][batch_sample_i] = Tensor(
                [sample_set[3][batch_sample_i * batch_sample_interval] * 32 + 16,
                 sample_set[2][batch_sample_i * batch_sample_interval] * 32 + 16]
            )
        batch_i_start = batch_i_end
    return list(set(sample_set[0])),sample_points,sample_points_label


from efficient_sam.efficient_sam import build_efficient_sam
sam_model = build_efficient_sam(
        encoder_patch_embed_dim=192,
        encoder_num_heads=3,
        checkpoint="./weights/efficient_sam_vitt.pt",
    ).eval()
sam_model=sam_model.to(train_device)


ct_dir = "./pku_train_dataset/ct"
mask_dir = "./pku_train_dataset/label"
# json_dir = "./pku_train_dataset/json"
image_dir = "./pku_train_dataset/img"
seed = 42
train_batch_size = 16
hu_min=20
hu_max=80

dataset = SAMDataset(ct_dir, mask_dir, image_dir, hu_min=hu_min, hu_max=hu_max, organ_idx=7)
val_ratio = 0.2
n_total = len(dataset)
n_val = int(n_total * val_ratio)
n_train = n_total - n_val
train_dataset, val_dataset = torch.utils.data.random_split(dataset, [n_train, n_val], generator=torch.Generator().manual_seed(seed))
train_loader = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=train_batch_size, shuffle=False)

net = ClsNet().to(train_device)
net.load_state_dict(torch.load("./saved_models/best_model_7_liver_20to80.pth"))

import cv2
import numpy as np
import matplotlib.pyplot as plt
import tqdm
pred_threshold=0.8#只有大于该阈值才有可能被选择为提示点
max_points=3

# total_dice = 0.0
# num_batches = 0
correct_pixels=0
wrong_pixels=0
with tqdm.tqdm(val_loader) as titer:
    titer.set_description(f'val')
    for _x, _y in titer:
        with torch.no_grad():
            x, y = _x.to(train_device), _y.to(train_device)

            x_cls = net(x) #(B,1,16,16)
            y = torch.nn.functional.interpolate(y, scale_factor=1 / 32, mode='nearest')

            batch_list,sample_points,sample_points_label=get_prompt_points(x_cls,
                                                                           pred_threshold=pred_threshold,
                                                                           max_points=max_points)
            sample_points=sample_points.to(train_device)
            sample_points_label=sample_points_label.to(train_device)
            predicted_logits, predicted_iou = sam_model(
                x/(hu_max-hu_min),
                # 注意 net输入的x并未归一化，而SAM模型输入的x需要归一化
                sample_points,
                sample_points_label,
            )#predicted_logits(16,1,3,512,512)  predicted_iou(16,1,3) 模型有个超参num_multimask_outputs = 3
            sorted_ids = torch.argsort(predicted_iou, dim=-1, descending=True)
            predicted_iou = torch.take_along_dim(predicted_iou, sorted_ids, dim=2) #按iou排序
            predicted_logits = torch.take_along_dim(
                predicted_logits, sorted_ids[..., None, None], dim=2
            ) #同步排序，选第一个

            # 取 IoU 最高的预测掩码（索引 0）
            best_pred_logits = predicted_logits[:, :, 0, :, :]  # (16,1,512,512)

            # 将 logits 转换为概率并二值化（阈值 0.5）
            pred_mask = torch.sigmoid(best_pred_logits) > 0.5  # (16,1,512,512)

            _y= _y.to(train_device)

            # 计算重合的面积
            correct_pixels += ((pred_mask == 1) & (_y == 1)).sum().item()
            wrong_pixels += ((pred_mask == 1) ^ (_y == 1)).sum().item()

            # # 计算 Dice Score
            # intersection = (pred_mask & _y.bool()).sum(dim=(2, 3)).float()
            # union = (pred_mask | _y.bool()).sum(dim=(2, 3)).float()
            # dice_score = (2 * intersection) / (union + 1e-6)  # 加平滑项避免除零
            # batch_dice = dice_score.mean()  # 取 batch 平均
            #
            # # 更新累计值
            # total_dice += batch_dice.item() * x.size(0)  # 乘以batch size
            # num_batches += x.size(0)
            #
            # # 更新进度条显示
            # titer.set_postfix(current_batch_dice=batch_dice.item())

            for sample_batch in batch_list: #有prompt的才处理，需要在之前处理得分
                img=(_x[sample_batch][0]*255.0).numpy().astype(np.uint8)

                pred_mask = torch.ge(predicted_logits[sample_batch, 0, 0, :, :], 0).cpu().detach().numpy()
                pred = pred_mask.astype(np.uint8)*255

                img=cv2.cvtColor(img,cv2.COLOR_GRAY2RGB)
                ret, binary = cv2.threshold(pred, 127, 255, cv2.THRESH_BINARY)
                contours, hierarchy = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                img_with_contours=cv2.drawContours(img, contours, -1, [0,255,0], 3)
                for point_i in range(6):
                    if sample_points_label[sample_batch][0][point_i]==1:
                        center_point=(int(sample_points[sample_batch][0][point_i][0].item()),int(sample_points[sample_batch][0][point_i][1].item()))
                        img_with_contours = cv2.circle(
                            img_with_contours, center_point, 3, [255, 0, 0], 3
                        )

                plt.imshow(img_with_contours)
                plt.show()

# 计算整个验证集的平均Dice
mean_dice=2*correct_pixels/(2*correct_pixels+wrong_pixels)
print(f'\nValidation Mean Dice Score: {mean_dice:.4f}')