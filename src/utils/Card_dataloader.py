import torch
from torch.utils.data import Dataset
import numpy as np
from src.utils.custom_typing import CardData  # 假设你有类似的类型定义，如果没有，可以去掉
import matplotlib.pyplot as plt

class CardDataset(Dataset):
    def __init__(self, images_pair: np.ndarray, labels_pair: np.ndarray):
        """
        Args:
            images_pair: numpy数组，形状 (N, 2, H, W, C)，存储图片对
            labels_pair: numpy数组，形状 (N, 2, 2)，存储对应标签
        """
        super().__init__()
        self.images = images_pair
        self.labels = labels_pair
        self.show_image = False

    def __getitem__(self, index):
        images = self.images[index]  # shape: (2, C, H, W)
        labels = self.labels[index]  # shape: (2, 2)

        img_x = torch.tensor(images[0], dtype=torch.float32)
        img_y = torch.tensor(images[1], dtype=torch.float32)

        # 标签转tensor，转long用于分类等
        label_x = torch.tensor(labels[0], dtype=torch.long)
        label_y = torch.tensor(labels[1], dtype=torch.long)
        
        
        if self.show_image:
            # 绘制图片对
            fig, axs = plt.subplots(1, 2, figsize=(8, 4))
            axs[0].imshow(img_x.permute(1, 2, 0).numpy())
            axs[0].set_title(f'Image X\nLabels: {labels[0]}')
            axs[0].axis('off')
            axs[1].imshow(img_y.permute(1, 2, 0).numpy())
            axs[1].set_title(f'Image Y\nLabels: {labels[1]}')
            axs[1].axis('off')
            plt.tight_layout()
            plt.show()  # 非阻塞显示
            #plt.pause(0.001)  # 保证图能更新

        # 如果你有定义数据结构，可以用这个返回，否则用字典也可以
        return CardData(
            x=img_x,
            y=img_y,
            card_x=torch.tensor(label_x[0], dtype=torch.long),
            card_y=torch.tensor(label_y[0], dtype=torch.long),
            suit_x=torch.tensor(label_x[1], dtype=torch.long),
            suit_y=torch.tensor(label_y[1], dtype=torch.long),
        )


    def __len__(self):
        return len(self.images)



