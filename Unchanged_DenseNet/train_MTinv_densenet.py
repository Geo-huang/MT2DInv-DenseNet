import torch.nn as nn
import torch
import torch.optim as optim
from numpy import shape
from pyts.approximation import PiecewiseAggregateApproximation
from pyts.preprocessing import MinMaxScaler
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import os
import  pandas  as pd
import torch.nn.functional as F
import logging
from torch.utils.tensorboard import SummaryWriter
import time
from pyts.image import GramianAngularField
from sklearn.model_selection import train_test_split
import torchvision.ops as ops


class NewDataset(Dataset):
    def __init__(self, file_list, file_path, validation_ratio=0.1):
        self.file_dir = file_path
        self.data_list = file_list
        self.train_data, self.val_data = train_test_split(file_list, test_size=validation_ratio, random_state=42)

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        data = pd.read_csv(self.file_dir + self.data_list[idx], header=None).values
        y = data[:2176].reshape(34, -1)
        data = data[2176:].reshape(16, -1)
        data = data.reshape(16, 33, -1)

        rho1_norm = data[:, :, 0] - np.average(data[:, :, 0])
        rho1_norm = rho1_norm / np.max(abs(rho1_norm))
        rho1_norm = torch.Tensor(rho1_norm)

        rho2_norm = data[:, :, 2] - np.average(data[:, :, 2])
        rho2_norm = rho2_norm / np.max(abs(rho2_norm))
        rho2_norm = torch.Tensor(rho2_norm)

        phase1_norm = data[:, :, 1] - np.average(data[:, :, 1])
        phase1_norm = phase1_norm / np.max(abs(phase1_norm))
        phase1_norm = torch.Tensor(phase1_norm)

        phase2_norm = data[:, :, 3] - np.average(data[:, :, 3])
        phase2_norm = phase2_norm / np.max(abs(phase2_norm))
        phase2_norm = torch.Tensor(phase2_norm)

        x = torch.cat((rho1_norm.unsqueeze(-1), phase1_norm.unsqueeze(-1),
                       rho2_norm.unsqueeze(-1), phase2_norm.unsqueeze(-1)), dim=-1)

        y = np.log10(y)
        y = torch.Tensor(y)

        return x, y

    def get_validation_set(self):
        return self.val_data



file_path = './sampleset/'
file_list = os.listdir(file_path)

# 使用 train_test_split 划分训练集和验证集
train_list, val_list = train_test_split(file_list, test_size=0.1, random_state=42)

train_dataset = NewDataset(file_list=train_list, file_path=file_path)
train_dataloader = DataLoader(train_dataset, batch_size=100, shuffle=True)

val_dataset = NewDataset(file_list=val_list, file_path=file_path)
val_dataloader = DataLoader(val_dataset, batch_size=100, shuffle=False)

# 获取划分后的训练集样本数
total_train_samples_dataloader = len(train_dataloader.dataset)
print("train_dataloader中的样本总数：", total_train_samples_dataloader)
total_val_samples_dataloader = len(val_dataloader.dataset)
print("val_dataloader中的样本总数：", total_val_samples_dataloader)

def get_logger(filename, verbosity=1, name=None):
    level_dict = {0: logging.DEBUG, 1: logging.INFO, 2: logging.WARNING}
    formatter = logging.Formatter(
        "[%(asctime)s][%(filename)s][line:%(lineno)d][%(levelname)s] %(message)s"
    )
    logger = logging.getLogger(name)
    logger.setLevel(level_dict[verbosity])

    fh = logging.FileHandler(filename, "w")
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    logger.addHandler(sh)

    return logger


class DenseBlock(nn.Module):
    def __init__(self, in_channels, growth_rate, num_layers):
        super(DenseBlock, self).__init__()
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            self.layers.append(self._make_layer(in_channels + i * growth_rate, growth_rate))

    def _make_layer(self, in_channels, out_channels):
        return nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, out_channels, kernel_size=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        )

    def forward(self, x):
        features = [x]
        for layer in self.layers:
            new_features = layer(torch.cat(features, 1))
            features.append(new_features)
        return torch.cat(features, 1)


class TransitionLayer(nn.Module):

    def __init__(self, in_channels, out_channels):
        super(TransitionLayer, self).__init__()
        self.layer = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, out_channels, kernel_size=1),
            nn.AvgPool2d(kernel_size=2, stride=2, ceil_mode=True)
        )
    def forward(self, x):
        return self.layer(x)


class DenseNet(nn.Module):
    def __init__(self, num_blocks, growth_rate, num_classes):
        super(DenseNet, self).__init__()
        self.conv1 = nn.Conv2d(16, 64, kernel_size=7, stride=2, padding=3)
        self.dense_blocks = nn.ModuleList()
        in_channels = 64

        for i, num_layers in enumerate(num_blocks):
            self.dense_blocks.append(DenseBlock(in_channels, growth_rate, num_layers))
            in_channels += num_layers * growth_rate

            if i < len(num_blocks) - 1:
                out_channels = in_channels // 2
                self.dense_blocks.append(TransitionLayer(in_channels, out_channels))
                in_channels = out_channels

        self.fc = nn.Linear(in_channels, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        for block in self.dense_blocks:
            x = block(x)
        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


device = torch.device("cuda:0"  if torch.cuda.is_available() else "cpu")
# 创建DenseNet模型     DenseNetWithTransition
model = DenseNet(num_blocks=[6, 12, 24, 16], growth_rate=32, num_classes=2176).to(device)


criterion = nn.MSELoss(reduction='none').to(device)    #loss张量

params = [p for p in model.parameters() if p.requires_grad]
optimizer = optim.Adam(params, lr=0.0001)

save_path = './MTinv_DenseNet.pth'

logger = get_logger('./expS5g.log')

# 添加tensorboard
writer = SummaryWriter("./logs_train")
val_writer = SummaryWriter("./logs_val")
# 记录最初时间
start_time = time.time()


def train_and_validate(epoch):
    running_loss = 0.0
    model.train()

    for step, data in enumerate(train_dataloader):
        inputs, target = data
        inputs = inputs.to(device).to(torch.float32)
        target = target.to(device).to(torch.float32).reshape(-1, 34 * 64)

        optimizer.zero_grad()
        outputs = model(inputs).reshape(-1, 34 * 64)

        loss = criterion(outputs, target)

        mask1 = target != np.log10(300)
        mask_k = 10
        loss[mask1] *= mask_k
        loss = loss.mean(-1).mean(-1)

        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        print(step)
    print('[%d] loss: %.8f' % (epoch + 1, running_loss / len(train_dataloader)))
    # 在每个 epoch 结束后记录整体的训练损失
    writer.add_scalar("train_loss", running_loss / len(train_dataloader), epoch)
    #保存模型
    torch.save(model.state_dict(), save_path)

    # 在验证集上计算损失
    model.eval()  # 切换到评估模式
    val_loss = 0.0
    with torch.no_grad():
        for val_step, val_data in enumerate(val_dataloader):
            val_inputs, val_target = val_data
            val_inputs = val_inputs.to(device).to(torch.float32)
            val_target = val_target.to(device).to(torch.float32).reshape(-1, 34 * 64)

            val_outputs = model(val_inputs).reshape(-1, 34 * 64)

            val_loss_batch = criterion(val_outputs, val_target)

            mask1 = val_target != np.log10(300)
            mask_k = 10
            val_loss_batch[mask1] *= mask_k
            val_loss += val_loss_batch.mean(-1).mean(-1).item()
            print(val_step)
    val_loss /= len(val_dataloader)
    print(val_loss)
    # 记录训练集和验证集的损失
    writer.add_scalar("val_loss", val_loss, epoch)

    model.train()  # 切换回训练模式


if __name__ == '__main__':
    for epoch in range(200):
        train_and_validate(epoch)

    writer.close()
    val_writer.close()



