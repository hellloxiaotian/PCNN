import torch
import torch.nn as nn
import torch.nn.functional as F

import torchvision.models as models
     
# resnet18+多区域拼接+图像校正+多分支输出
class PCNN(nn.Module):      
    def __init__(self, num_class=7, device='cpu'):
        super(PCNN, self).__init__()
        
        self.resnet = models.resnet18()
        self.resnet1 = models.resnet18()
        self.resnet2 = models.resnet18()
        self.resnet3 = models.resnet18()
        self.resnet4 = models.resnet18()
        self.resnet5 = models.resnet18()
        self.resnet6 = models.resnet18()
        
        checkpoint = torch.load('models/resnet18_msceleb.pth', map_location=device)
        
        # 加载多个模型参数
        self.resnet.load_state_dict(checkpoint['state_dict'], strict=True)
        self.resnet1.load_state_dict(checkpoint['state_dict'], strict=True)
        self.resnet2.load_state_dict(checkpoint['state_dict'], strict=True)
        self.resnet3.load_state_dict(checkpoint['state_dict'], strict=True)
        self.resnet4.load_state_dict(checkpoint['state_dict'], strict=True)
        self.resnet5.load_state_dict(checkpoint['state_dict'], strict=True)
        self.resnet6.load_state_dict(checkpoint['state_dict'], strict=True)

        # 特征提取层
        self.features1 = nn.Sequential(*list(self.resnet.children())[:-3])
        self.features2 = nn.Sequential(*list(self.resnet1.children())[:-3])
        self.features3 = nn.Sequential(*list(self.resnet2.children())[:-3])
        self.features4 = nn.Sequential(*list(self.resnet3.children())[:-3])
        self.features6 = nn.Sequential(*list(self.resnet5.children())[:-3])
        self.features7 = nn.Sequential(*list(self.resnet6.children())[:-3])
        self.features8 = nn.Sequential(*list(self.resnet.children())[-3:-2])

        # 区域划分参数
        self.w1 = torch.tensor(0.5)
        self.w2 = torch.tensor(0.75)
        self.h1 = torch.tensor(0.5)
        self.h2 = torch.tensor(0.65)

        # STN定位网络
        self.fc_loc = nn.Sequential(
            nn.Linear(512 * 7 * 7, 32),
            nn.ReLU(),
            nn.Linear(32, 3 * 2)
        )

        # 多分支分类层
        self.fc = nn.Linear(512, num_class)
        self.fc2 = nn.Linear(256, num_class)
        self.fc3 = nn.Linear(256, num_class)
        self.fc4 = nn.Linear(256, num_class)
        self.fc6 = nn.Linear(256, num_class)
        self.fc7 = nn.Linear(256, num_class)
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.relu = nn.ReLU()
        
    def forward(self, x):
    
        w = x.size(3)
        h = x.size(2)
        w_1 = int(w * self.w1)
        w_2 = int(w * self.w2)
        h_1 = int(h * self.h1)
        h_2 = int(h * self.h2)

        # 划分区域
        x2 = x[:, :, 0:h_1, 0:w_1]
        x3 = x[:, :, 0:h_1, w_1:w]
        x4 = x[:, :, h_1:h_2, 0:w_1]
        x6 = x[:, :, h_1:h_2, w_1:w]
        x7 = x[:, :, h_2:h, :]
        
        # 提取特征
        x1 = self.features1(x)        
        x2 = self.features2(x2)
        x3 = self.features3(x3)
        x4 = self.features4(x4)
        x6 = self.features6(x6)
        x7 = self.features7(x7)
        
        # 特征拼接与调整
        x8 = torch.cat([x2, x3], dim=3)
        x8 = F.interpolate(x8, size=(x8.size(2), x1.size(3)), mode='bilinear', align_corners=True)
        x9 = torch.cat([x4, x6], dim=3)
        x9 = F.interpolate(x9, size=(x9.size(2), x1.size(3)), mode='bilinear', align_corners=True)
        x10 = torch.cat([x8, x9, x7], dim=2)
        x10 = F.interpolate(x10, size=(x1.size(2), x1.size(3)), mode='bilinear', align_corners=True)
        
        # 各区域分支输出
        x2 = self.avgpool(x2)
        x2 = x2.view(x2.size(0), -1)
        x2 = self.fc2(x2)
        
        x3 = self.avgpool(x3)
        x3 = x3.view(x.size(0), -1)
        x3 = self.fc3(x3)
        
        x4 = self.avgpool(x4)
        x4 = x4.view(x.size(0), -1)
        x4 = self.fc4(x4)
        
        x6 = self.avgpool(x6)
        x6 = x6.view(x.size(0), -1)
        x6 = self.fc6(x6)
        
        x7 = self.avgpool(x7)
        x7 = x7.view(x.size(0), -1)
        x7 = self.fc7(x7)
        
        # 分支特征融合
        heads = x2 + x3 + x4 + x6 + x7
        
        # STN校正
        xs = self.features8(x10)
        xs = xs.view(x1.size(0), -1)
        theta = self.fc_loc(xs)
        theta = theta.view(-1, 2, 3)

        grid = F.affine_grid(theta, x1.size(), align_corners=True)
        x11 = F.grid_sample(x10, grid, align_corners=True)
        
        x1 = x1 + x11  # 融合特征
        
        # 主分支分类
        x1 = self.features8(x1)
        x1 = self.avgpool(x1)
        x1 = x1.view(x.size(0), -1)
        x1 = self.fc(x1)
        
        return x1, heads
