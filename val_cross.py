import os
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.models as models
import torch.utils.data.distributed
import matplotlib.pyplot as plt
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import numpy as np
import datetime
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from network.models import *  # 假设Model_5已在该路径定义
from tqdm import tqdm
from torchsummary import summary
from thop import profile
plt.switch_backend('agg')

# -------------------------- 1. 基础配置（新增测试子目录列表）--------------------------
now = datetime.datetime.now()
time_str = now.strftime("[%m-%d]-[%H-%M]-")
dataset_name = 'rafdb'  # 主数据集名称（不变）
test_subdirs = ['v30', 'v45', 'occlusion']  # 需测试的子目录（适配图中结构）

# 模型与路径基础配置（不变）
model_name = dataset_name + '_'
checkpoint_path = './checkpoints/' + model_name + time_str + '.pth'
best_checkpoint_path = './checkpoints/' + model_name + time_str + '_best.pth'
model_path = './experiment/' + dataset_name + '/' + dataset_name + '.pth'

# 训练/验证参数（不变）
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'  # 增加CPU兼容
net = 5
eval = True  # 固定为验证模式
lr = 0.01
momentum = 0.9
weight_decay = 1e-4
epochs = 100
ls = 15
batch_size = 1
workers = 4
print_freq = 10
pretrained = True


def main():
    # 打印基础信息（不变）
    tqdm.write('Training time: ' + now.strftime("%m-%d %H:%M"))
    tqdm.write(f'device:    {device}')
    tqdm.write(f'dataset:    {dataset_name}')
    tqdm.write(f'Testing subdirectories: {test_subdirs}')  # 新增：打印待测试子目录

    # -------------------------- 2. 加载模型（仅加载1次，循环外执行）--------------------------
    model = PCNN(num_class=7, device=device)  # rafdb为7类情绪
    model = model.to(device)
    criterion_cls = nn.CrossEntropyLoss().to(device)
    cudnn.benchmark = True  # 加速网络

    # 加载预训练权重（原逻辑保留，修复True/False均加载的潜在问题）
    if pretrained:
        tqdm.write('Loading pretrained model...')
        try:
            checkpoint = torch.load(model_path, map_location=torch.device(device))
            pretrained_state_dict = checkpoint['state_dict']
            model_state_dict = model.state_dict()

            # 过滤不匹配的权重键（避免模型结构不兼容）
            matched_state_dict = {k: v for k, v in pretrained_state_dict.items() if k in model_state_dict}
            model_state_dict.update(matched_state_dict)
            model.load_state_dict(model_state_dict)
            tqdm.write('Pretrained model loaded successfully!')
        except Exception as e:
            tqdm.write(f'Failed to load pretrained model: {str(e)}')
            return
    else:
        tqdm.write('Warning: Not using pretrained model (pretrained=False)')
        # 原False逻辑可能冗余，此处简化为仅提示

    # -------------------------- 3. 循环测试每个子目录（核心修改）--------------------------
    for subdir in test_subdirs:
        tqdm.write(f'\n=== Starting test on subdirectory: {subdir} ===')
        
        # 3.1 生成当前子目录的专属路径（防覆盖）
        data_path = f'/datasets/occlusion/{dataset_name}/{subdir}'  # 适配图结构：dataset/rafdb/[v30/v45/occlusion]
        # 结果文件路径（含子目录名，避免覆盖）
        txt_name = f'./logs/{model_name}{time_str}{subdir}_results.txt'  # 日志文件
        cm_path = f'./logs/{model_name}{time_str}{subdir}_confusion_matrix.png'  # 混淆矩阵
        curve_name = f'./logs/{model_name}{time_str}{subdir}_curve.png'  # 曲线文件（eval模式暂未用）

        # 3.2 路径容错：跳过不存在的子目录
        if not os.path.exists(data_path):
            tqdm.write(f'Warning: Data path {data_path} does not exist. Skipping {subdir}...')
            continue

        # 3.3 创建当前子目录的验证数据集（ImageFolder适配类别子文件夹结构）
        val_dataset = datasets.ImageFolder(
            root=data_path,
            transform=transforms.Compose([
                transforms.Resize((224, 224)),  # 适配Model_5输入尺寸
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],  # ImageNet标准化（同预训练模型）
                                     std=[0.229, 0.224, 0.225])
            ])
        )
        val_loader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,  # 验证集不打乱
            num_workers=workers,
            pin_memory=True  # 加速GPU数据传输
        )
        tqdm.write(f'Loaded {len(val_dataset)} samples from {data_path}')

        # 3.4 调用验证函数（传入子目录专属参数）
        validate(
            val_loader=val_loader,
            model=model,
            criterion_cls=criterion_cls,
            subdir=subdir,  # 标识当前子目录
            txt_name=txt_name,  # 日志保存路径
            cm_path=cm_path  # 混淆矩阵保存路径
        )

    tqdm.write('\n=== All subdirectories tested completed! ===')


# -------------------------- 4. 验证函数（修改：支持多目录结果保存）--------------------------
def validate(val_loader, model, criterion_cls, subdir, txt_name, cm_path):
    # 初始化指标计算器
    losses = AverageMeter('Loss', ':.4f')
    top1 = AverageMeter('Accuracy', ':6.3f')
    progress = ProgressMeter(
        len(val_loader),
        [losses, top1],
        prefix=f'Test ({subdir}): '  # 进度条标识子目录
    )

    # 情绪类别标签（rafdb固定7类，与Model_5输出匹配）
    labels_name = ['Neutral', 'Happiness', 'Sadness', 'Surprise', 'Fear', 'Disgust', 'Anger']
    # if dataset_name == 'affectnet-8':  # 若需支持8类，可保留此逻辑
    #     labels_name.append('contempt')

    model.eval()  # 进入验证模式（禁用Dropout/BatchNorm更新）
    all_preds = []  # 保存所有预测结果
    all_targets = []  # 保存所有真实标签

    with torch.no_grad():  # 禁用梯度计算（加速+防内存泄漏）
        for i, (images, targets) in enumerate(val_loader):
            # 数据送设备
            targets = targets.to(device)
            images = images.to(device)

            # 模型前向传播（Model_5输出：主分类结果out + 辅助分类结果heads）
            out, heads = model(images)
            # 计算损失（主损失+辅助损失，放大10倍以平衡梯度）
            loss = (criterion_cls(out, targets) + criterion_cls(heads, targets)) * 10

            # 计算预测结果与准确率
            predict_np = np.argmax(out.cpu().detach().numpy(), axis=-1)
            labels_np = targets.cpu().numpy()
            acc = accuracy(out, targets)

            # 更新指标（按batch_size加权平均）
            losses.update(loss.item(), images.size(0))
            top1.update(acc, images.size(0))

            # 收集结果（用于后续混淆矩阵计算）
            all_preds.extend(predict_np)
            all_targets.extend(labels_np)

            # 打印进度
            if i % print_freq == 0:
                progress.display(i)

        # -------------------------- 5. 结果保存与输出（新增文件写入）--------------------------
        # 5.1 打印整体结果
        overall_result = f'**** {subdir} - Overall Accuracy: {top1.avg:.3f}, Average Loss: {losses.avg:.4f} ***'
        tqdm.write(overall_result)

        # 5.2 写入日志文件（含时间、子目录、整体指标）
        with open(txt_name, 'w', encoding='utf-8') as f:
            f.write(f"Test Time: {now.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Subdirectory: {subdir}\n")
            f.write(f"Total Samples: {len(val_loader.dataset)}\n")
            f.write(f"Overall Accuracy: {top1.avg:.3f}\n")
            f.write(f"Average Loss: {losses.avg:.4f}\n\n")

        # 5.3 计算并保存混淆矩阵
        cm = confusion_matrix(all_targets, all_preds)
        plt.figure(figsize=(10, 8))
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels_name)
        disp.plot(include_values=True, cmap='Blues', xticks_rotation='vertical')
        plt.title(f'Confusion Matrix - {subdir}')
        plt.tight_layout()  # 防止标签截断
        plt.savefig(cm_path, dpi=300)  # 高分辨率保存
        plt.close()
        tqdm.write(f'Confusion matrix saved to: {cm_path}')

        # 5.4 计算并保存每类准确率
        class_accuracy = cm.diagonal() / cm.sum(axis=1)  # 每类准确率（对角线/行和）
        class_result = "\nClass-wise Accuracy:\n"
        for i, (class_name, acc) in enumerate(zip(labels_name, class_accuracy)):
            class_acc_str = f"{class_name}: {acc:.4f}"
            class_result += f"{class_acc_str}\n"
            tqdm.write(class_acc_str)

        # 写入每类准确率到日志
        with open(txt_name, 'a', encoding='utf-8') as f:
            f.write(class_result)

    return top1.avg, losses.avg


# -------------------------- 以下为原工具类（不变）--------------------------
class AverageMeter(object):
    """计算并存储当前值、平均值、总和、计数"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    """显示验证进度"""
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        tqdm.write('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def accuracy(logits, labels):
    """计算单batch准确率（百分比）"""
    acc = (logits.argmax(dim=-1) == labels).float().mean()
    return acc * 100.0


class RecorderMeter(object):
    """原训练曲线记录类（eval模式暂未使用，保留以备扩展）"""
    def __init__(self, total_epoch):
        self.reset(total_epoch)

    def reset(self, total_epoch):
        self.total_epoch = total_epoch
        self.current_epoch = 0
        self.epoch_losses = np.zeros((self.total_epoch, 2), dtype=np.float32)
        self.epoch_accuracy = np.zeros((self.total_epoch, 2), dtype=np.float32)

    def update(self, idx, train_loss, train_acc, val_loss, val_acc):
        self.epoch_losses[idx, 0] = train_loss * 30
        self.epoch_losses[idx, 1] = val_loss * 30
        self.epoch_accuracy[idx, 0] = train_acc
        self.epoch_accuracy[idx, 1] = val_acc
        self.current_epoch = idx + 1

    def plot_curve(self, save_path):
        if save_path is None:
            return
        title = f'Train/Val Curve - {save_path.split("_")[-2]}'  # 含子目录名
        dpi = 80
        width, height = 1800, 800
        figsize = width / float(dpi), height / float(dpi)
        fig = plt.figure(figsize=figsize)
        x_axis = np.arange(self.total_epoch)
        plt.xlim(0, self.total_epoch)
        plt.ylim(0, 100)
        plt.xticks(np.arange(0, self.total_epoch + 5, 5))
        plt.yticks(np.arange(0, 100 + 5, 5))
        plt.grid()
        plt.title(title, fontsize=20)
        plt.xlabel('Epoch', fontsize=16)
        plt.ylabel('Accuracy/Loss', fontsize=16)

        # 训练准确率
        plt.plot(x_axis, self.epoch_accuracy[:, 0], 'g-', label='Train-Accuracy', lw=2)
        # 验证准确率
        plt.plot(x_axis, self.epoch_accuracy[:, 1], 'y-', label='Val-Accuracy', lw=2)
        # 训练损失（×30）
        plt.plot(x_axis, self.epoch_losses[:, 0], 'g:', label='Train-Loss×30', lw=2)
        # 验证损失（×30）
        plt.plot(x_axis, self.epoch_losses[:, 1], 'y:', label='Val-Loss×30', lw=2)

        plt.legend(loc=4, fontsize=10)
        fig.savefig(save_path, dpi=dpi, bbox_inches='tight')
        plt.close(fig)


if __name__ == '__main__':
    main()