import os
import time
import shutil
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import matplotlib.pyplot as plt
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import numpy as np
import datetime
from collections import defaultdict

from network.models import *
from tqdm import tqdm
import matplotlib.pyplot as plt

plt.switch_backend('agg')
now = datetime.datetime.now()
time_str = now.strftime("[%m-%d]-[%H-%M]-")

device = 'cuda:0'
dataset_name = 'rafdb'
data_path = os.path.join("dataset", dataset_name)
model_name = dataset_name + '_'
checkpoint_path = './checkpoints/' + model_name + time_str + '.pth'
best_checkpoint_path = './checkpoints/' + model_name + time_str + '_best.pth'
txt_name = './logs/' + model_name + time_str + '.txt'
curve_name = './logs/' + model_name + time_str + '.png'
pretrained_model_path = './experiment/' + dataset_name + '/' + dataset_name + '.pth'

alpha                = 12
beta                 = 8
eval                 = False
lr                   = 0.01 
momentum             = 0.9
weight_decay         = 1e-4
epochs               = 100
ls                   = 15
batch_size           = 128
workers              = 8
print_freq           = 100
pretrained           = True
do_validation        = True  # set to False to skip validation during training

traindir = os.path.join(data_path, 'train')
valdir = os.path.join(data_path, 'test')

def main():
    best_acc = 0.0
    start_epoch = 0

    print('Training time: ' + now.strftime("%m-%d %H:%M"))
    print('device:    ' + device)
    print('dataset:    ' + dataset_name)
    print('alpha:  ' + str(alpha) + '   beta:  ' + str(beta))

    # 初始化模型
    model = PCNN(num_class=7, device=device)
    if pretrained:
        checkpoint = torch.load(pretrained_model_path, map_location=device)['state_dict']
        model.load_state_dict(checkpoint, strict=False)
    model = model.to(device)

    # 不带权重的交叉熵损失
    criterion_cls = nn.CrossEntropyLoss().to(device)

    optimizer = torch.optim.SGD(model.parameters(), lr, momentum=momentum, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=ls, gamma=0.5)
    recorder = RecorderMeter(epochs)
    
    cudnn.benchmark = True

    # 加载数据集
    train_dataset = datasets.ImageFolder(traindir, transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomApply([transforms.RandomRotation(20), transforms.RandomCrop(224, padding=32)], p=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        transforms.RandomErasing(scale=(0.02, 0.25)),
    ]))

    val_dataset = datasets.ImageFolder(valdir, transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ]))

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=workers, pin_memory=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=workers, pin_memory=True)
    
    # 训练循环
    for epoch in tqdm(range(start_epoch, epochs)):
        start_time = time.time()
        current_learning_rate = optimizer.state_dict()['param_groups'][0]['lr']
        tqdm.write('Current learning rate: ' + str(current_learning_rate))
        with open(txt_name, 'a') as f:
            f.write('Current learning rate: ' + str(current_learning_rate) + '\n')
    
        train_acc, train_los = train(train_loader, model, criterion_cls, optimizer, epoch+1)
        if do_validation:
            val_acc, val_los = validate(val_loader, model, criterion_cls)
            # ensure numeric float
            try:
                val_acc = float(val_acc)
            except Exception:
                # if tensor-like
                val_acc = float(val_acc.item()) if hasattr(val_acc, 'item') else float(np.array(val_acc))
        else:
            # when skipping validation, use training accuracy as a proxy (or 0.0)
            val_acc = float(train_acc)
            val_los = float(train_los)
        
        scheduler.step()
        recorder.update(epoch, train_los, train_acc, val_los, val_acc)    
        recorder.plot_curve(curve_name)  
    
        is_best = val_acc > best_acc
        best_acc = max(best_acc, val_acc)

        tqdm.write(f'Current best accuracy: {best_acc:.3f}')
        with open(txt_name, 'a') as f:
            f.write(f'********************Current best accuracy: {best_acc:.3f}\n')
    
        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'best_acc': best_acc,
            'optimizer': optimizer.state_dict(),
            'recorder': recorder,
        }, is_best)
    
        end_time = time.time()
        epoch_time = end_time - start_time
        tqdm.write("An Epoch Time: " + str(epoch_time))
        with open(txt_name, 'a') as f:      
            f.write('An epoch time: ' + str(epoch_time) + '\n')
    
    print('Training time: ' + now.strftime("%m-%d %H:%M"))
    print('device:    ' + device)
    print('dataset:    ' + dataset_name) 
    print('best_checkpoint_path: ' + best_checkpoint_path)
    print('alpha:  ' + str(alpha) + '   beta:  ' + str(beta))


# 训练函数
def train(train_loader, model, criterion_cls, optimizer, epoch):
    losses = AverageMeter('Loss', ':.4f')
    top1 = AverageMeter('Accuracy', ':6.3f')
    progress = ProgressMeter(len(train_loader), [losses, top1], prefix="Epoch: [{}]".format(epoch))
    model.train()
    
    for i, (images, targets) in enumerate(train_loader):
        targets = targets.to(device)
        images = images.to(device)
        optimizer.zero_grad()
        out, heads = model(images)

        loss = criterion_cls(out, targets) * alpha + criterion_cls(heads, targets) * beta
        acc = accuracy(out, targets)

        losses.update(loss.item(), images.size(0))
        top1.update(acc.item(), images.size(0))
        
        loss.backward()
        optimizer.step()
        
        if i % print_freq == 0:
            progress.display(i)
     
    return top1.avg, losses.avg


# 验证函数
def validate(val_loader, model, criterion_cls):
    losses = AverageMeter('Loss', ':.4f')
    top1 = AverageMeter('Accuracy', ':6.3f')
    progress = ProgressMeter(len(val_loader), [losses, top1], prefix='Test: ')
    
    model.eval()
    with torch.no_grad():
        for i, (images, targets) in enumerate(val_loader):
            targets = targets.to(device)
            images = images.to(device)
            
            out, heads = model(images)
            loss = criterion_cls(out, targets) * alpha + criterion_cls(heads, targets) * beta
            acc = accuracy(out, targets)

            losses.update(loss.item(), images.size(0))
            top1.update(acc, images.size(0))

            if i % print_freq == 0:
                progress.display(i)
        
        tqdm.write(' **** Accuracy {top1.avg:.3f} *** '.format(top1=top1))
        with open(txt_name, 'a') as f:
            f.write(' * Accuracy {top1.avg:.3f}'.format(top1=top1) + '\n')

    return top1.avg, losses.avg


# 以下函数保持不变
def accuracy(logits, labels):
    acc = (logits.argmax(dim=-1) == labels).float().mean()
    return acc * 100.0

def save_checkpoint(state, is_best):
    torch.save(state, checkpoint_path)   
    if is_best: 
        shutil.copyfile(checkpoint_path, best_checkpoint_path)

class AverageMeter(object):
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
        return fmtstr.format(** self.__dict__)

class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]       
        entries += [str(meter) for meter in self.meters]
        print_txt = '\t'.join(entries)
        tqdm.write(print_txt)
        with open(txt_name, 'a') as f:    
            f.write(print_txt + '\n')

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'

class RecorderMeter(object):
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
        title = 'Training Loss Curve'
        dpi = 80
        width, height = 1800, 800
        legend_fontsize = 10
        figsize = width / float(dpi), height / float(dpi)

        fig = plt.figure(figsize=figsize)
        x_axis = np.array([i for i in range(self.total_epoch)])
        y_axis = np.zeros(self.total_epoch)

        plt.xlim(0, self.total_epoch)
        plt.ylim(0, max(self.epoch_losses[:, 0]))
        interval_y = 5
        interval_x = 5
        plt.xticks(np.arange(0, self.total_epoch + interval_x, interval_x))
        plt.yticks(np.arange(0, max(self.epoch_losses[:, 0]) + interval_y, interval_y))
        plt.grid()
        plt.title(title, fontsize=20)
        plt.xlabel('Training Epoch', fontsize=16)
        plt.ylabel('Loss', fontsize=16)

        y_axis[:] = self.epoch_losses[:, 0]
        plt.plot(x_axis, y_axis, color='r', linestyle='-', label='Train Loss', lw=2)
        plt.legend(loc=4, fontsize=legend_fontsize)

        if save_path is not None:
            fig.savefig(save_path, dpi=dpi, bbox_inches='tight')
            print('Saved figure')
        plt.close(fig)

if __name__ == '__main__':
    main()