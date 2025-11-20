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
from network.models import *
from tqdm import tqdm
from torchsummary import summary
from thop import profile

plt.switch_backend('agg')
now = datetime.datetime.now()
time_str = now.strftime("[%m-%d]-[%H-%M]-")


dataset_name='rafdb'

data_path               = './dataset/'+dataset_name+'/test'
model_name              = dataset_name+'_'
checkpoint_path         = './checkpoints/' + model_name+time_str +'.pth'
best_checkpoint_path    = './checkpoints/'+model_name+time_str+'_best.pth'
txt_name                = './logs/' + model_name +time_str+  '.txt'
curve_name              = './logs/' + model_name +time_str+  '.png'
model_path              = './experiment/'+dataset_name+'/'+dataset_name+'.pth'
device          = 'cuda:0'
eval            = True
lr              = 0.01
momentum        = 0.9
weight_decay    = 1e-4
epochs          = 100
ls              = 15
batch_size      = 1
workers         = 4
print_freq      = 10
pretrained      =True

def main():

    print('Training time: ' + now.strftime("%m-%d %H:%M"))
    print('device:    ' + device)
    print('dataset:    ' + dataset_name)
    
    model = PCNN(num_class=7,device=device)

    model=model.to(device)

    criterion_cls = nn.CrossEntropyLoss().to(device)

    cudnn.benchmark=True        #加速网络
  #  if pretrained:

   #     checkpoint = torch.load(model_path,map_location=device)
     #   pre_trained_dict = checkpoint['state_dict']                                                          
     #   model.load_state_dict({k.replace('module.',''):v for k,v in pre_trained_dict.items()})
    if pretrained:
        tqdm.write('load pretrained model......')
        checkpoint = torch.load(model_path,map_location=torch.device('cpu'))
        pretrained_state_dict = checkpoint['state_dict']
        model_state_dict = model.state_dict()
        
        for key in pretrained_state_dict:
           if key in model_state_dict:
               model_state_dict[key] = pretrained_state_dict[key]
        
        model.load_state_dict(model_state_dict)
        tqdm.write('load succesful!')

    if not pretrained:
        tqdm.write('load pretrained model......')
        checkpoint = torch.load(model_path,map_location=torch.device('cuda'))
        pretrained_state_dict = checkpoint['state_dict']
        model_state_dict = model.state_dict()

        # for key in pretrained_state_dict:
        #     model_state_dict[key] = pretrained_state_dict[key]
        print(pretrained_state_dict)
        for key in model_state_dict:
            print(key)
            if key[:7]=='resnet2':
                print(key)
            else:
                # model_state_dict[key] = pretrained_state_dict['module.'+key]
                model_state_dict[key] = pretrained_state_dict[key]

        model.load_state_dict(model_state_dict)
        tqdm.write('load succesful!')
        state={'epoch': checkpoint['epoch'],
                         'state_dict': model.state_dict(),
                         'best_acc': checkpoint['best_acc'],
                         'optimizer': checkpoint['optimizer'],
                         'recorder': checkpoint['recorder'],}
        torch.save(state,  's.pth')
    val_dataset=datasets.ImageFolder(
        data_path,
        transforms.Compose([
            transforms.Resize((224,224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485,0.456,0.406],
                std=[0.229,0.224,0.225])
        ])
    )


    val_loader=torch.utils.data.DataLoader(
        val_dataset,
        batch_size= batch_size,
        shuffle=False,
        num_workers= workers,
        pin_memory=True
    )
 
    if  eval:
        validate(val_loader, model, criterion_cls)
        return

    # 验证
def validate(val_loader,model,criterion_cls):
    losses = AverageMeter('Loss', ':.4f')
    top1 = AverageMeter('Accuracy', ':6.3f')
    progress = ProgressMeter(len(val_loader),
                             [losses, top1],
                             prefix='Test: ')
    model.eval()

    labels_name=['Neutral', 'Happiness', 'Sadness', 'Surprise', 'Fear', 'Disgust', 'Anger']
    if dataset_name=='affectnet-8' or dataset_name=='ferplus':
        labels_name=['Neutral', 'Happiness', 'Sadness', 'Surprise', 'Fear', 'Disgust', 'Anger','contempt']
    
    # 初始化预测和真实标签列表
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for i,(images,targets) in enumerate(val_loader):

            targets=targets.to(device)
            images=images.to(device)

            out,heads=model(images)
            loss = (criterion_cls(out, targets) + criterion_cls(heads, targets)) * 10 

            predict_np=np.argmax(out.cpu().detach().numpy(),axis=-1)
            labels_np=targets.cpu().numpy()
    
            acc=accuracy(out,targets)
            losses.update(loss.item(),images.size(0))       #传入一个batch平均损失，batch_size，计算平均值
            top1.update(acc,images.size(0))        #传入一个batch平均精度，batch_size，计算平均值
            
            # 收集所有预测和真实标签
            all_preds.extend(predict_np)
            all_targets.extend(labels_np)

            if i %  print_freq == 0:
                progress.display(i)
        tqdm.write(' **** Accuracy {top1.avg:.3f} *** '.format(top1=top1))
        
        # 计算混淆矩阵
        cm = confusion_matrix(all_targets, all_preds)
        
        # 绘制混淆矩阵
        plt.figure(figsize=(10, 8))
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels_name)
        disp.plot(include_values=True, cmap='Blues', ax=None, xticks_rotation='vertical')
        plt.title('Confusion Matrix')
        plt.tight_layout()
        
        # 保存混淆矩阵图像
        cm_path = './logs/' + model_name + time_str + '_confusion_matrix.png'
        plt.savefig(cm_path)
        tqdm.write(f'Confusion matrix saved to {cm_path}')
        
        # 打印分类准确率
        class_accuracy = cm.diagonal() / cm.sum(axis=1)
        for i, class_name in enumerate(labels_name):
            tqdm.write(f'Accuracy for {class_name}: {class_accuracy[i]:.4f}')

    return top1.avg,losses.avg
            

    #计算均值
class AverageMeter(object):     
    """Computes and stores the average and current value"""
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

    #显示进度
class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)     #获取格式
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]       #
        entries += [str(meter) for meter in self.meters]
        print_txt = '\t'.join(entries)
        tqdm.write(print_txt)

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'

    #返回一个batch精度
def accuracy(logits,labels):
    acc=(logits.argmax(dim=-1)==labels).float().mean()
    return acc*100.0

    #绘图
class RecorderMeter(object):
    """Computes and stores the minimum loss value and its epoch index"""

    def __init__(self, total_epoch):
        self.reset(total_epoch)

    def reset(self, total_epoch):
        self.total_epoch = total_epoch
        self.current_epoch = 0
        self.epoch_losses = np.zeros((self.total_epoch, 2), dtype=np.float32)    # [epoch, train/val]
        self.epoch_accuracy = np.zeros((self.total_epoch, 2), dtype=np.float32)  # [epoch, train/val]

    def update(self, idx, train_loss, train_acc, val_loss, val_acc):
        self.epoch_losses[idx, 0] = train_loss * 30
        self.epoch_losses[idx, 1] = val_loss * 30
        self.epoch_accuracy[idx, 0] = train_acc
        self.epoch_accuracy[idx, 1] = val_acc
        self.current_epoch = idx + 1

    def plot_curve(self, save_path):

        title = 'the accuracy/loss curve of train/val'
        dpi = 80
        width, height = 1800, 800
        legend_fontsize = 10
        figsize = width / float(dpi), height / float(dpi)

        fig = plt.figure(figsize=figsize)
        x_axis = np.array([i for i in range(self.total_epoch)])  # epochs
        y_axis = np.zeros(self.total_epoch)

        plt.xlim(0, self.total_epoch)
        plt.ylim(0, 100)
        interval_y = 5
        interval_x = 5
        plt.xticks(np.arange(0, self.total_epoch + interval_x, interval_x))
        plt.yticks(np.arange(0, 100 + interval_y, interval_y))
        plt.grid()
        plt.title(title, fontsize=20)
        plt.xlabel('the training epoch', fontsize=16)
        plt.ylabel('accuracy', fontsize=16)

        y_axis[:] = self.epoch_accuracy[:, 0]
        plt.plot(x_axis, y_axis, color='g', linestyle='-', label='train-accuracy', lw=2)
        plt.legend(loc=4, fontsize=legend_fontsize)

        y_axis[:] = self.epoch_accuracy[:, 1]
        plt.plot(x_axis, y_axis, color='y', linestyle='-', label='valid-accuracy', lw=2)
        plt.legend(loc=4, fontsize=legend_fontsize)

        y_axis[:] = self.epoch_losses[:, 0]
        plt.plot(x_axis, y_axis, color='g', linestyle=':', label='train-loss-x30', lw=2)
        plt.legend(loc=4, fontsize=legend_fontsize)

        y_axis[:] = self.epoch_losses[:, 1]
        plt.plot(x_axis, y_axis, color='y', linestyle=':', label='valid-loss-x30', lw=2)
        plt.legend(loc=4, fontsize=legend_fontsize)

        if save_path is not None:
            fig.savefig(save_path, dpi=dpi, bbox_inches='tight')
            tqdm.write('Saved figure')
        plt.close(fig)

if __name__ == '__main__':
    main()    