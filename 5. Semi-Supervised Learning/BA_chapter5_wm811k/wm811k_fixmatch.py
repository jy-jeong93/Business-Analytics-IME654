from __future__ import print_function

import argparse
import os
import shutil
from tabnanny import check
import time
import random
import numpy as np

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data as data
import torchvision.transforms as transforms
import torch.nn.functional as F

import models.wrn as models
from utils import Logger
from scipy import optimize
import torchmetrics

from dataset.wm811k import WM811k_valid, WM811k, WM811k_unlabel
from dataset.transforms import WM811KTransform

parser = argparse.ArgumentParser(description='PyTorch FixMatch Training')
# Optimization options
parser.add_argument('--epochs', default=200, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts')
parser.add_argument('--batch-size', default=64, type=int, metavar='N',
                    help='train batchsize')
parser.add_argument('--lr', '--learning-rate', default=0.002, type=float,
                    metavar='LR', help='initial learning rate')

# Checkpoints
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--out', default='result',
                    help='Directory to output the result')

# Training options
parser.add_argument('--manualSeed', type=int, default=0, help='manual seed')
parser.add_argument('--gpu', default='0', type=str,
                    help='id(s) for CUDA_VISIBLE_DEVICES')

parser.add_argument('--val-iteration', type=int, default=500,
                        help='Frequency for the evaluation')
parser.add_argument('--input_size', type=int, default=96,
                        help='Dataset_input_size')
parser.add_argument('--weak-aug', type=str, default='rotate', choices=['crop', 'rotate', 'shift', 'cutout'],
                    help='fixmatch_weak_augmentation')
parser.add_argument('--strong-aug', type=str, default='cutout+noise', choices=['cutout+crop','cutout+noise','cutout+rotate','cutout+shift'],
                    help='fixmatch_strong_augmentation')

parser.add_argument('--tau', default=0.95, type=float, help='hyper-parameter for pseudo-label of FixMatch')
parser.add_argument('--ema-decay', default=0.999, type=float)

parser.add_argument('--train-proportion', default=1.0, type=float, help='Proportion of Training labeled data')

args = parser.parse_args()
state = {k: v for k, v in args._get_kwargs()}

# Use CUDA
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
use_cuda = torch.cuda.is_available()

# Random seed
if args.manualSeed is None:
    args.manualSeed = random.randint(1, 10000)
np.random.seed(args.manualSeed)


best_macro_f1 = 0 # best valid macro_f1 score
num_class = 9


def main():
    global best_acc
    global best_macro_f1
    
    if not os.path.isdir(args.out):
        os.makedirs(args.out)
        
        
    vaild_data_kwargs = {
        'tranform' : WM811KTransform(size=args.input_size, mode = 'test')
    }
    valid_dataset = WM811k_valid('./data/wm811k/labeled/valid', **vaild_data_kwargs)
    
    train_data_kwargs = {
        'transform' : WM811KTransform(size=args.input_size, mode='rotate'),
        'proportion' : args.train_proportion,
        'seed' : args.manualSeed
    }
    train_dataset = WM811k('./data/wm811k/labeled/train/', **train_data_kwargs)

    train_unlabeled_data_kwargs = {
        'transform' : WM811KTransform(size=args.input_size, mode=args.weak_aug),
        'transform2' : WM811KTransform(size=args.input_size, mode=args.strong_aug)
    }
    train_unlabeled_dataset = WM811k_unlabel('./data/wm811k/unlabeled/', **train_unlabeled_data_kwargs)
    
    labeled_trainloader = data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0, drop_last=True)
    unlabeled_trainloader = data.DataLoader(train_unlabeled_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0, drop_last=True)
    valid_loader = data.DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

    # Model
    def create_model(ema=False):
        model = models.WRN(2, num_class)
        if use_cuda:
            model = model.cuda()
        if ema:
            for param in model.parameters():
                param.detach()
        return model
    
    model = create_model()
    ema_model = create_model(ema=True)
    
    if use_cuda:
        cudnn.benchmark = True
    print('Total params: %2.fM' % (sum(p.numel() for p in model.parameters())/1000000.0))
    
    train_criterion = SemiLoss()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    ema_optimizer = WeightEMA(model, ema_model, alpha=args.ema_decay)
    
    # Resume
    title = 'WBM811k'
    if args.resume:
        # Load checkpoint
        print('==> Resuming from checkpoint..')
        assert os.path.isfile(args.resume), 'Error: no checkpoint directory found!'
        args.out = os.path.dirname(args.resume)
        checkpoint = torch.load(args.resume)
        start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])
        ema_model.load_state_dict(checkpoint['ema_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        logger = Logger(os.path.join(args.out, 'log.txt'), title=title, resume=True)
    else:
        logger = Logger(os.path.join(args.out, 'log.txt'), title=title)
        logger.set_names(['Train Loss', 'Train Loss X', 'Train Loss U', 'Valid Loss', 'Valid Acc', 'Macro_F1'])
        start_epoch = 0
        
    valid_f1s = []
    
    tmp_class_list = torch.Tensor(list([i for i in range(1,10)])).cuda()
    desi_p_o = torch.ones(tmp_class_list.shape).cuda()
    desi_p = desi_p_o
    
    # Main func.
    for epoch in range(start_epoch, args.epochs):
        print('\nEpoch: [%d | %d] LR: %f' % (epoch + 1, args.epochs, state['lr']))
        
        train_loss, train_loss_x, train_loss_u = train(labeled_trainloader,
                                                       unlabeled_trainloader,
                                                       model, optimizer,
                                                       ema_optimizer, train_criterion,
                                                       epoch, use_cuda, desi_p)
        
        valid_classwise_precision, valid_classwise_recall, valid_classwise_f1 = validate(valid_loader,
                                                                                            ema_model, 
                                                                                            criterion,
                                                                                            use_cuda, 
                                                                                            mode='Test Stats ')
        valid_macro_f1 = valid_classwise_f1.mean()
        print('valid macro f1: ', valid_macro_f1)
        logger.append([train_loss, train_loss_x, train_loss_u, valid_macro_f1])
        
        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'ema_state_dict': ema_model.state_dict(),
            'optimizer': optimizer.state_dict(),
        }, epoch + 1, is_best=(valid_macro_f1 > best_macro_f1))
        
            
        if valid_macro_f1 > best_macro_f1:
            best_macro_f1 = valid_macro_f1
            
        valid_f1s.append(valid_macro_f1)

    logger.close()
    
    print('Best F1_score')
    print(best_macro_f1)
    

def train(labeled_trainloader, unlabeled_trainloader, model, optimizer, ema_optimizer, criterion, epoch, use_cuda, desi_p):
    
    labeled_train_iter = iter(labeled_trainloader)
    unlabeled_train_iter = iter(unlabeled_trainloader)

    model.train()
    for batch_idx in range(args.val_iteration):
        try:
            inputs_x, targets_x, _ = labeled_train_iter.next()
        except:
            labeled_train_iter = iter(labeled_trainloader)
            inputs_x, targets_x, _ = labeled_train_iter.next()

        try:
            (inputs_u, inputs_u2), _, idx_u = unlabeled_train_iter.next()
        except:
            unlabeled_train_iter = iter(unlabeled_trainloader)
            (inputs_u, inputs_u2), _, idx_u = unlabeled_train_iter.next()
            
            
        batch_size = inputs_x.size(0)
        
        targets_x = torch.zeros(batch_size, num_class).scatter_(1, targets_x.type(torch.int64).view(-1, 1), 1)
        if use_cuda:
            inputs_x, targets_x = inputs_x.cuda(), targets_x.cuda(non_blocking=True)
            inputs_u, inputs_u2 = inputs_u.cuda(), inputs_u2.cuda()
        
        with torch.no_grad():
                        
            outputs_u, _ = model(inputs_u)
            targets_u = torch.softmax(outputs_u, dim=1)
            
        max_p, p_hat = torch.max(targets_u, dim=1)

        select_mask1 = max_p.ge(args.tau)
        select_mask2 = torch.rand(batch_size).cuda() < desi_p[p_hat]
        select_mask = select_mask1 * select_mask2
        select_mask = select_mask.float()
        p_hat = torch.zeros(batch_size, num_class).cuda().scatter_(1, p_hat.view(-1, 1), 1)
        all_inputs = torch.cat([inputs_x, inputs_u2], dim=0)
        all_targets = torch.cat([targets_x, p_hat], dim=0)

        all_outputs, _ = model(all_inputs)
        logits_x = all_outputs[:batch_size]
        logits_u = all_outputs[batch_size:]

        Lx, Lu = criterion(logits_x, all_targets[:batch_size], logits_u, all_targets[batch_size:], select_mask)
        loss = Lx + Lu

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        ema_optimizer.step()


def validate(valloader, model, criterion, use_cuda, mode):
    
    # switch to evaluate mode
    model.eval()
    
    classwise_TP = torch.zeros(num_class)
    classwise_FP = torch.zeros(num_class)
    classwise_correct = torch.zeros(num_class)
    classwise_num = torch.zeros(num_class)
    section_acc = torch.zeros(3)
    if use_cuda:
        classwise_TP = torch.zeros(num_class).cuda()
        classwise_FP = torch.zeros(num_class).cuda()
        classwise_correct = classwise_correct.cuda()
        classwise_num = classwise_num.cuda()
        section_acc = section_acc.cuda()

    with torch.no_grad():
        for batch_idx, (inputs, targets, _) in enumerate(valloader):


            if use_cuda:
                inputs, targets = inputs.cuda(), targets.cuda(non_blocking=True)
            outputs, _ = model(inputs)
            loss = criterion(outputs, targets)

            # classwise prediction
            pred_label = outputs.max(1)[1]
            assert pred_label.shape == targets.shape
            confusion_matrix = torch.zeros(num_class, num_class).cuda()
            for r, c in zip(targets, pred_label):
                confusion_matrix[r][c] += 1
                
            classwise_num += confusion_matrix.sum(dim=1)
            classwise_TP += torch.diagonal(confusion_matrix, 0)
            diag_mask = torch.eye(num_class).cuda()
            # 순서 주의 : confusion_matrix가 masked_fiil_(diag_mask.bool(), 0)을 통해서 대각원소 0으로 교체돼서 저장됨
            classwise_FP += confusion_matrix.masked_fill_(diag_mask.bool(), 0).sum(dim=0)
            

    print('classwise_num')
    print(classwise_num)  
    print('classwise_FP')
    print(classwise_FP)
    print('classwise_TP')
    print(classwise_TP)
    classwise_precision = (classwise_TP / (classwise_TP + classwise_FP))
    classwise_recall = (classwise_TP / classwise_num)
    classwise_f1 = torch.nan_to_num((2*classwise_precision*classwise_recall) / (classwise_precision + classwise_recall))
    if use_cuda:
        classwise_precision = classwise_precision.cpu()
        classwise_recall = classwise_recall.cpu()
        classwise_f1 = classwise_f1.cpu()

    return (classwise_precision.numpy(), classwise_recall.numpy(), classwise_f1.numpy())


def save_checkpoint(state, epoch, checkpoint=args.out, filename='checkpoint.pth.tar', is_best=False):
    filepath = os.path.join(checkpoint, filename)
    torch.save(state, filepath)

    if epoch % 100 == 0:
        shutil.copyfile(filepath, os.path.join(checkpoint, 'model_' + str(epoch) + '.pth.tar'))
    if is_best:
        shutil.copyfile(filepath, os.path.join(checkpoint, 'best_model.pth.tar'))

class SemiLoss(object):
    def __call__(self, outputs_x, targets_x, outputs_u, targets_u, mask):
        Lx = -torch.mean(torch.sum(F.log_softmax(outputs_x, dim=1) * targets_x, dim=1))
        Lu = -torch.mean(torch.sum(F.log_softmax(outputs_u, dim=1) * targets_u, dim=1) * mask)

        return Lx, Lu

class WeightEMA(object):
    def __init__(self, model, ema_model, alpha=0.999):
        self.model = model
        self.ema_model = ema_model
        self.alpha = alpha
        self.params = list(model.state_dict().values())
        self.ema_params = list(ema_model.state_dict().values())
        self.wd = 0.02 * args.lr

        for param, ema_param in zip(self.params, self.ema_params):
            param.data.copy_(ema_param.data)

    def step(self):
        one_minus_alpha = 1.0 - self.alpha
        for param, ema_param in zip(self.params, self.ema_params):
            ema_param = ema_param.float()
            param = param.float()
            ema_param.mul_(self.alpha)
            ema_param.add_(param * one_minus_alpha)
            # customized weight decay
            param.mul_(1 - self.wd)

if __name__ == '__main__':
    main()