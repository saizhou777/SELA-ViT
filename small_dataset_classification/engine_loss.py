# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
# ------------------------------------------
# Modification:
# Added code for adjusting keep rate -- Youwei Liang
"""
Train and eval functions used in main.py
"""
import math
import sys
from typing import Iterable, Optional
import torch
from timm.data import Mixup
from timm.utils import accuracy, ModelEma
from losses import DistillationLoss
import utils
from helpers import adjust_keep_rate

def train_one_epoch(model: torch.nn.Module, criterion: DistillationLoss,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler, max_norm: float = 0,
                    model_ema: Optional[ModelEma] = None, mixup_fn: Optional[Mixup] = None,
                    writer=None,
                    set_training_mode=True,
                    args=None):

    
    model.train(set_training_mode)
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    
    # p.s. better to use 20 for cifar and 200 for imagenet
    print_freq = 100      # 20
    log_interval = 10    # 10
    it = epoch * len(data_loader)
    ITERS_PER_EPOCH = len(data_loader)

    for samples, targets in metric_logger.log_every(data_loader, print_freq, header):
        samples = samples.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        if mixup_fn is not None:
            samples, targets = mixup_fn(samples, targets)
        
        with torch.cuda.amp.autocast():
            outputs, attn_similarity_loss = model(samples)
            loss = criterion(samples, outputs, targets) + attn_similarity_loss * 0.009
            #outputs= model(samples)
            #loss = criterion(samples, outputs, targets)
        

        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        optimizer.zero_grad()

        # this attribute is added by timm on one optimizer (adahessian)
        is_second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order

        loss_scaler(loss, optimizer, clip_grad=max_norm,
                    parameters=model.parameters(), create_graph=is_second_order)

        torch.cuda.synchronize()
        if model_ema is not None:
            model_ema.update(model)

        metric_logger.update(loss=loss_value)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

        if torch.distributed.get_rank() == 0 and it % log_interval == 0:
            writer.add_scalar('loss', loss_value, it)
            writer.add_scalar('lr', optimizer.param_groups[0]["lr"], it)
        it += 1
        '''is_main_process = not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0

        if is_main_process and it % log_interval == 0:
            writer.add_scalar('loss', loss_value, it)
            writer.add_scalar('lr', optimizer.param_groups[0]["lr"], it)'''
        

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate(data_loader, model, device, keep_rate=None):
    criterion = torch.nn.CrossEntropyLoss()

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'

    # switch to evaluation mode
    model.eval()

    for images, target in metric_logger.log_every(data_loader, 50, header):
        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        # compute output
        with torch.cuda.amp.autocast():
            #output = model(images)
            #loss = criterion(output, target)
            output, attn_similarity_loss = model(images)
            loss = criterion(output, target) + attn_similarity_loss * 0.009

        acc1, acc5 = accuracy(output, target, topk=(1, 5))

        batch_size = images.shape[0]
        metric_logger.update(loss=loss.item())
        metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)
        metric_logger.meters['acc5'].update(acc5.item(), n=batch_size)
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print('* Acc@1 {top1.global_avg:.3f} Acc@5 {top5.global_avg:.3f} loss {losses.global_avg:.3f}'
          .format(top1=metric_logger.acc1, top5=metric_logger.acc5, losses=metric_logger.loss))

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def get_acc(data_loader, model, device, keep_rate=None, tokens=None):
    criterion = torch.nn.CrossEntropyLoss()

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'

    # switch to evaluation mode
    model.eval()

    for images, target in metric_logger.log_every(data_loader, 50, header):
        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        # compute output
        with torch.cuda.amp.autocast():
            #output = model(images, keep_rate, tokens)
            #loss = criterion(output, target)
            output, attn_similarity_loss = model(images, keep_rate, tokens)
            loss = criterion(output, target) + attn_similarity_loss * 0.009

        acc1, acc5 = accuracy(output, target, topk=(1, 5))

        batch_size = images.shape[0]
        metric_logger.update(loss=loss.item())
        metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)
        metric_logger.meters['acc5'].update(acc5.item(), n=batch_size)
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print('* Acc@1 {top1.global_avg:.3f} Acc@5 {top5.global_avg:.3f} loss {losses.global_avg:.3f}'
          .format(top1=metric_logger.acc1, top5=metric_logger.acc5, losses=metric_logger.loss))

    return metric_logger.acc1.global_avg
