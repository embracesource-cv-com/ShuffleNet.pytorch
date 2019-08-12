# -*- coding: utf-8 -*-
"""
   File Name：     train.py.py
   Description :   训练
   Author :       mick.yi
   Date：          2019/8/12
"""

import os
import numpy as np
from torch import nn, optim
import torch
from config import cfg
from tqdm import tqdm
from PIL import ImageFile
import timeit
from tensorboardX import SummaryWriter
import argparse
import codecs
from utils import model_utils
from module.losses import SoftCrossEntropyLoss
import warnings

warnings.filterwarnings('ignore')

ImageFile.LOAD_TRUNCATED_IMAGES = True


def train(args):
    # parameters
    num_gpus = args.num_gpu
    batch_size = args.batch_size * num_gpus
    start_epoch = args.start_epoch
    epochs = args.epochs
    torch.backends.cudnn.benchmark = True
    if num_gpus > 1:
        # net = nn.DataParallel(net)
        os.environ["MASTER_ADDR"] = "127.0.0.1"
        os.environ["MASTER_PORT"] = "6066"
        torch.distributed.init_process_group(backend='nccl', world_size=1, rank=0, init_method='env://')
    # variables
    device = cfg.device(num_gpus)
    train_loader = cfg.train_loader(num_gpus, batch_size)
    val_loader = cfg.val_loader(batch_size)
    net = cfg.model()
    # criterion = nn.CrossEntropyLoss().to(device)
    criterion = SoftCrossEntropyLoss(label_smoothing=0.1, num_classes=cfg.NUM_CLASSES).to(device)
    optimizer = optim.SGD(model_utils.split_weights(net), lr=args.lr, momentum=0.9, weight_decay=1e-4)
    # scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[10, 20, 40], gamma=0.1)
    # scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    writer = SummaryWriter(log_dir=cfg.log_dir)

    # load weights or init weights
    if start_epoch > 0:
        weight_path = cfg.snapshot_save_path.format(start_epoch)
        print("load weight from file {}".format(weight_path))
        checkpoint = torch.load(weight_path)
        net.load_state_dict(checkpoint)
    else:
        model_utils.init_weights(net)
    net.to(device)

    if num_gpus > 1:
        # net = nn.DataParallel(net)
        net = nn.parallel.DistributedDataParallel(net)
    print("type of net:{}".format(type(net)))

    # training
    for epoch in range(start_epoch, epochs):
        running_loss, running_corrects = 0.0, 0.0
        start_time = timeit.default_timer()
        net.train()
        # scheduler.step(epoch)
        adjust_learning_rate(optimizer, epoch, args)
        _add_weight_history(writer, net, epoch)
        for images, labels in tqdm(train_loader):
            # print(type(labels), type(images))
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()  # 梯度置0
            if args.mixup:
                inputs, targets_a, targets_b, lam = model_utils.mix_up_data(images, labels, args.alpha, True)
                outputs = net(inputs)
                loss_func = model_utils.mix_up_criterion(targets_a, targets_b, lam)
                loss = loss_func(criterion, outputs)
            else:
                outputs = net.forward(images)  # [B,class_logits]
                loss = criterion(outputs, labels)
            # backward
            loss.backward()
            optimizer.step()

            _, preds = torch.max(outputs, 1)
            running_loss += loss.item() * images.size(0)
            running_corrects += torch.sum(preds == labels).item()

        train_loss = running_loss / len(train_loader.dataset)
        train_acc = 100. * running_corrects / len(train_loader.dataset)

        # 记录日志
        writer.add_scalar('scalar/learning_rate', optimizer.param_groups[0]['lr'], epoch + 1)
        writer.add_scalar('scalar/train_loss', train_loss, epoch + 1)
        writer.add_scalar('scalar/train_acc', train_acc, epoch + 1)
        # 打印状态信息
        print("[{}] Epoch: {}/{} Loss: {:03f} Acc: {:03f}".format('train',
                                                                  epoch + 1,
                                                                  epochs,
                                                                  train_loss,
                                                                  train_acc))
        stop_time = timeit.default_timer()
        print("Execution time: " + str(stop_time - start_time) + "\n")
        # 验证集
        acc = val(net, val_loader, device)
        writer.add_scalar('scalar/val_acc', acc, epoch + 1)
        print('Epoch: {}/{}  Val Acc：{:03f}'.format(epoch + 1, epochs, acc))

        # 保存中间模型
        if (epoch + 1) % cfg.SNAPSHOT == 0:
            # torch.save(net.state_dict(), cfg.snapshot_save_path.format(epoch + 1))
            model_utils.save_model(net, cfg.snapshot_save_path.format(epoch + 1))
    # 保存最终模型
    # torch.save(net.state_dict(), cfg.save_path)
    model_utils.save_model(net, cfg.save_path)


def inference(args):
    # parameters
    weight_path = args.weight_path
    num_gpus = args.num_gpu
    batch_size = args.batch_size * num_gpus
    # variables
    device = cfg.device(num_gpus)
    net = cfg.model().to(device)
    test_loader = cfg.test_loader(batch_size=batch_size)
    # load weights
    checkpoint = torch.load(weight_path)
    net.load_state_dict(checkpoint)

    net.eval()
    with codecs.open('/tmp/classification.txt', mode='w', encoding='utf-8') as f:
        for images, img_names in tqdm(test_loader):
            images = images.to(device)
            with torch.no_grad():
                outputs = net.forward(images)
            _, class_ids = torch.max(outputs, dim=1)  # batch_size is 1
            # print("class_ids.cpu().numpy():{}".format(class_ids.cpu().numpy()))
            for name, class_id in zip(img_names, class_ids.cpu().numpy()):
                f.write('{} {}\n'.format(name, class_id))


def val(net, val_loader, device):
    correct = 0
    total = 0
    net.eval()
    for images, labels in tqdm(val_loader):
        images, labels = images.to(device), labels.to(device)
        with torch.no_grad():
            outputs = net(images)
        # 取得分最高的那个类 (outputs.data的索引号)
        _, class_ids = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (class_ids == labels).sum().item()
    acc = 100. * correct / total
    return acc


def evaluate(args):
    # parameters
    weight_path = args.weight_path
    num_gpus = args.num_gpu
    batch_size = args.batch_size * num_gpus
    # variables
    device = cfg.device(num_gpus)
    net = cfg.model().to(device)
    val_loader = cfg.val_loader(batch_size)
    # load weights
    checkpoint = torch.load(weight_path)
    net.load_state_dict(checkpoint)
    # eval
    acc = val(net, val_loader, device)
    print("acc:{}".format(acc))


def _add_weight_history(writer, net, epoch):
    for name, param in net.named_parameters():
        writer.add_histogram(name, param.clone().cpu().data.numpy(), epoch)


def adjust_learning_rate(optimizer, epoch, args):
    global state

    def adjust_optimizer():
        for param_group in optimizer.param_groups:
            param_group['lr'] = state['lr']

    if epoch < args.warmup:
        state['lr'] = args.lr * (epoch + 1) / args.warmup

    elif args.cos:  # cosine decay lr schedule (Note: epoch-wise, not batch-wise)
        state['lr'] = args.lr * 0.5 * (1 + np.cos(np.pi * epoch / args.epochs))

    elif epoch in args.schedule:  # step lr schedule
        state['lr'] *= args.gamma

    adjust_optimizer()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='遥感分类竞赛')
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'test', 'eval'],
                        help='mode: train or test')
    parser.add_argument('--num-gpu', type=int, default=1, help='number of gpus used for training')
    parser.add_argument('--epochs', type=int, default=cfg.EPOCHS, help='number of training epochs')
    parser.add_argument('--lr', type=float, default=5e-2, help='learning rate')
    parser.add_argument('--batch-size', type=int, default=cfg.BATCH_SIZE, help='batch size')
    parser.add_argument('--start-epoch', type=int, default=0, help='start epoch')
    parser.add_argument('--weight-path', type=str, default=cfg.save_path, help='weight path in the test stage')
    parser.add_argument('--mixup', action='store_true', help='whether to use mixup')
    parser.add_argument('--alpha', default=1., type=float, metavar='mixup alpha',
                        help='alpha value for mixup B(alpha, alpha) distribution')
    parser.add_argument('--warmup', '--wp', default=5, type=int,
                        help='number of epochs to warmup')
    parser.add_argument('--cos', dest='cos', action='store_true',
                        help='using cosine decay lr schedule')
    parser.add_argument('--schedule', type=int, nargs='+', default=[30, 60, 90],
                        help='Decrease learning rate at these epochs.')
    parser.add_argument('--gamma', type=float, default=0.1, help='LR is multiplied by gamma on schedule.')
    parser.add_argument('--opt-level', default='O2', type=str,
                        help='O2 is mixed FP16/32 training, see more in https://github.com/NVIDIA/apex/tree/f5cd5ae937f168c763985f627bbf850648ea5f3f/examples/imagenet')
    parser.add_argument('--keep-batchnorm-fp32', default=True, action='store_true',
                        help='keeping cudnn bn leads to fast training')
    parser.add_argument('--loss-scale', type=float, default=None)
    arguments = parser.parse_args()
    state = {k: v for k, v in arguments._get_kwargs()}

    if 'train' == arguments.mode:
        train(arguments)
    elif 'eval' == arguments.mode:
        evaluate(arguments)
    else:
        inference(arguments)