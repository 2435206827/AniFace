"""
@author: ronghuaiyang
"""

import os
import numpy as np
import time
import torch
import torchvision
import yaml
from torch.nn import DataParallel
from torch.optim.lr_scheduler import StepLR
from torch.utils import data
from model.focal_loss import *
from model.metrics import *
from model.resnet import *
from data.dataset import Dataset

def save_model(model, save_path, name, iter_cnt):
    save_name = os.path.join(save_path, name + '_' + str(iter_cnt) + '.pth')
    torch.save(model.state_dict(), save_name)
    print("model was saved.")
    return save_name

if __name__ == "__main__":

    with open('config.yml') as config_file:
        opt = yaml.load(config_file)

    device = torch.device("cuda")

    train_dataset = Dataset(opt["train_root"], opt["train_list"])
    trainloader = data.DataLoader(train_dataset, batch_size = opt["train_batch_size"], shuffle = True, num_workers = opt["num_workers"])

    print('{} train iters per epoch:'.format(len(trainloader)))

    if opt["loss"] == 'focal_loss':
        criterion = FocalLoss(gamma = 2)
    else:
        criterion = torch.nn.CrossEntropyLoss()

    if opt["download"]:
        if opt["backbone"] == 'resnet18':
            model = torchvision.models.resnet18(pretrained = True)
        elif opt["backbone"] == 'resnet34':
            model = torchvision.models.resnet34(pretrained = True)
        elif opt["backbone"] == 'resnet50':
            model = torchvision.models.resnet50(pretrained = True)

        model.fc = nn.Linear(model.fc.in_features, 512)
        nn.init.xavier_uniform_(model.fc.weight)
    else:
        pass

    if opt["metric"] == 'add_margin':
        metric_fc = AddMarginProduct(512, opt["num_classes"], s = 30, m = 0.35)
    elif opt["metric"] == 'arc_margin':
        metric_fc = ArcMarginProduct(512, opt["num_classes"], s = 30, m = 0.5)
    elif opt["metric"] == 'sphere':
        metric_fc = SphereProduct(512, opt["num_classes"], m = 4)
    else:
        metric_fc = nn.Linear(512, opt["num_classes"])

    print(model)
    model.to(device)
    model = DataParallel(model)
    metric_fc.to(device)
    metric_fc = DataParallel(metric_fc)

    if opt["optimizer"] == 'sgd':
        optimizer = torch.optim.SGD([
            {'params': model.parameters()}, 
            {'params': metric_fc.parameters()}
        ], lr = opt["lr"], weight_decay = opt["weight_decay"])
    else:
        optimizer = torch.optim.Adam([
            {'params': model.parameters()}, 
            {'params': metric_fc.parameters()}
        ], lr = opt["lr"], weight_decay = opt["weight_decay"])
    scheduler = StepLR(optimizer, step_size = opt["lr_step"], gamma = 0.1)

    start = time.time()
    for i in range(opt["max_epoch"]):
        scheduler.step()

        model.train()
        for ii, data in enumerate(trainloader):
            data_input, label = data
            data_input = data_input.to(device)
            label = label.to(device).long()
            feature = model(data_input)
            output = metric_fc(feature, label)
            loss = criterion(output, label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            iters = i * len(trainloader) + ii

            if iters % opt["print_freq"] == 0:
                output = output.data.cpu().numpy()
                output = np.argmax(output, axis=1)
                label = label.data.cpu().numpy()
                acc = np.mean((output == label).astype(int))
                speed = opt["print_freq"] / (time.time() - start)
                time_str = time.asctime(time.localtime(time.time()))
                print('{} train epoch {} iter {} {} iters/s loss {} acc {}'.format(time_str, i, ii, speed, loss.item(), acc))

                start = time.time()

        if i % opt["save_interval"] == 0 or i == opt["max_epoch"]:
            save_model(model, opt["checkpoints_path"], opt["backbone"], i)
