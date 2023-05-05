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
    save_name = os.path.join(save_path, "{}_{}.pth".format(name, iter_cnt))
    torch.save(model.state_dict(), save_name)
    return save_name

def list_images(directory):
    ls = []
    subdirs = [d for d in os.listdir(directory) if os.path.isdir(os.path.join(directory, d))]
    for i, subdir in enumerate(subdirs):
        subdir_path = os.path.join(directory, subdir)
        for file in os.listdir(subdir_path):
            if file.endswith(".png") or file.endswith(".jpg"):
                ls.append(str(os.path.join(subdir, file)) + " " + str(i + 1))
    return ls

def get_classes_num(directory):
    subdirs = [d for d in os.listdir(directory) if os.path.isdir(os.path.join(directory, d))]
    return len(subdirs) + 1

if __name__ == "__main__":
    # config
    try:
        with open("config.yml") as config_file:
            opt = yaml.load(config_file, Loader = yaml.FullLoader)
    except Exception as e:
        print(e)
        assert False, "Fail to read config file"

    # load config & prepare dataset
    device = torch.device("cuda")
    root = opt["train_root"]

    train_dataset = Dataset(root, list_images(root))
    trainloader = data.DataLoader(train_dataset, batch_size = opt["train_batch_size"], shuffle = True, num_workers = opt["num_workers"])

    print("{} train iters per epoch:".format(len(trainloader)))

    # loss
    if opt["loss"] == "focal_loss":
        criterion = FocalLoss(gamma = 2)
    elif opt["loss"] == "cross_entropy_loss":
        criterion = torch.nn.CrossEntropyLoss()
    else:
        assert False, "Invaild loss function: {}".format(opt["loss"])

    # backbone
    if opt["backbone"] == "resnet18":
        model = torchvision.models.resnet18(pretrained = opt["pretrained"])
    elif opt["backbone"] == "resnet34":
        model = torchvision.models.resnet34(pretrained = opt["pretrained"])
    elif opt["backbone"] == "resnet50":
        model = torchvision.models.resnet50(pretrained = opt["pretrained"])
    else:
        assert False, "Invaild model: {}".format(opt["backbone"])
    model.fc = nn.Linear(model.fc.in_features, 512)
    nn.init.xavier_uniform_(model.fc.weight)
    
    num_classes = get_classes_num(root)

    # metric
    if opt["metric"] == "add_margin":
        metric_fc = AddMarginProduct(512, num_classes, s = 30, m = 0.35)
    elif opt["metric"] == "arc_margin":
        metric_fc = ArcMarginProduct(512, num_classes, s = 30, m = 0.5)
    elif opt["metric"] == "sphere":
        metric_fc = SphereProduct(512, num_classes, m = 4)
    else:
        metric_fc = nn.Linear(512, num_classes)

    # optimizer & scheduler
    if opt["optimizer"] == "sgd":
        optimizer = torch.optim.SGD([
            {"params": model.parameters()}, 
            {"params": metric_fc.parameters()}
        ], lr = opt["lr"], weight_decay = opt["weight_decay"])
    elif opt["optimizer"] == "Adam":
        optimizer = torch.optim.Adam([
            {"params": model.parameters()}, 
            {"params": metric_fc.parameters()}
        ], lr = opt["lr"], weight_decay = opt["weight_decay"])
    else:
        assert False, "illegal optimizer"
    scheduler = StepLR(optimizer, step_size = opt["lr_step"], gamma = 0.1)

    # start to train
    print(model)
    model.to(device)
    model = DataParallel(model)
    metric_fc.to(device)
    metric_fc = DataParallel(metric_fc)

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
                output = np.argmax(output.data.cpu().numpy(), axis = 1)
                label = label.data.cpu().numpy()
                acc = np.mean((output == label).astype(int))
                speed = opt["print_freq"] / (time.time() - start)
                time_str = time.asctime(time.localtime(time.time()))

                print("{} train epoch {} iter {} {} iters/s loss {} acc {}".format(time_str, i, iters, speed, loss.item(), acc))

                start = time.time()

        if i % opt["save_interval"] == 0 or i == opt["max_epoch"]:
            if not os.path.exists(opt["checkpoints_path"]):
                os.makedirs(opt["checkpoints_path"])
            save_model(model, opt["checkpoints_path"], opt["backbone"], i)
            print("model was saved.")
