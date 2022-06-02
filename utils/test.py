#!/usr/bin/env python3

import argparse
import copy
import os
import os.path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.autograd import Variable
from torchvision import models
from torchvision import transforms

parser = argparse.ArgumentParser(description='PyTorch Example')
parser.add_argument('--batch_size', type=int, default=64, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--test_batch', type=int, default=64, metavar='N',
                    help='input batch size for testing (default: 64)')
parser.add_argument('--epochs', type=int, default=1, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--lr', type=float, default=0.5, metavar='LR',
                    help='learning rate (default: 0.1)')
parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                    help='SGD momentum (default: 0.5)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=25, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--resume', type=bool, default=True, metavar='N',
                    help='resume from the last weights')

torch.cuda.empty_cache()
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

rgb_dir = '/home/kai/Desktop/RGB'
depth_dir = '/home/kai/Desktop/Depth'
label_dir = '/home/kai/Desktop/yoliclabel'


class MultiLabelDataSet(torch.utils.data.Dataset):
    def __init__(self, imgspath, dpath, imgslist, annotationpath, transforms=None, mode=0):
        self.imgslist = imgslist
        self.imgspath = imgspath
        self.dpath = dpath
        self.transform = transforms
        self.annotationpath = annotationpath
        self.mode = mode
        # print(annotationpath)

    def __len__(self):
        return len(self.imgslist)

    def __getitem__(self, index):
        ipath = os.path.join(self.imgspath, self.imgslist[index])
        dpath = os.path.join(self.dpath, self.imgslist[index])
        color_image = cv2.imread(ipath)
        depth_image = cv2.imread(dpath)
        d, d, d = cv2.split(depth_image)
        if self.mode == 1:
            rgbtrans = transforms.Compose(
                ([transforms.ColorJitter(brightness=0.3, contrast=0.2, saturation=0.2, hue=0.1)]))
            color_image = Image.fromarray(color_image)
            color_image = rgbtrans(color_image)
            color_image = np.asarray(color_image)
        b, g, r = cv2.split(color_image)
        img = cv2.merge([r, g, b, d])
        img = cv2.resize(img, (224, 224))
        # print(ipath)
        if self.transform is not None:
            img = self.transform(img)
        (filename, extension) = os.path.splitext(ipath)
        filename = os.path.basename(filename)
        annotation = os.path.join(self.annotationpath, filename + ".txt")
        label = np.loadtxt(annotation, dtype=np.int64)
        return img, label, filename


class MultiLabelRGBataSet(torch.utils.data.Dataset):
    def __init__(self, imgspath, imgslist, annotationpath, transforms=None):
        self.imgslist = imgslist
        self.imgspath = imgspath
        self.transform = transforms
        self.annotationpath = annotationpath
        # print(annotationpath)

    def __len__(self):
        return len(self.imgslist)

    def __getitem__(self, index):
        ipath = os.path.join(self.imgspath, self.imgslist[index])
        img = Image.open(ipath)
        # print(ipath)
        if self.transform is not None:
            img = self.transform(img)
        (filename, extension) = os.path.splitext(ipath)
        filename = os.path.basename(filename)
        annotation = os.path.join(self.annotationpath, filename + ".txt")
        label = np.loadtxt(annotation, dtype=np.int64)
        return img, label, filename


trans = transforms.Compose(([
    # transforms.Resize((224,224)),
    transforms.ToTensor()  # divides by 255
]))

alist = os.listdir(rgb_dir)
x_train, x_test = train_test_split(alist, test_size=0.3, random_state=2)


train = MultiLabelDataSet(rgb_dir, depth_dir,
                          x_train, label_dir, trans, 1)

test = MultiLabelDataSet(rgb_dir, depth_dir,
                         x_test, label_dir, trans, 0)


train_loader = torch.utils.data.DataLoader(train,
                                           batch_size=args.batch_size,
                                           shuffle=True, num_workers=16)

test_loader = torch.utils.data.DataLoader(test,
                                          batch_size=args.test_batch,
                                          shuffle=False, num_workers=16)


model = models.resnet50()
model.fc = nn.Linear(in_features=2048, out_features=1248, bias=True)
model.conv1 = nn.Conv2d(4, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
save_name = 'mobilenet_v2_new_time'
# print(model)
if args.cuda:
    model.cuda()

optimizer = optim.Adam(model.parameters())

criterion = nn.MultiLabelSoftMarginLoss()

best_correct = -999
def pred_acc(original, predicted, errornum):
    pred = torch.round(predicted).detach().numpy()
    orig = original.detach().numpy()
    num = 0
    enum = 0
    normal = np.asarray([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1])
    for cell in range(0, 1248, 12):
        if (orig[cell:cell + 12] == pred[cell:cell + 12]).all():
            num = num + 1
        else:
            if not (orig[cell:cell + 12] == normal).all() and not (pred[cell:cell + 12] == normal).all():
                if errornum == -1:
                    enum = enum + 1
    if errornum == -1:
        return (num + enum) / 104
    if errornum == 0:
        return num / 104


def train(epoch, model, loss_fn):
    model.train()
    for batch_idx, (data, target, filenames) in enumerate(train_loader):
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data), Variable(target)
        optimizer.zero_grad()
        output = model(data)
        loss = loss_fn(output, target)
        loss.backward()
        optimizer.step()


        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.10f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss.item()))


def train_evaluate(model):
    model.eval()
    running_loss = []
    running_acc = []
    running_acc2 = []
    global best_correct
    for batch_idx, (data, target, filenames) in enumerate(train_loader):
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data), Variable(target)
        output = model(data)
        loss = criterion(output, target)
        output = torch.sigmoid(output)
        # print(output)
        acc_ = []
        for i, d in enumerate(output):
            acc = pred_acc(torch.Tensor.cpu(target[i]), torch.Tensor.cpu(d), 0)
            acc_.append(acc)

        acc_2 = []
        for i, d in enumerate(output):
            acc = pred_acc(torch.Tensor.cpu(target[i]), torch.Tensor.cpu(d), -1)
            acc_2.append(acc)
        running_loss.append(loss.item())
        running_acc.append(np.asarray(acc_).mean())
        running_acc2.append(np.asarray(acc_2).mean())
    total_loss = np.asarray(running_loss).mean()
    total_acc = np.asarray(running_acc).mean()
    total_acc2 = np.asarray(running_acc2).mean()
    print('\nTrain set: total_batch_loss: {:.4f}, total imgs: {} , Acc: ({:.4f}%), 2ACC: ({:.4f}%)\n'.format(
        total_loss, len(train_loader.dataset), total_acc, total_acc2))
    now_correct = total_acc
    if best_correct < now_correct:
        best_correct = now_correct
        best_model_wts = copy.deepcopy(model.state_dict())
        torch.save(best_model_wts,
                   os.path.join(os.getcwd(), save_name + ".pth.tar"))
        print("New weight!")
    return total_loss, total_acc


def test(model):
    model.eval()
    running_loss = []
    running_acc = []
    running_acc2 = []
    global best_correct
    for batch_idx, (data, target, filenames) in enumerate(test_loader):
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data), Variable(target)
        output = model(data).float()
        loss = criterion(output, target)
        output = torch.sigmoid(output)

        acc_ = []
        for i, d in enumerate(output):
            acc = pred_acc(torch.Tensor.cpu(target[i]), torch.Tensor.cpu(d), 0)
            acc_.append(acc)

        acc_2 = []
        for i, d in enumerate(output):
            acc = pred_acc(torch.Tensor.cpu(target[i]), torch.Tensor.cpu(d), -1)
            acc_2.append(acc)
        running_loss.append(loss.item())
        running_acc.append(np.asarray(acc_).mean())
        running_acc2.append(np.asarray(acc_2).mean())
    total_loss = np.asarray(running_loss).mean()
    total_acc = np.asarray(running_acc).mean()
    total_acc2 = np.asarray(running_acc2).mean()
    print('\nTest set: total_batch_loss: {:.4f}, total imgs: {} , Acc: ({:.4f}%), 2classAcc: ({:.4f}%)\n'.format(
        total_loss, len(test_loader.dataset), total_acc, total_acc2))
    # now_correct = total_batch_acc
    # if best_correct < now_correct:
    #     best_correct = now_correct
    #     best_model_wts = copy.deepcopy(model.state_dict())
    #     torch.save(best_model_wts, os.path.join(os.getcwd(), "mobilenetv2_fp16_34294rgbd_1248output.pth.tar"))
    #     print("New weight!")
    return total_loss, total_acc


if __name__ == '__main__':

    for epoch in range(1, args.epochs + 1):
        train(epoch, model, loss_fn=criterion)
        train_loss, train_acc = train_evaluate(model)
        test_loss, test_acc = test(model)