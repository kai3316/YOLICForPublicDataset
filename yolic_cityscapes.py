# camera-ready

import os
import pickle

# import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torch.utils.data
# sys.path.append("/home/kai/Desktop/deeplabv3/model")
from PIL import Image
# from utils.utils import add_weight_decay
from torchvision import transforms, models

# from dataset_Cityscapes import dataset_Cityscapes
# from dataset_KITTI import dataset_KITTI
from model.mobilenet128 import mobilenet_v2


# from sklearn.model_selection import train_test_split
# from timm.models.convnext import convnext_tiny
# from torch.autograd import Variable
# sys.path.append("/home/kai/Desktop/deeplabv3/utils")
# from torchvision.models import mobilenet_v2, resnet50, resnet18, mobilenet_v3_large


# NOTE! NOTE! change this to not overwrite all log data when you train the model:


class dataset_Cityscapes(torch.utils.data.Dataset):
    def __init__(self, imgspath, annotationpath, transforms=None):
        self.imgspath = imgspath
        self.imgslist = os.listdir(imgspath)
        self.transform = transforms
        self.annotationpath = annotationpath

    def __len__(self):
        return len(self.imgslist)

    def __getitem__(self, index):
        ipath = os.path.join(self.imgspath, self.imgslist[index])

        # color_image = cv2.imread(ipath)
        color_image = Image.open(ipath)
        # resize color_image to (img_w, img_h)
        # color_image = color_image.resize((512, 256), Image.ANTIALIAS)
        # color_image = color_image[512:1024, :, :]
        # color_image = cv2.resize(color_image, (224, 224))
        # print(ipath)
        # rgbtrans = transforms.Compose(
        #     ([transforms.ColorJitter(brightness=0.3, contrast=0.2, saturation=0.2, hue=0.1)]))
        # # color_image = Image.fromarray(color_image)
        # color_image = rgbtrans(color_image)
        # color_image = np.asarray(color_image)
        if self.transform is not None:
            img = self.transform(color_image)
        (filename, extension) = os.path.splitext(ipath)
        filename = os.path.basename(filename)
        # annotation = os.path.join(self.annotationpath, filename + ".txt")
        # label = np.loadtxt(annotation, dtype=np.int64)
        annotation = os.path.join(self.annotationpath, filename + ".npy")
        label = np.load(annotation)
        # label to tensor
        label = torch.from_numpy(label).long()
        # flip the image and label with 50% probability
        if np.random.rand() > 0.5:
            img = torch.flip(img, [2])
            label = torch.flip(label, [2])
        # apply color jittering with 50% probability
        if np.random.rand() > 0.5:
            img = transforms.ColorJitter(brightness=0.3, contrast=0.2, saturation=0.2, hue=0.1)(img)
        # print(label.shape)
        return img, label


pre_trained_mean, pre_trained_std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
trans = transforms.Compose(([
    # transforms.Resize((224,224)),
    # transforms.ColorJitter(brightness=0.3, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.ToTensor(),
    transforms.Normalize(mean=pre_trained_mean, std=pre_trained_std)
]))
l1_param = 0.0000001
model_id = "./mn_cityscapes_128_128"
if not os.path.exists(model_id):
    os.mkdir(model_id)
num_epochs = 200
batch_size = 4
learning_rate = 0.01
network = mobilenet_v2(model_id)
# network = convnext_tiny(model_id, './')
imageNet_model = models.mobilenet_v2(pretrained=True).state_dict()
yolic_net = network.state_dict()
state_dict = {k: v for k, v in imageNet_model.items() if k in yolic_net.keys()}
yolic_net.update(state_dict)
network.load_state_dict(yolic_net)
# network = resnet50(pretrained=True)
# network = models.mobilenet_v2(pretrained=True)
# print(network)
# network.classifier[3] = nn.Linear(in_features=1280, out_features=3800)
# network.fc = nn.Linear(2048, 960, bias=True)
# print(network)
# network.classifier[1] = nn.Linear(1280, 112, bias=True)

# network.load_state_dict(torch.load('/home/kai/Desktop/YOLICForPublicData/mn_cityscapes_128_128/model__epoch_7_loss_0.112673946.pth'))
network = network.cuda()

rgb_dir = '/home/kai/Desktop/Cityscapes/meta/imgsForTrain'
label_dir = '/home/kai/Desktop/Cityscapes/meta/train_128_128'
train_dataset = dataset_Cityscapes(rgb_dir, label_dir, transforms=trans)

rgb_dir = '/home/kai/Desktop/Cityscapes/meta/imgsForTest'
label_dir = '/home/kai/Desktop/Cityscapes/meta/test_128_128'
val_dataset = dataset_Cityscapes(rgb_dir, label_dir, transforms=trans)

num_train_batches = int(len(train_dataset) / batch_size)
num_val_batches = int(len(val_dataset) / batch_size)
print("num_train_batches:", num_train_batches)
print("num_val_batches:", num_val_batches)

train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=batch_size, shuffle=True,
                                           num_workers=0)
val_loader = torch.utils.data.DataLoader(dataset=val_dataset,
                                         batch_size=batch_size, shuffle=False,
                                         num_workers=0)

# params = add_weight_decay(network, l2_value=0.005)
# optimizer = torch.optim.AdamW(params, lr=learning_rate)
# optimizer = torch.optim.Adam(network.parameters(), lr=learning_rate)
optimizer = torch.optim.SGD(network.parameters(), lr=learning_rate, momentum=0.9, weight_decay=0.0005)
# learning rate cos scheduler
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=0)
# optimizer = torch.optim.SGD(network.parameters(), lr=learning_rate, momentum=0.9, weight_decay=0.0005)
# scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)

# with open("/home/kai/Desktop/Cityscapes/meta/class_weights.pkl", "rb") as file:  # (needed for python3)
#     weights = np.array(pickle.load(file))
# # print(weights[:-1].shape)
# class_weights = np.tile(weights[:-1], 200)
# class_weights = torch.from_numpy(class_weights)
# class_weights = Variable(class_weights.type(torch.FloatTensor)).cuda()
# class_weights = np.ones((20, 16, 16))
# for i in range(20):
#     class_weights[i, :, :] = weights[i]
# class_weights = np.transpose(class_weights, (1, 2, 0))  # (shape: (207, 75, 20))
# class_weights = class_weights.flatten()
# class_weights = torch.from_numpy(class_weights)
# class_weights = Variable(class_weights.type(torch.FloatTensor)).cuda()

# loss function
# loss_fn = nn.MultiLabelSoftMarginLoss(weight=class_weights)
loss_fn = nn.MultiLabelSoftMarginLoss()

epoch_losses_train = []
epoch_losses_val = []
currentLoss = 999
for epoch in range(num_epochs):
    print("###########################")
    print("######## NEW EPOCH ########")
    print("###########################")
    print("epoch: %d/%d" % (epoch + 1, num_epochs))

    ############################################################################
    # train:
    ############################################################################
    network.train()  # (set in training mode, this affects BatchNorm and dropout)
    batch_losses = []
    for step, (imgs, label_imgs) in enumerate(train_loader):
        # current_time = time.time()
        # print(imgs.shape)
        # print(label_imgs.shape)

        imgs = imgs.cuda()  # (shape: (batch_size, 3, img_h, img_w))
        label_imgs = label_imgs.cuda()

        outputs = network(imgs)  # (shape: (batch_size, num_classes, img_h, img_w))
        # print(outputs.shape, label_imgs.shape)
        # exit(0)
        # compute the loss:
        # regularize_loss = 0
        # for param in network.parameters():
        #     regularize_loss += torch.sum(torch.abs(param))
        # loss = loss_fn(outputs, label_imgs) + l1_param * regularize_loss
        loss = loss_fn(outputs, label_imgs)
        loss_value = loss.data.cpu().numpy()
        batch_losses.append(loss_value)

        # optimization step:
        optimizer.zero_grad()  # (reset gradients)
        loss.backward()  # (compute gradients)
        optimizer.step()  # (perform optimization step)
        scheduler.step()
        # print (time.time() - current_time)

    epoch_loss = np.mean(batch_losses)
    epoch_losses_train.append(epoch_loss)
    with open("%s/epoch_losses_train.pkl" % model_id, "wb") as file:
        pickle.dump(epoch_losses_train, file)
    print("train loss: %g" % epoch_loss)
    plt.figure(1)
    plt.plot(epoch_losses_train, "k^")
    plt.plot(epoch_losses_train, "k")
    plt.ylabel("loss")
    plt.xlabel("epoch")
    plt.title("train loss per epoch")
    plt.savefig("%s/epoch_losses_train.png" % model_id)
    plt.close(1)

    print("####")

    ############################################################################
    # val:
    ############################################################################
    network.eval()  # (set in evaluation mode, this affects BatchNorm and dropout)
    batch_losses = []
    for step, (imgs, label_imgs) in enumerate(val_loader):
        with torch.no_grad():  # (corresponds to setting volatile=True in all variables, this is done during inference to reduce memory consumption)
            imgs = imgs.cuda()  # (shape: (batch_size, 3, img_h, img_w))
            label_imgs = label_imgs.cuda()  # (shape: (batch_size, img_h, img_w))

            outputs = network(imgs)  # (shape: (batch_size, num_classes, img_h, img_w))

            # compute the loss:
            loss = loss_fn(outputs, label_imgs)
            loss_value = loss.data.cpu().numpy()
            batch_losses.append(loss_value)

    epoch_loss = np.mean(batch_losses)
    epoch_losses_val.append(epoch_loss)
    with open("%s/epoch_losses_val.pkl" % model_id, "wb") as file:
        pickle.dump(epoch_losses_val, file)
    print("val loss: %g" % epoch_loss)
    plt.figure(1)
    plt.plot(epoch_losses_val, "k^")
    plt.plot(epoch_losses_val, "k")
    plt.ylabel("loss")
    plt.xlabel("epoch")
    plt.title("val loss per epoch")
    plt.savefig("%s/epoch_losses_val.png" % model_id)
    plt.close(1)

    # save the model weights to disk:
    if epoch_loss < currentLoss:
        currentLoss = epoch_loss
        checkpoint_path = model_id + "/model_" + "_epoch_" + str(epoch + 1) + "_loss_" + str(currentLoss) + ".pth"
        print(checkpoint_path)
        torch.save(network.state_dict(), checkpoint_path)
