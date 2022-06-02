# camera-ready

import os
import pickle

import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torch.utils.data
# sys.path.append("/home/kai/Desktop/deeplabv3/model")
from sklearn.model_selection import train_test_split
from torch.autograd import Variable

# sys.path.append("/home/kai/Desktop/deeplabv3/utils")
from dataset_KITTI import dataset_KITTI
from model.convnext import convnext_base

# NOTE! NOTE! change this to not overwrite all log data when you train the model:
model_id = "16_base"
num_epochs = 150
batch_size = 32
learning_rate = 0.001

network = convnext_base(model_id, project_dir="./").cuda()

rgb_dir = '/home/kai/Desktop/KITTI/training/image_2'
label_dir = '/home/kai/Desktop/KITTI/training/semantic'

alist = os.listdir(rgb_dir)
x_train, x_test = train_test_split(alist, test_size=0.3, random_state=2)

train_dataset = dataset_KITTI(rgb_dir, x_train, label_dir, img_w=224, img_h=224, cell_size_h=16, cell_size_w=16)
val_dataset = dataset_KITTI(rgb_dir, x_test, label_dir, img_w=224, img_h=224, cell_size_h=16, cell_size_w=16)

num_train_batches = int(len(train_dataset)/batch_size)
num_val_batches = int(len(val_dataset)/batch_size)
print ("num_train_batches:", num_train_batches)
print ("num_val_batches:", num_val_batches)

train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=batch_size, shuffle=True,
                                           num_workers=16)
val_loader = torch.utils.data.DataLoader(dataset=val_dataset,
                                         batch_size=batch_size, shuffle=False,
                                         num_workers=16)

# params = add_weight_decay(network, l2_value=0.0001)
# optimizer = torch.optim.Adam(params, lr=learning_rate)
optimizer = torch.optim.Adam(network.parameters(), lr=learning_rate)

with open("/home/kai/Desktop/KITTI/meta/class_weights.pkl", "rb") as file: # (needed for python3)
    weights = np.array(pickle.load(file))
class_weights = np.ones((20, 16, 16))
for i in range(20):
    class_weights[i, :, :] = weights[i]
class_weights = np.transpose(class_weights, (1, 2, 0))  # (shape: (207, 75, 20))
class_weights = class_weights.flatten()
class_weights = torch.from_numpy(class_weights)
class_weights = Variable(class_weights.type(torch.FloatTensor)).cuda()

# loss function
# loss_fn = nn.MultiLabelSoftMarginLoss(weight=class_weights)
loss_fn = nn.MultiLabelSoftMarginLoss()

epoch_losses_train = []
epoch_losses_val = []
currentLoss = 999
for epoch in range(num_epochs):
    print ("###########################")
    print ("######## NEW EPOCH ########")
    print ("###########################")
    print ("epoch: %d/%d" % (epoch+1, num_epochs))

    ############################################################################
    # train:
    ############################################################################
    network.train() # (set in training mode, this affects BatchNorm and dropout)
    batch_losses = []
    for step, (imgs, label_imgs) in enumerate(train_loader):
        #current_time = time.time()
        # print(imgs.shape)
        # print(label_imgs.shape)

        imgs = imgs.cuda() # (shape: (batch_size, 3, img_h, img_w))
        label_imgs = label_imgs.cuda()

        outputs = network(imgs) # (shape: (batch_size, num_classes, img_h, img_w))
        # print(outputs.shape, label_imgs.shape)
        # exit(0)
        # compute the loss:
        loss = loss_fn(outputs, label_imgs)
        # print(loss)
        # exit(0)
        loss_value = loss.data.cpu().numpy()
        batch_losses.append(loss_value)

        # optimization step:
        optimizer.zero_grad() # (reset gradients)
        loss.backward() # (compute gradients)
        optimizer.step() # (perform optimization step)

        #print (time.time() - current_time)

    epoch_loss = np.mean(batch_losses)
    epoch_losses_train.append(epoch_loss)
    with open("%s/epoch_losses_train.pkl" % network.model_dir, "wb") as file:
        pickle.dump(epoch_losses_train, file)
    print ("train loss: %g" % epoch_loss)
    plt.figure(1)
    plt.plot(epoch_losses_train, "k^")
    plt.plot(epoch_losses_train, "k")
    plt.ylabel("loss")
    plt.xlabel("epoch")
    plt.title("train loss per epoch")
    plt.savefig("%s/epoch_losses_train.png" % network.model_dir)
    plt.close(1)

    print ("####")

    ############################################################################
    # val:
    ############################################################################
    network.eval() # (set in evaluation mode, this affects BatchNorm and dropout)
    batch_losses = []
    for step, (imgs, label_imgs) in enumerate(val_loader):
        with torch.no_grad(): # (corresponds to setting volatile=True in all variables, this is done during inference to reduce memory consumption)
            imgs = imgs.cuda() # (shape: (batch_size, 3, img_h, img_w))
            label_imgs = label_imgs.cuda() # (shape: (batch_size, img_h, img_w))

            outputs = network(imgs) # (shape: (batch_size, num_classes, img_h, img_w))

            # compute the loss:
            loss = loss_fn(outputs, label_imgs)
            loss_value = loss.data.cpu().numpy()
            batch_losses.append(loss_value)

    epoch_loss = np.mean(batch_losses)
    epoch_losses_val.append(epoch_loss)
    with open("%s/epoch_losses_val.pkl" % network.model_dir, "wb") as file:
        pickle.dump(epoch_losses_val, file)
    print ("val loss: %g" % epoch_loss)
    plt.figure(1)
    plt.plot(epoch_losses_val, "k^")
    plt.plot(epoch_losses_val, "k")
    plt.ylabel("loss")
    plt.xlabel("epoch")
    plt.title("val loss per epoch")
    plt.savefig("%s/epoch_losses_val.png" % network.model_dir)
    plt.close(1)

    # save the model weights to disk:
    if epoch_loss < currentLoss:
        currentLoss = epoch_loss
        checkpoint_path = network.checkpoints_dir + "/model_" + model_id +"_epoch_" + str(epoch+1) + ".pth"
        print(checkpoint_path)
        torch.save(network.state_dict(), checkpoint_path)
