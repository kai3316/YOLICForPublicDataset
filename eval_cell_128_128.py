# camera-ready

import itertools
import os

import matplotlib.pyplot as plt
import numpy as np
import torch.utils.data
from PIL import Image
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from torchvision import transforms

from model.mobilenet128 import mobilenet_v2


# sys.path.append("/home/kai/Desktop/deeplabv3/model")
# sys.path.append("/home/kai/Desktop/deeplabv3/utils")


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """

    cm_normalize = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    plt.imshow(cm_normalize, interpolation='nearest', cmap=cmap)
    plt.title(title, fontsize=16)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.4f' if normalize else 'd'
    thresh = cm_normalize.max() / 2.
    for i, j in itertools.product(range(cm_normalize.shape[0]), range(cm_normalize.shape[1])):
        plt.text(j, i - 0.1, format(cm_normalize[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm_normalize[i, j] > thresh else "black")
        plt.text(j, i + 0.2, format(cm[i, j], 'd'),
                 horizontalalignment="center",
                 color="white" if cm_normalize[i, j] > thresh else "black")
    plt.ylabel('True label', fontsize=18)
    plt.xlabel('Predicted label', fontsize=18)
    plt.tight_layout()


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

        color_image = Image.open(ipath)
        # print(ipath)
        if self.transform is not None:
            img = self.transform(color_image)
        (filename, extension) = os.path.splitext(ipath)
        filename = os.path.basename(filename)
        annotation = os.path.join(self.annotationpath, filename + ".npy")
        label = np.load(annotation)
        # print(label.shape)
        return img, label


trans = transforms.Compose(([
    # transforms.Resize((224,224)),
    transforms.ToTensor()  # divides by 255
]))

num_epochs = 300
batch_size = 1
learning_rate = 0.01
network = mobilenet_v2("1")
network.load_state_dict(torch.load(
    '/home/kai/Desktop/YOLICForPublicData/mobilenet_v2_cityscapes_128_128_flip1/model__epoch_42_loss_0.088234484.pth'))
# network.fc = nn.Linear(512, 4000, bias=True)
print(network)
# network.classifier[1] = nn.Linear(1280, 4000)
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

# train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
#                                            batch_size=batch_size, shuffle=True,
#                                            num_workers=16)
val_loader = torch.utils.data.DataLoader(dataset=val_dataset,
                                         batch_size=batch_size, shuffle=False,
                                         num_workers=0)
# c
# cellList = [[[0, 512], [64, 576]], [[64, 512], [128, 576]], [[128, 512], [192, 576]], [[192, 512], [256, 576]],
#             [[256, 512], [320, 576]], [[320, 512], [384, 576]], [[384, 512], [448, 576]], [[448, 512], [512, 576]],
#             [[512, 512], [576, 576]], [[576, 512], [640, 576]], [[640, 512], [704, 576]], [[704, 512], [768, 576]],
#             [[768, 512], [832, 576]], [[832, 512], [896, 576]], [[896, 512], [960, 576]], [[960, 512], [1024, 576]],
#             [[1024, 512], [1088, 576]], [[1088, 512], [1152, 576]], [[1152, 512], [1216, 576]],
#             [[1216, 512], [1280, 576]], [[1280, 512], [1344, 576]], [[1344, 512], [1408, 576]],
#             [[1408, 512], [1472, 576]], [[1472, 512], [1536, 576]], [[1536, 512], [1600, 576]],
#             [[1600, 512], [1664, 576]], [[1664, 512], [1728, 576]], [[1728, 512], [1792, 576]],
#             [[1792, 512], [1856, 576]], [[1856, 512], [1920, 576]], [[1920, 512], [1984, 576]],
#             [[1984, 512], [2048, 576]],
#             [[0, 576], [64, 640]], [[64, 576], [128, 640]], [[128, 576], [192, 640]], [[192, 576], [256, 640]],
#             [[256, 576], [320, 640]], [[320, 576], [384, 640]], [[384, 576], [448, 640]], [[448, 576], [512, 640]],
#             [[512, 576], [576, 640]], [[576, 576], [640, 640]], [[640, 576], [704, 640]], [[704, 576], [768, 640]],
#             [[768, 576], [832, 640]], [[832, 576], [896, 640]], [[896, 576], [960, 640]], [[960, 576], [1024, 640]],
#             [[1024, 576], [1088, 640]], [[1088, 576], [1152, 640]], [[1152, 576], [1216, 640]],
#             [[1216, 576], [1280, 640]], [[1280, 576], [1344, 640]], [[1344, 576], [1408, 640]],
#             [[1408, 576], [1472, 640]], [[1472, 576], [1536, 640]], [[1536, 576], [1600, 640]],
#             [[1600, 576], [1664, 640]], [[1664, 576], [1728, 640]], [[1728, 576], [1792, 640]],
#             [[1792, 576], [1856, 640]], [[1856, 576], [1920, 640]], [[1920, 576], [1984, 640]],
#             [[1984, 576], [2048, 640]],
#             [[0, 640], [64, 704]], [[64, 640], [128, 704]], [[128, 640], [192, 704]], [[192, 640], [256, 704]],
#             [[256, 640], [320, 704]], [[320, 640], [384, 704]], [[384, 640], [448, 704]], [[448, 640], [512, 704]],
#             [[512, 640], [576, 704]], [[576, 640], [640, 704]], [[640, 640], [704, 704]], [[704, 640], [768, 704]],
#             [[768, 640], [832, 704]], [[832, 640], [896, 704]], [[896, 640], [960, 704]], [[960, 640], [1024, 704]],
#             [[1024, 640], [1088, 704]], [[1088, 640], [1152, 704]], [[1152, 640], [1216, 704]],
#             [[1216, 640], [1280, 704]], [[1280, 640], [1344, 704]], [[1344, 640], [1408, 704]],
#             [[1408, 640], [1472, 704]], [[1472, 640], [1536, 704]], [[1536, 640], [1600, 704]],
#             [[1600, 640], [1664, 704]], [[1664, 640], [1728, 704]], [[1728, 640], [1792, 704]],
#             [[1792, 640], [1856, 704]], [[1856, 640], [1920, 704]], [[1920, 640], [1984, 704]],
#             [[1984, 640], [2048, 704]],
#             [[0, 704], [64, 768]], [[64, 704], [128, 768]], [[128, 704], [192, 768]], [[192, 704], [256, 768]],
#             [[256, 704], [320, 768]], [[320, 704], [384, 768]], [[384, 704], [448, 768]], [[448, 704], [512, 768]],
#             [[512, 704], [576, 768]], [[576, 704], [640, 768]], [[640, 704], [704, 768]], [[704, 704], [768, 768]],
#             [[768, 704], [832, 768]], [[832, 704], [896, 768]], [[896, 704], [960, 768]], [[960, 704], [1024, 768]],
#             [[1024, 704], [1088, 768]], [[1088, 704], [1152, 768]], [[1152, 704], [1216, 768]],
#             [[1216, 704], [1280, 768]], [[1280, 704], [1344, 768]], [[1344, 704], [1408, 768]],
#             [[1408, 704], [1472, 768]], [[1472, 704], [1536, 768]], [[1536, 704], [1600, 768]],
#             [[1600, 704], [1664, 768]], [[1664, 704], [1728, 768]], [[1728, 704], [1792, 768]],
#             [[1792, 704], [1856, 768]], [[1856, 704], [1920, 768]], [[1920, 704], [1984, 768]],
#             [[1984, 704], [2048, 768]],
#             [[0, 768], [128, 896]], [[128, 768], [256, 896]], [[256, 768], [384, 896]], [[384, 768], [512, 896]],
#             [[512, 768], [640, 896]], [[640, 768], [768, 896]], [[768, 768], [896, 896]], [[896, 768], [1024, 896]],
#             [[1024, 768], [1152, 896]], [[1152, 768], [1280, 896]], [[1280, 768], [1408, 896]],
#             [[1408, 768], [1536, 896]], [[1536, 768], [1664, 896]], [[1664, 768], [1792, 896]],
#             [[1792, 768], [1920, 896]], [[1920, 768], [2048, 896]],
#             [[0, 896], [128, 1024]], [[128, 896], [256, 1024]], [[256, 896], [384, 1024]], [[384, 896], [512, 1024]],
#             [[512, 896], [640, 1024]], [[640, 896], [768, 1024]], [[768, 896], [896, 1024]], [[896, 896], [1024, 1024]],
#             [[1024, 896], [1152, 1024]], [[1152, 896], [1280, 1024]], [[1280, 896], [1408, 1024]],
#             [[1408, 896], [1536, 1024]], [[1536, 896], [1664, 1024]], [[1664, 896], [1792, 1024]],
#             [[1792, 896], [1920, 1024]], [[1920, 896], [2048, 1024]]]  # w,h
#
# newCellList = [[[384, 512], [512, 640]],
#                [[512, 512], [640, 640]], [[640, 512], [768, 640]], [[768, 512], [896, 640]], [[896, 512], [1024, 640]],
#                [[1024, 512], [1152, 640]], [[1152, 512], [1280, 640]], [[1280, 512], [1408, 640]],
#                [[1408, 512], [1536, 640]], [[1536, 512], [1664, 640]],
#                [[128, 640], [256, 768]], [[256, 640], [384, 768]], [[384, 640], [512, 768]], [[512, 640], [640, 768]],
#                [[640, 640], [768, 768]], [[768, 640], [896, 768]], [[896, 640], [1024, 768]], [[1024, 640], [1152, 768]],
#                [[1152, 640], [1280, 768]], [[1280, 640], [1408, 768]], [[1408, 640], [1536, 768]], [[1536, 640], [1664, 768]],
#                [[1664, 640], [1792, 768]], [[1792, 640], [1920, 768]],
#                [[0, 768], [128, 896]], [[128, 768], [256, 896]], [[256, 768], [384, 896]], [[384, 768], [512, 896]],
#                [[512, 768], [640, 896]], [[640, 768], [768, 896]], [[768, 768], [896, 896]], [[896, 768], [1024, 896]],
#                [[1024, 768], [1152, 896]], [[1152, 768], [1280, 896]], [[1280, 768], [1408, 896]],
#                [[1408, 768], [1536, 896]], [[1536, 768], [1664, 896]], [[1664, 768], [1792, 896]],
#                [[1792, 768], [1920, 896]], [[1920, 768], [2048, 896]],
#                [[0, 896], [128, 1024]], [[128, 896], [256, 1024]], [[256, 896], [384, 1024]], [[384, 896], [512, 1024]],
#                [[512, 896], [640, 1024]], [[640, 896], [768, 1024]], [[768, 896], [896, 1024]],
#                [[896, 896], [1024, 1024]],
#                [[1024, 896], [1152, 1024]], [[1152, 896], [1280, 1024]], [[1280, 896], [1408, 1024]],
#                [[1408, 896], [1536, 1024]], [[1536, 896], [1664, 1024]], [[1664, 896], [1792, 1024]],
#                [[1792, 896], [1920, 1024]], [[1920, 896], [2048, 1024]]]  # w,h
# print(len(newCellList))

windowSize = 128
size = 2048 // windowSize


def getNewCellList(windowSize):
    newCellList = []
    for i in range(0, int(1024 / windowSize)):
        for j in range(0, int(2048 / windowSize)):
            newCellList.append([[i * windowSize, j * windowSize], [(i + 1) * windowSize, (j + 1) * windowSize]])
    return newCellList


newCellList = getNewCellList(windowSize)
print(len(newCellList))
priority1 = [11, 12, 18, 17, 16, 15, 14, 13, 3, 4, 5, 9, 1, 2, 8, 0, 6, 7, 10]
priority2 = [6, 7, 11, 12, 18, 17, 16, 15, 14, 13, 3, 4, 5, 2, 8, 10, 9, 1, 0]
priority = [3, 4, 2, 1, 0]

optimizer = torch.optim.Adam(network.parameters(), lr=learning_rate)
network.eval()  # (set in evaluation mode, this affects BatchNorm and dropout)
batch_losses = []
ans = []
orig = []
ans2 = []
orig2 = []
for step, (imgs, label_imgs) in enumerate(val_loader):
    with torch.no_grad():  # (corresponds to setting volatile=True in all variables, this is done during inference to reduce memory consumption)
        imgs = imgs.cuda()  # (shape: (batch_size, 3, img_h, img_w))
        label_imgs = label_imgs[0]  # (shape: (batch_size, img_h, img_w))
        # print(label_imgs[0].shape)
        outputs = network(imgs)[0]  # (shape: (batch_size, num_classes, img_h, img_w))
        outputs = torch.sigmoid(outputs)
        outputs = torch.permute(outputs, (1, 2, 0))
        labels = torch.permute(label_imgs, (1, 2, 0)).numpy()
        outputs = outputs.cpu().numpy()
        outputs = np.where(outputs >= 0.5, 1, 0)
        outputs = np.reshape(outputs, (5 * 8 * 16, 1)).flatten()
        labels = np.reshape(labels, (5 * 8 * 16, 1)).flatten()
        # outputs = np.where(outputs >= 0.5, 1, 0)

        for index, i in enumerate(newCellList):
            # if index <191 then continue:
            if not (67 <= index <= 76 or 81 <= index <= 94 or index >= 96):
                continue
            # imgs = cv2.rectangle(imgs, (i[0][0], i[0][1]), (i[1][0], i[1][1]), (0, 255, 0), 2)
            # if index > 24:
            #     priority = priority2
            # else:
            #     priority = priority1
            pred = outputs[index * 5:index * 5 + 5]
            origin = labels[index * 5:index * 5 + 5]
            flagp = np.zeros(5, dtype=int)
            flago = np.zeros(5, dtype=int)
            # # # print(pred)
            # # # print(origin)
            # # # print()
            # for j in range(0, 5):
            #     if pred[j] == 1:
            #         flagp[priority.index(j)] = 1
            #     if origin[j] == 1:
            #         flago[priority.index(j)] = 1
            # #
            # # # print(origin)
            #
            # predClass = priority[np.argmax(flagp)]
            #
            # originClass = priority[np.argmax(flago)]
            predClass = np.argmax(pred)
            #
            originClass = np.argmax(origin)
            # print(predClass, originClass)
            ans.append(predClass)
            orig.append(originClass)

            if predClass == 0:
                ans2.append(0)
            else:
                ans2.append(1)
            if originClass == 0:
                orig2.append(0)
            else:
                orig2.append(1)

class_names = ["Road", "Sidewalk", "Other risks", "Human", "Vehicle"]
class_names2 = ["Road", "all risks"]
# class_names = ["Bump", "Column", "Dent", "Fence", "Creature", "Vehicle", "Wall", "Weed", "ZebraCrossing", "TrafficCone",
#                "Normal"]

print(metrics.classification_report(orig, ans, target_names=class_names, digits=4))
matrix = confusion_matrix(orig, ans)
plt.figure(figsize=(9, 9))
plot_confusion_matrix(matrix, classes=class_names, normalize=True, title="Normalized confusion matrix")
plt.show()
# plt.savefig( 'Confusion Matrix.png')
plt.close()

print(metrics.classification_report(orig2, ans2, target_names=class_names2, digits=4))
matrix2 = confusion_matrix(orig2, ans2)
plt.figure(figsize=(6, 6))
plot_confusion_matrix(matrix2, classes=class_names2, normalize=True, title="Normalized confusion matrix")
plt.show()
# plt.savefig( 'Confusion Matrix.png')
plt.close()
# ./t-rex -a ethash -o stratum+tcp://eth-jp1.nanopool.org:9999 -u 0x262C4bEE249AD25eB45F855498C32F001Fc8BbAe -p x

# if predClass != originClass:
#     print(np.round(pred))
#     print(origin)
# label = ''
# for classindex, j in enumerate(pred):
#     if j==1:
#         label = label+str(classindex) + ' '
# print(label)
# cv2.putText(imgs, str(ans), (i[0][0], i[0][1]+30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)
# for i in range(0,len(outputs),20):
#     # print(outputs[i:i+20])
#     if np.argmax(outputs[i:i+20]) ==19:
#         print(np.argsort(outputs[i:i+20])[-2])
#     else:
#         print(np.argsort(outputs[i:i+20])[-1])
# print((cellList[int(i/20)][0][0],cellList[int(i/20)][0][1]))
# for (index, value) in enumerate(outputs[i:i+20]):
#     if value ==1:
#         cv2.putText(imgs, str(index), (cellList[i/20][0][0],cellList[i/20][0][1]), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
# print(step)
# plt.savefig('./'+str(step)+'.png',imgs)
# plt.imshow(imgs)
# plt.show()
# exit(0)

# with open("/home/kai/Desktop/Cityscapes/meta/class_weights.pkl", "rb") as file:  # (needed for python3)
#     weights = np.array(pickle.load(file))
# class_weights = np.ones((20, 16, 16))
# for i in range(20):
#     class_weights[i, :, :] = weights[i]
# class_weights = np.transpose(class_weights, (1, 2, 0))  # (shape: (207, 75, 20))
# class_weights = class_weights.flatten()
# class_weights = torch.from_numpy(class_weights)
# class_weights = Variable(class_weights.type(torch.FloatTensor)).cuda()

# loss function
# loss_fn = nn.MultiLabelSoftMarginLoss(weight=class_weights)
# loss_fn = nn.MultiLabelSoftMarginLoss()
