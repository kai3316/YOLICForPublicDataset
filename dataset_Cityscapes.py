import os
from collections import Counter

import cv2
import numpy as np
import torch

color_dict = {7: [128, 64, 128], 8: [244, 35, 232], 11: [70, 70, 70], 12: [102, 102, 156], 13: [190, 153, 153],
              17: [153, 153, 153], 19: [250, 170, 30], 20: [220, 220, 0], 21: [107, 142, 35], 22: [152, 251, 152],
              23: [70, 130, 180], 24: [220, 20, 60], 25: [255, 0, 0], 26: [0, 0, 142], 27: [0, 0, 70], 28: [0, 60, 100],
              31: [0, 80, 100], 32: [0, 0, 230], 33: [119, 11, 32]}
index_dict = {7: 0, 8: 1, 11: 2, 12: 3, 13: 4, 17: 5, 19: 6, 20: 7, 21: 8, 22: 9, 23: 10, 24: 11, 25: 12, 26: 13,
              27: 14, 28: 15, 31: 16, 32: 17, 33: 18}


class dataset_Cityscapes(torch.utils.data.Dataset):
    def __init__(self, imgspath, annotationpath, img_w=224, img_h=224, cell_size_h=14, cell_size_w=14,
                 num_classes=20, transforms=None):

        self.imagesPath = imgspath
        self.imagesList = os.listdir(imgspath)
        self.transform = transforms
        self.annotationPath = annotationpath
        self.img_w = img_w
        self.img_h = img_h
        self.cell_size_h = cell_size_h
        self.cell_size_w = cell_size_w
        self.num_classes = num_classes

    def __len__(self):
        return len(self.imagesList)

    def __getitem__(self, index):
        imgPath = os.path.join(self.imagesPath, self.imagesList[index])
        labeledImgPath = os.path.join(self.annotationPath, self.imagesList[index])
        colorImage = cv2.imread(imgPath, -1)
        imgname = self.imagesList[index]
        print(imgname)
        label_image = cv2.imread(labeledImgPath, -1)
        colorImage = cv2.resize(colorImage, (self.img_w, self.img_h), interpolation=cv2.INTER_NEAREST)  # (w,h)
        label_image = cv2.resize(label_image, (self.img_w, self.img_h), interpolation=cv2.INTER_NEAREST)  # (w,h)
        cellLabelImage = np.zeros(
            (self.num_classes, int(self.img_h / self.cell_size_h), int(self.img_w / self.cell_size_w)),
            np.uint8)  # (h,w,c)
        for y in range(0, label_image.shape[0], self.cell_size_h):
            for x in range(0, label_image.shape[1], self.cell_size_w):
                subImage = label_image[y:y + self.cell_size_h, x:x + self.cell_size_w]
                self.regionSearch(subImage, int(y / self.cell_size_h), int(x / self.cell_size_w), cellLabelImage)

        if self.transform is not None:
            colorImage = self.transform(colorImage)
        colorImage = colorImage / 255.0
        # img = img - np.array([0.485, 0.456, 0.406])
        # img = img / np.array([0.229, 0.224, 0.225])  # (shape: (256, 256, 3))
        colorImage = np.transpose(colorImage, (2, 0, 1))  # (shape: (3, 256, 256))
        colorImage = colorImage.astype(np.float32)

        # convert numpy -> torch:
        colorImage = torch.from_numpy(colorImage)  # (shape: (3, 256, 256))
        # cellLabelImage = torch.from_numpy(cellLabelImage)  # (shape: (20, 14, 14))
        # cellLabelImage = torch.from_numpy(cellLabelImage)  # (shape: (75, 207))
        # cellLabel = np.transpose(cellLabelImage, (1, 2, 0))  # (shape: (207, 75, 20))
        # cellLabel = cellLabel.flatten()
        # cellLabel = torch.from_numpy(cellLabel)  # (shape: (207*75*20))

        return colorImage, cellLabelImage

    @staticmethod
    def regionSearch(subImage, iy, ix, newImg):
        # print(subImage.shape)
        subColor = []
        for y in range(subImage.shape[0]):
            for x in range(subImage.shape[1]):
                temp = subImage[y][x]
                subColor.append(temp)
        counterResult = Counter(subColor)
        # print(emptyImg[y][x])
        if len(counterResult) != 0:
            for item in counterResult:
                # print(item, index_dict[item])
                newImg[item][iy][ix] = 1



if __name__ == '__main__':
    img_w = 2048
    img_h = 1024
    cell_size_h = 32
    cell_size_w = 32
    rgb_dir = '/home/kai/Desktop/Cityscapes/meta/imgsForTrain'
    label_dir = '/home/kai/Desktop/Cityscapes/meta/label_imgsForTrain'
    train = dataset_Cityscapes(rgb_dir, label_dir, img_w, img_h, cell_size_h, cell_size_w)
    color_image, newLabel_image = train.__getitem__(0)
    print(color_image.shape, newLabel_image.shape)
    # rgb_dir = '/home/kai/Desktop/Cityscapes/meta/imgsForTest'
    # label_dir = '/home/kai/Desktop/Cityscapes/meta/label_imgsForTest'
    # test = dataset_Cityscapes(rgb_dir, label_dir)
    # color_image, newLabel_image = test.__getitem__(0)
    # cv2.imshow('color_image', color_image)
    # cv2.waitKey(0)
