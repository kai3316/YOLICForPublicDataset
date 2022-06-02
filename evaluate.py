import numpy as np
import torch


class EvalSegErr(Exception):
    def __init__(self, value):
        self.value = value

    def __str__(self):
        return repr(self.value)


def check_size(eval_segm, gt_segm):
    h_e, w_e = segm_size(eval_segm)
    h_g, w_g = segm_size(gt_segm)

    if (h_e != h_g) or (w_e != w_g):
        raise EvalSegErr("DiffDim: Different dimensions of matrices!")


def segm_size(segm):
    try:
        height = segm.shape[0]
        width = segm.shape[1]
    except IndexError:
        raise

    return height, width


def IOU(eval_segm, gt_segm):
    eval_segm = eval_segm.detach().numpy()
    gt_segm = gt_segm.detach().numpy()
    check_size(eval_segm, gt_segm)
    defined_classes = 19
    IOU = list([-1]) * defined_classes
    for index in range(defined_classes):
        cur_eval_mask = eval_segm[index, :, :]
        cur_gt_mask = gt_segm[index, :, :]
        if np.sum(cur_eval_mask) == 0 and np.sum(cur_gt_mask) == 0:
            continue
        n_ii = np.sum(np.logical_and(cur_eval_mask, cur_gt_mask))
        t_i = np.sum(cur_gt_mask)
        n_ij = np.sum(cur_eval_mask)
        IOU[index] = n_ii / (t_i + n_ij - n_ii)
    return IOU

def calculate_mIOU(val_loader, model):
    model.eval()
    iou_list_class1 = []
    iou_list_class2 = []
    iou_list_class3 = []
    iou_list_class4 = []
    iou_list_class5 = []
    iou_list_class6 = []
    iou_list_class7 = []
    iou_list_class8 = []
    iou_list_class9 = []
    iou_list_class10 = []
    iou_list_class11 = []
    iou_list_class12 = []
    iou_list_class13 = []
    iou_list_class14 = []
    iou_list_class15 = []
    iou_list_class16 = []
    iou_list_class17 = []
    iou_list_class18 = []
    iou_list_class19 = []
    for imgs, label_imgs in val_loader:
        with torch.no_grad(): # (corresponds to setting volatile=True in all variables, this is done during inference to reduce memory consumption)
            imgs = imgs.cuda()
            outputs = model(imgs)
            for i, d in enumerate(outputs):
                iou = IOU(torch.Tensor.cpu(label_imgs[i]), torch.Tensor.cpu(d))
                if iou[0] != -1:
                    iou_list_class1.append(iou[0])
                if iou[1] != -1:
                    iou_list_class2.append(iou[1])
                if iou[2] != -1:
                    iou_list_class3.append(iou[2])
                if iou[3] != -1:
                    iou_list_class4.append(iou[3])
                if iou[4] != -1:
                    iou_list_class5.append(iou[4])
                if iou[5] != -1:
                    iou_list_class6.append(iou[5])
                if iou[6] != -1:
                    iou_list_class7.append(iou[6])
                if iou[7] != -1:
                    iou_list_class8.append(iou[7])
                if iou[8] != -1:
                    iou_list_class9.append(iou[8])
                if iou[9] != -1:
                    iou_list_class10.append(iou[9])
                if iou[10] != -1:
                    iou_list_class11.append(iou[10])
                if iou[11] != -1:
                    iou_list_class12.append(iou[11])
                if iou[12] != -1:
                    iou_list_class13.append(iou[12])
                if iou[13] != -1:
                    iou_list_class14.append(iou[13])
                if iou[14] != -1:
                    iou_list_class15.append(iou[14])
                if iou[15] != -1:
                    iou_list_class16.append(iou[15])
                if iou[16] != -1:
                    iou_list_class17.append(iou[16])
                if iou[17] != -1:
                    iou_list_class18.append(iou[17])
                if iou[18] != -1:
                    iou_list_class19.append(iou[18])
    iou_list = [iou_list_class1, iou_list_class2, iou_list_class3, iou_list_class4, iou_list_class5, iou_list_class6, iou_list_class7, iou_list_class8, iou_list_class9, iou_list_class10, iou_list_class11, iou_list_class12, iou_list_class13, iou_list_class14, iou_list_class15, iou_list_class16, iou_list_class17, iou_list_class18, iou_list_class19]
    return iou_list
