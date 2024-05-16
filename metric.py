from pymic.util.evaluation_seg import binary_dice, binary_iou, binary_assd
import torch
from torch.nn import functional as F
from scipy import ndimage
import numpy as np

def get_metric(pred, gt):
    assert pred.ndim == 3 and gt.ndim == 3
    return binary_dice(pred, gt), binary_iou(pred, gt), binary_assd(pred, gt)


def get_multi_class_metric(pred, gt, class_num, include_backgroud=False):
    dice_list, assd_list, iou_list = [], [], []
    i = 1 if not include_backgroud else 0
    for label in range(i, class_num):
        p, g = pred == label, gt == label
        dice_list.append(round(binary_dice(p, g), 3))
        assd_list.append(round(binary_assd(p, g), 3))
        iou_list.append(round(binary_iou(p, g), 3))

    return dice_list, iou_list, assd_list


def get_classwise_dice(predict, soft_y, pix_w=None):
    """
    Get dice scores for each class in predict (after softmax) and soft_y.

    :param predict: (tensor) Prediction of a segmentation network after softmax.
    :param soft_y: (tensor) The one-hot segmentation ground truth.
    :param pix_w: (optional, tensor) The pixel weight map. Default is None.

    :return: Dice score for each class.
    """

    if (pix_w is None):
        y_vol = torch.sum(soft_y, dim=0)
        p_vol = torch.sum(predict, dim=0)
        intersect = torch.sum(soft_y * predict, dim=0)
    else:
        y_vol = torch.sum(soft_y * pix_w, dim=0)
        p_vol = torch.sum(predict * pix_w, dim=0)
        intersect = torch.sum(soft_y * predict * pix_w, dim=0)
    dice_score = (2.0 * intersect + 1e-5) / (y_vol + p_vol + 1e-5)
    return dice_score


def dc(pred, gt, class_num, include_backgroud=False):
    i = 1 if not include_backgroud else 0

    #onehot
    onehot_gt = F.one_hot(gt, class_num).permute(0, 3, 1, 2)
    pred = F.one_hot(pred, class_num).permute(0, 3, 1, 2)

    pred, gt = pred[:, i:], gt[:, i:]
    intersection = (onehot_gt * pred).sum(dim=(-1, -2))
    union = onehot_gt.sum(dim=(-1, -2)) + pred.sum(dim=(-1, -2))

    return ((2 * intersection + 1e-5) / (union + 1e-5)).mean(0)



def get_edge_points(img):
    """
    Get edge points of a binary segmentation result.

    :param img: (numpy.array) a 2D or 3D array of binary segmentation.
    :return: an edge map.
    """
    dim = len(img.shape)
    if (dim == 2):
        strt = ndimage.generate_binary_structure(2, 1)
    else:
        strt = ndimage.generate_binary_structure(3, 1)
    ero = ndimage.binary_erosion(img, strt)
    edge = np.asarray(img, np.uint8) - np.asarray(ero, np.uint8)
    return edge


def binary_assd(s, g, spacing=None):
    """
    Get the Average Symetric Surface Distance (ASSD) between a binary segmentation
    and the ground truth.

    :param s: (numpy.array) a 2D or 3D binary image for segmentation.
    :param g: (numpy.array) a 2D or 2D binary image for ground truth.
    :param spacing: (list) A list for image spacing, length should be 2 or 3.

    :return: The ASSD value.
    """
    s_edge = get_edge_points(s)
    g_edge = get_edge_points(g)
    image_dim = len(s.shape)
    assert (image_dim == len(g.shape))
    if (spacing == None):
        spacing = [1.0] * image_dim
    else:
        assert (image_dim == len(spacing))
    s_dis = ndimage.distance_transform_edt(1 - s_edge, sampling=spacing)
    g_dis = ndimage.distance_transform_edt(1 - g_edge, sampling=spacing)

    ns = s_edge.sum()
    ng = g_edge.sum()
    s_dis_g_edge = s_dis * g_edge
    g_dis_s_edge = g_dis * s_edge
    assd = (s_dis_g_edge.sum() + g_dis_s_edge.sum()) / (ns + ng)
    return assd


def binary_dice(s, g):
    """
    Calculate the Dice score of two N-d volumes for binary segmentation.

    :param s: The segmentation volume of numpy array.
    :param g: the ground truth volume of numpy array.
    :param resize: (optional, bool)
        If s and g have different shapes, resize s to match g.
        Default is `True`.

    :return: The Dice value.
    """
    assert (len(s.shape) == len(g.shape))
    prod = np.multiply(s, g)
    s0 = prod.sum()
    s1 = s.sum()
    s2 = g.sum()
    dice = (2.0 * s0 + 1e-5) / (s1 + s2 + 1e-5)
    return dice


def metrics(pred, gt, class_num):
    if isinstance(pred,torch.Tensor):
        pred = pred.cpu().numpy()
    if isinstance(pred,torch.Tensor):
        gt = gt.cpu().numpy()
    class_dice = []
    class_assd = []
    for i in range(class_num):
        p, g = (pred == i), (gt == i)
        class_dice.append(binary_dice(p, g))
        class_assd.append(binary_assd(p, g))
    return class_dice, class_assd
