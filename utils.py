import cv2
import os
import time
import numpy as np
import torch
import matplotlib.pyplot as plt

def bold(x):       return '\033[1m'  + str(x) + '\033[0m'
def dim(x):        return '\033[2m'  + str(x) + '\033[0m'
def italicized(x): return '\033[3m'  + str(x) + '\033[0m'
def underline(x):  return '\033[4m'  + str(x) + '\033[0m'
def blink(x):      return '\033[5m'  + str(x) + '\033[0m'
def inverse(x):    return '\033[7m'  + str(x) + '\033[0m'
def gray(x):       return '\033[90m' + str(x) + '\033[0m'
def red(x):        return '\033[91m' + str(x) + '\033[0m'
def green(x):      return '\033[92m' + str(x) + '\033[0m'
def yellow(x):     return '\033[93m' + str(x) + '\033[0m'
def blue(x):       return '\033[94m' + str(x) + '\033[0m'
def magenta(x):    return '\033[95m' + str(x) + '\033[0m'
def cyan(x):       return '\033[96m' + str(x) + '\033[0m'
def white(x):      return '\033[97m' + str(x) + '\033[0m'

class Tick():
    def __init__(self, name='', silent=False):
        self.name = name
        self.silent = silent

    def __enter__(self):
        self.t_start = time.time()
        if not self.silent:
            print(cyan('%s ' % (self.name)), end='', flush=True)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.t_end = time.time()
        self.delta = self.t_end-self.t_start
        self.fps = 1/self.delta

        if not self.silent:
            print(cyan('[%.0fms]' % (self.delta * 1000)), flush=True)

class Tock():
    def __init__(self, name=None, report_time=True):
        self.name = '' if name == None else name+':'
        self.report_time = report_time

    def __enter__(self):
        self.t_start = time.time()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.t_end = time.time()
        self.delta = self.t_end-self.t_start
        self.fps = 1/self.delta
        if self.report_time:
            print(yellow('%s%.0fms ' % (self.name, self.delta * 1000)), end='', flush=True)
        else:
            print(yellow('.'), end='', flush=True)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def save_model(model, ckpt):
    torch.save(model.state_dict(), ckpt)
    return model

def load_model(model, ckpt, device):
    if os.path.exists(ckpt):
        model.load_state_dict(torch.load(ckpt, map_location=device))
        print('Load ckpt from %s'%(green(ckpt)))
    else:
        print('Load ckpt failed, %s not found'%(yellow(ckpt)))
        print('Model weights are initialized randomly')
    return model

def binary_PR_Dice(y_true, y_pred, epsilon=1e-7):
    '''
        Parameters:
            label: [24, size, size]
            pred : [24, size, size]
        Returns:
            Precision, Recall, IoU, Dice
    '''
    assert y_true.shape == y_pred.shape

    y_true = y_true.reshape((-1))
    y_pred = y_pred.reshape((-1))

    tp = (y_true * y_pred).sum().astype(np.float32)
    tn = ((1 - y_true) * (1 - y_pred)).sum().astype(np.float32)
    fp = ((1 - y_true) * y_pred).sum().astype(np.float32)
    fn = (y_true * (1 - y_pred)).sum().astype(np.float32)

    prec = (tp + epsilon) / (tp + fp + epsilon)
    recall = (tp + epsilon) / (tp + fn + epsilon)
    # iou = (tp + epsilon) / (tp + fp + fn + epsilon)
    dice = (2 * tp + epsilon) / (2 * tp + fp + fn + epsilon)

    return np.array((prec, recall, dice), dtype=np.float32)

def fmt_PRDice(scores, color=True):
    if color:
        d1, d2, d3 = green('%.4f'%(scores[0])), yellow('%.4f'%(scores[1])), cyan('%.4f'%(scores[2]))
    else:
        d1, d2, d3 = '%.4f'%(scores[0]), '%.4f'%(scores[1]), '%.4f'%(scores[2])
    return 'P:%s R:%s D:%s'%(d1, d2, d3)

color_palate = [(0,0,0), (255, 255, 0), (0, 255, 0), (255, 0, 255), (44, 255, 64), (60, 255, 60), (60, 60, 255)]
def paint_label(image, label, color, thickness=3):
    assert label.dtype == np.uint8, label.dtype
    assert label.max() <= 1, label.max()
    paint = image.copy()

    binary_mask = (label * 255).astype(np.uint8)
    contours, hierarchy = cv2.findContours(binary_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(paint, contours, -1 , color, thickness)
    return paint

def paint_heatmap(image, heatmap, alpha = 0.5):
    paint = image.copy()
    paint = cv2.applyColorMap(norm_0_255(heatmap), cv2.COLORMAP_HOT)
    paint = cv2.addWeighted(paint, alpha, image, 1-alpha, 0)
    return paint

def overlay_img_mask(b_images, b_label, b_pred=None, b_heatmap=None, b_pred2=None):
    """
        Overlay image with masks for visualization
        b_images -> (24, M, N) or (24, M, N, 1)
        b_label -> (24, M, N) float32 or uint8
    """
    assert len(b_label.shape) == 3, b_label.shape
    assert b_label.dtype == np.uint8, b_label.dtype
    assert b_pred.dtype == np.uint8, b_pred.dtype

    if b_pred is not None:
        assert len(b_pred.shape) == 3, b_pred.shape
        assert b_pred.dtype == np.uint8, b_pred.dtype
    
    if b_heatmap is not None:
        assert len(b_heatmap.shape) == 3, b_heatmap.shape
        # assert b_heatmap.dtype == np.float32, b_heatmap.dtype

    if b_pred2 is not None:
        assert len(b_pred2.shape) == 3, b_pred2.shape
        assert b_pred2.dtype == np.uint8, b_pred2.dtype

    pairs = []
    for i in range(b_images.shape[0]):
        image = b_images[i]
        label = b_label[i]

        if len(image.shape) == 2 or image.shape[-1] == 1:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        
        image = (image * 0.8).astype(np.uint8)

        pair = [paint_label(image, label, color_palate[1])]

        paint_pred_heatmap = image.copy()
        if b_heatmap is not None:
            heatmap = b_heatmap[i]
            paint_pred_heatmap = paint_heatmap(paint_pred_heatmap, heatmap)
        
        if b_pred is not None:
            pred = b_pred[i]
            paint_pred_heatmap = paint_label(paint_pred_heatmap, pred, color_palate[2], thickness=4)

        if b_pred2 is not None:
            pred2 = b_pred2[i]
            paint_pred_heatmap = paint_label(paint_pred_heatmap, pred2, color_palate[3], thickness=2)

        if (b_heatmap is not None) or (b_pred is not None):
            pair.append(paint_pred_heatmap)

        pair = np.concatenate(pair, axis=0)
        pair = cv2.resize(pair, None, fx=0.5, fy=0.5)
        pairs.append(pair)

    row1, row2, row3 = [], [], []
    for i, pair in enumerate(pairs):
        if i // 8 == 0:
            row1.append(pair)
        if i // 8 == 1:
            row2.append(pair)
        if i // 8 == 2:
            row3.append(pair)

    row1 = np.concatenate(row1, axis=1)
    row2 = np.concatenate(row2, axis=1)
    row3 = np.concatenate(row3, axis=1)
    overlay = np.concatenate([row1, row2, row3], axis=0)
    return overlay