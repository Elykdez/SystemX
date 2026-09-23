"""The unit detector's network (needs torch; the player only runs its exported TorchScript form).

CenterNet-style, fully convolutional, about half a million parameters: a stride-4 heatmap per
class (infantry, vehicle) marks unit centers, with the box size and the center's sub-cell offset
at each. RA2 draws units at a fixed scale, so one scale suffices; 1/16 features with dilated
convolutions see a whole vehicle and its surroundings. It learns from frames as the game drew
them (tools/detector.py), and whose unit a box holds is left to the caller: the house color
inside the box says it (perception/detector.py).

`Exported` wraps it for TorchScript: one uint8 BGR image (H, W, 3) in, the K best peaks out as
rows [x0, y0, x1, y1, score, class] in image pixels, highest score first.
"""

import torch
from torch import nn
import torch.nn.functional as F

CLASSES = ('infantry', 'vehicle')
STRIDE = 4
MULTIPLE = 16


def block(cin, cout, stride=1, dilation=1):
    return nn.Sequential(nn.Conv2d(cin, cout, 3, stride, padding=dilation, dilation=dilation, bias=False),
                         nn.BatchNorm2d(cout), nn.ReLU(inplace=True))


class UnitNet(nn.Module):
    def __init__(self, classes=len(CLASSES), width=16):
        super().__init__()
        w = width
        self.s2 = nn.Sequential(block(3, w, 2), block(w, w))
        self.s4 = nn.Sequential(block(w, 2 * w, 2), block(2 * w, 2 * w))
        self.s8 = nn.Sequential(block(2 * w, 4 * w, 2), block(4 * w, 4 * w))
        self.s16 = nn.Sequential(block(4 * w, 6 * w, 2), block(6 * w, 6 * w, dilation=2), block(6 * w, 6 * w, dilation=4))
        self.up8 = block(10 * w, 4 * w)
        self.up4 = block(6 * w, 3 * w)
        self.heat = nn.Sequential(block(3 * w, 3 * w), nn.Conv2d(3 * w, classes, 1))
        self.size = nn.Sequential(block(3 * w, 3 * w), nn.Conv2d(3 * w, 2, 1))
        self.offset = nn.Sequential(block(3 * w, 3 * w), nn.Conv2d(3 * w, 2, 1))
        nn.init.constant_(self.heat[-1].bias, -2.19)           # a prior of 0.1 for every class

    def forward(self, x):
        """x: (N, 3, H, W) in [-0.5, 0.5], H and W multiples of 16 -> heat logits, size (in strides), offset."""
        a = self.s2(x)
        b = self.s4(a)
        c = self.s8(b)
        d = self.s16(c)
        u = self.up8(torch.cat([F.interpolate(d, size=c.shape[-2:], mode='nearest'), c], 1))
        u = self.up4(torch.cat([F.interpolate(u, size=b.shape[-2:], mode='nearest'), b], 1))
        return self.heat(u), self.size(u), self.offset(u)


def normalize(images, multiple: int = 16):
    """uint8 BGR (N, H, W, 3) -> float (N, 3, H, W), padded to multiples of 16."""
    x = images.permute(0, 3, 1, 2).float() / 255.0 - 0.5
    h, w = x.shape[-2], x.shape[-1]
    return F.pad(x, (0, (multiple - w % multiple) % multiple, 0, (multiple - h % multiple) % multiple))


class Exported(nn.Module):
    def __init__(self, net, k: int = 128):
        super().__init__()
        self.net = net
        self.k = k
        self.stride, self.multiple = STRIDE, MULTIPLE

    def forward(self, image):
        heat, size, offset = self.net(normalize(image.unsqueeze(0), self.multiple))
        heat = torch.sigmoid(heat)
        peaks = heat * (heat == F.max_pool2d(heat, 3, 1, 1)).float()
        classes, height, width = peaks.shape[1], peaks.shape[2], peaks.shape[3]
        k = min(self.k, classes * height * width)
        scores, index = peaks.reshape(-1).topk(k)
        cls = torch.div(index, height * width, rounding_mode='floor')
        position = index - cls * height * width
        ys = torch.div(position, width, rounding_mode='floor')
        xs = position - ys * width
        off = offset[0].reshape(2, -1)[:, position]
        wh = size[0].reshape(2, -1)[:, position].clamp(min=0.5) * self.stride
        cx = (xs.float() + off[0]) * self.stride
        cy = (ys.float() + off[1]) * self.stride
        return torch.stack([cx - wh[0] / 2, cy - wh[1] / 2, cx + wh[0] / 2, cy + wh[1] / 2, scores, cls.float()], 1)


def focal_loss(logits, target, ignore):
    """CenterNet's penalty-reduced focal loss; `ignore` (1) removes negatives where labels are missing."""
    p = torch.sigmoid(logits).clamp(1e-4, 1 - 1e-4)
    positive = target.eq(1).float()
    negative = (1 - positive) * (1 - ignore)
    loss = (torch.log(p) * (1 - p) ** 2 * positive).sum() + \
        (torch.log(1 - p) * p ** 2 * (1 - target) ** 4 * negative).sum()
    return -loss / positive.sum().clamp(min=1)
