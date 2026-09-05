"""U-Net for flood extent. Width and depth are CLI knobs so capacity can be swept.

GroupNorm, not BatchNorm: a batch here is a handful of 128x128 crops, and BatchNorm
statistics over that are noise. Fully convolutional, so it trains on crops and evaluates
on whole scenes of different sizes (Patna 128, Mumbai 256 on the 60 m grid).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


def gn(c):
    return nn.GroupNorm(min(8, c), c)


class Block(nn.Module):
    def __init__(self, cin, cout, p_drop=0.0):
        super().__init__()
        self.f = nn.Sequential(
            nn.Conv2d(cin, cout, 3, padding=1, bias=False), gn(cout), nn.SiLU(),
            nn.Dropout2d(p_drop),
            nn.Conv2d(cout, cout, 3, padding=1, bias=False), gn(cout), nn.SiLU())

    def forward(self, x):
        return self.f(x)


class UNet(nn.Module):
    def __init__(self, cin, width=32, depth=4, p_drop=0.0):
        super().__init__()
        self.depth = depth
        ch = [width * 2 ** i for i in range(depth + 1)]
        self.inc = Block(cin, ch[0], p_drop)
        self.down = nn.ModuleList([Block(ch[i], ch[i + 1], p_drop) for i in range(depth)])
        self.up = nn.ModuleList([Block(ch[i + 1] + ch[i], ch[i], p_drop)
                                 for i in reversed(range(depth))])
        self.out = nn.Conv2d(ch[0], 1, 1)
        nn.init.constant_(self.out.bias, -3.0)      # start dry: wet cells are ~1-4 % of the map

    def forward(self, x):
        skips = [self.inc(x)]
        for d in self.down:
            skips.append(d(F.max_pool2d(skips[-1], 2)))
        h = skips[-1]
        for i, u in enumerate(self.up):
            s = skips[self.depth - 1 - i]
            h = F.interpolate(h, size=s.shape[-2:], mode="bilinear", align_corners=False)
            h = u(torch.cat([h, s], 1))
        return self.out(h).squeeze(1)


def masked_loss(logit, target, valid, dice_w=1.0, bce_w=0.5, pos_weight=None, eps=1.0):
    """Soft-Dice (a differentiable stand-in for CSI, which is IoU) + masked BCE.

    Dice and IoU are monotonically related, so descending soft-Dice is descending 1 - CSI.
    Everything is masked to scoreable cells: permanent water is excluded from the loss for
    the same reason it is excluded from the score.
    """
    v = valid.float()
    p = torch.sigmoid(logit) * v
    t = target.float() * v
    inter = (p * t).sum((-2, -1))
    dice = 1.0 - (2 * inter + eps) / (p.sum((-2, -1)) + t.sum((-2, -1)) + eps)
    bce = F.binary_cross_entropy_with_logits(
        logit, target.float(), reduction="none",
        pos_weight=pos_weight if pos_weight is None else torch.as_tensor(pos_weight, device=logit.device))
    bce = (bce * v).sum((-2, -1)) / v.sum((-2, -1)).clamp(min=1)
    return (dice_w * dice + bce_w * bce).mean()


def count_params(m):
    return sum(p.numel() for p in m.parameters())
