import torch.nn as nn


class Mixup(object):

    def __init__(self, alpha_sampler):
        self.sampler = alpha_sampler

    def __call__(self, x1, x2, y1, y2):
        alpha = self.sampler.rsample().to(x1.device)
        x = alpha * x1 + (1 - alpha) * x2
        y = alpha * y1 + (1 - alpha) * y2
        return x, y