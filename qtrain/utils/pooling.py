import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiSoftmax(nn.Module):
    """Multi softmax for given features."""

    def __init__(self, num_ftrs, num_classes):
        super().__init__()
        self.num_ftrs = num_ftrs
        self.num_classes = num_classes-1

        for i in range(self.num_classes):
            self.add_module(f"fc_{i}", nn.Linear(self.num_ftrs, 2))

    def _get_fc(self, i):
        return getattr(self, f"fc_{i}")

    def forward(self, ftrs):
        scores = [self._get_fc(i)(ftrs) for i in range(self.num_classes)]
        return scores


class MultiConv1x1(nn.Module):
    """Multi softmax for given features."""

    def __init__(self, num_ftrs, num_classes):
        super().__init__()
        self.num_ftrs = num_ftrs
        self.num_classes = num_classes

        for i in range(self.num_classes):
            self.add_module(f"conv1x1_{i}", nn.Conv2d(self.num_ftrs, 2, kernel_size=1))

    def _get_conv(self, i):
        return getattr(self, f"conv1x1_{i}")

    def forward(self, ftrs):
        scores = [self._get_conv(i)(ftrs) for i in range(self.num_classes)]
        return scores


def min_max(a):
    """Fusion method.

    Args:
        a: tensor or variable of size (b, 2, n)
    """
    assert (a.size(1) == 2) and (a.ndimension() == 3)
    a = a - a.mean(dim=1).unsqueeze(1)

    b = a.mean(dim=2).clone()
    b[:, 0] = a.min(dim=2)[0][:, 0]
    b[:, 1] = a.max(dim=2)[0][:, 1]
    return b


def soft_attention(a):
    # TODO: deprecate
    """Soft attention absed based on values itself

    Args:
        a: tensor or variable of size (b, 2, n)
    """
    assert a.ndimension() == 3
    a = a - a.mean(dim=1).unsqueeze(1)

    w = F.softmax(a[:, 1, :], dim=1)
    w = w.view(-1, 1, w.size(1))
    s = w * a
    return s.sum(dim=2)


def soft_pooling(a, beta=1):
    """Soft attention based pooling based on values itself

    Args:
        a: tensor or variable of size (b, 2, n1, n2, ...). Pooling is done by
            flattening n1, n2, ..
        beta: postive number. very large then this pooling is same as max
            pooling. 0, then this is avg pooling.
    """
    batchsize, ftrs, *shp = a.shape
    assert ftrs == 2
    a = a - a.mean(dim=1).unsqueeze(1)

    a_flat = a.view(batchsize, ftrs, -1)
    w_flat = F.softmax(beta * a_flat[:, 1], dim=1).unsqueeze(1)

    s_flat = (w_flat * a_flat).sum(dim=2)
    return s_flat


def soft_min_max_attention(a):
    # TODO: deprecate
    """Soft attention absed based on values itself

    Args:
        a: tensor or variable of size (b, 2, n)
    """
    assert a.ndimension() == 3
    a = a - a.mean(dim=1).unsqueeze(1)

    w_pos = F.softmax(a[:, 1, :], dim=1)
    w_neg = w_pos  # F.softmin(a[:, 0 ,:], dim=1)

    w_pos = w_pos.view(-1, 1, w_pos.size(1))
    w_neg = w_neg.view(-1, 1, w_neg.size(1))

    s_pos = w_pos * a[:, 1, :].contiguous().view(-1, 1, a.size(2))
    s_neg = w_neg * a[:, 0, :].contiguous().view(-1, 1, a.size(2))

    s = torch.cat([s_neg, s_pos], dim=1)

    return s.sum(dim=2)


def soft_artifact_attention(a, artifact_probs, a_thresh=0.38):
    # here remove the artifact slices beofre doing soft min max
    """Soft attention based on values itself

    Args:
        a: tensor or variable of size (b, 2, n)
        artifact_probs: tensor of size (b, n)
    """
    orig_shape = a.shape
    artifacts = (artifact_probs > a_thresh).view(-1)
    a = a.transpose(1, 2).view(-1, 2)
    a[artifacts, 0] = 1e6
    a[artifacts, 1] = -1e6
    a = a.view(orig_shape[0], -1, 2).transpose(1, 2)
    return soft_attention(a)


def _flatten(a):
    """Convert (b, 2, ...) to (b, 2, x)."""
    batchsize, ftrs, *shp = a.shape
    assert ftrs == 2
    a = a - a.mean(dim=1).unsqueeze(1)

    a_flat = a.view(batchsize, ftrs, -1)
    return a_flat


def _lse_max(a, r=20, dim=0):
    log_n = torch.log(torch.tensor(float(a.shape[dim])))
    return 1 / r * (torch.logsumexp(a * r, dim=dim) - log_n)


def lse_pooling(a, beta=20):
    a = _flatten(a)
    s_pos = _lse_max(a[:, 1], r=beta, dim=1)
    s_neg = -s_pos

    s = torch.stack([s_neg, s_pos], dim=1)
    return s