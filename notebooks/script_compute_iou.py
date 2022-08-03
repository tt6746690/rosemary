import timeit
import math
import numpy as np

import torch
import torchmetrics

from rosemary import np_mask_iou, torch_mask_iou

def np_mask_iou_sklearn(label, score, t=.5):
    """100x slower than `np_mask_iou`, probably due to having to 
        construct confusion matrix first. """
    from sklearn.metrics import jaccard_score
    label, score = label.squeeze(), score.squeeze()
    label, score = label.flatten(), score.flatten()
    # Don't need to np.nan -> 0. since thresholding nan values.
    # score = np.nan_to_num(score, nan=0.)
    pred = score >= t
    pred = pred.astype(label.dtype)
    iou = jaccard_score(label, pred, pos_label=1, average='binary')
    return iou


def np_iou_from_confmat(label, score, t=.5, ϵ=1e-5):
    from sklearn.metrics import confusion_matrix
    label, score = label.squeeze(), score.squeeze()
    pred = score >= t
    confmat = confusion_matrix(label, pred)
    D = np.diag(confmat)
    A = confmat.sum(1)
    B = confmat.sum(0)
    jaccard = D/(A+B-D+ϵ)
    return jaccard[1]


label = torch.tensor([[0,1,1,1,0,0]]).to(torch.bool)
score =  torch.tensor([[0,.6,.9,.6,.4,0]])
score[:, [0]] = float('nan')

assert(label.dtype == torch.bool)
assert(score.dtype == torch.float32)

iou_fns = {
    'torchmetrics': lambda x, y, t: \
        torchmetrics.functional.jaccard_index(
            y, x, average='none', num_classes=2, threshold=t)[1].item(),
    'np_mask_iou': lambda x, y, t: \
        np_mask_iou(x.numpy(), y.numpy(), t=t),
    'np_mask_iou_jaccard': lambda x, y, t: \
        np_mask_iou_sklearn(x.numpy(), y.numpy(), t=t),
    'np_mask_iou_confmat': lambda x, y, t: \
        np_iou_from_confmat(x.numpy(), y.numpy(), t=t),
    'torch_mask_iou': lambda x, y, t: \
        torch_mask_iou(x, y, t),
}

t = .5
print('Time:')
for k, fn in iou_fns.items():
    stmt = lambda: fn(label, score, t)
    timer = timeit.Timer(stmt)
    number, time_taken = timer.autorange()
    time_in_ms = time_taken/number*1e6
    print(f'{k:20}:\t{time_in_ms:8.3f} μs')

print('\nValue:')
for t in [.25,.5,.75,]:
    print(t)
    stmt = lambda: fn(label, score, t)
    vs = []
    for k, fn in iou_fns.items():
        v = stmt()
        vs.append(v)
        print(f"\t{k:20}:\t{v:.5f}")
    cond = [math.isclose(vs[0], v, abs_tol=1e-5) for v in vs]
    if not cond:
        print(f'Not Approximately Equal: {vs}')
    