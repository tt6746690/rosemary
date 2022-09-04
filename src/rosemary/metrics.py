import itertools
import numpy as np

from sklearn.metrics import (
    log_loss,
    accuracy_score,
    precision_recall_fscore_support,
    average_precision_score,
    roc_auc_score,
    mean_squared_error,
    brier_score_loss)
from sklearn.metrics.cluster import normalized_mutual_info_score
from sklearn.preprocessing import LabelBinarizer

try:
    from .torch import torch_tensor_to_ndarray
except ImportError:
    torch_tensor_to_ndarray = lambda x: x

from .parallel import joblib_parallel_process


__all__ = [
    'metrics_binary_classification',
    'metrics_multiclass_classification',
    'metrics_clustering',
    'metrics_grounding',
    'np_mask_iou',
    'np_box_to_mask',
    'torch_mask_iou',
    'torch_box_to_mask',
]


def metrics_binary_classification(label, score, threshold=.5, nll_class_weights=None):
    """Metrics for binary classification.
            label           (n_samples,)
            score           (n_samples,)
                after application of sigmoid
    """
    label = torch_tensor_to_ndarray(label)
    score = torch_tensor_to_ndarray(score)
    nll_class_weights = torch_tensor_to_ndarray(nll_class_weights)

    pred = (score > threshold).astype(np.int32)

    metrics = {}
    metrics['N'] = len(label)
    metrics['nll'] = log_loss_fp32(label, score)
    if nll_class_weights is not None:
        metrics['nll_weighted'] = log_loss_fp32(
            label, score, sample_weight=nll_class_weights[label.astype(np.int)])
    metrics['accuracy'] = accuracy_score(label, pred)
    metrics['precision'], metrics['recall'], metrics['f1_score'], _ = precision_recall_fscore_support(
        label, pred, average='macro', zero_division=0)
    metrics['precision_avg'] = average_precision_score(
        label, score, average='macro')
    metrics['auroc'] = roc_auc_score(label, score)
    metrics['mse'] = mean_squared_error(label, pred)
    metrics['brier_score'] = brier_score_loss(label, score)

    return metrics


def metrics_multiclass_classification(label, score, nll_class_weights=None):
    """Metrics for multiclass classification.
            label           (n_samples,)
            score           (n_samples, n_classes)
                after application of softmax
    """
    label = torch_tensor_to_ndarray(label)
    score = torch_tensor_to_ndarray(score)
    nll_class_weights = torch_tensor_to_ndarray(nll_class_weights)

    # (n_samples,)
    pred = np.argmax(score, axis=1)
    # (n_samples, n_classes)
    lb = LabelBinarizer()
    lb.fit(range(score.shape[1]))
    label_onehot = lb.transform(label)

    # Assuming `label` \in [0,1,...,n_classes]
    # Handle cases where `label` has missing classes
    class_labels = np.arange(score.shape[1])

    metrics = {}
    metrics['N'] = len(label)
    metrics['nll'] = log_loss_fp32(label, score, labels=class_labels)
    if nll_class_weights is not None:
        metrics['nll_weighted'] = log_loss_fp32(
            label, score, sample_weight=nll_class_weights[label])
    metrics['accuracy'] = accuracy_score(label, pred)
    metrics['precision'], metrics['recall'], metrics['f1_score'], _ = \
        precision_recall_fscore_support(
            label, pred, average='macro', zero_division=0, labels=class_labels)
    # [0v123, 1v023, 2v013, 3v012]
    if len(np.unique(label)) == len(class_labels):
        metrics['auroc_ovr'] = roc_auc_score(
            label_onehot, score, multi_class='ovr', average=None, labels=class_labels)
        metrics['auroc_ovr_macro'] = np.mean(metrics['auroc_ovr'])
    # [0v123, 01v23, 012v3]
    metrics['auroc_ordinal'] = auc_ordinal(label_onehot, score)
    metrics['auroc_ordinal_macro'] = np.mean(metrics['auroc_ordinal'])
    metrics['mse'] = mean_squared_error(label, pred)
    metrics['mae'] = mean_absolute_error(label, pred)
    metrics['mae_macro'] = np.mean(metrics['mae'])
    metrics['brier_score'] = multiclass_brier_score_loss(label_onehot, score)

    # Convert array-valued outputs to scalar-valued outputs
    # - `auroc_ordinal`, `auroc_ovr`, `mae`
    for i in range(len(metrics['auroc_ovr'])):
        metrics[f'auroc_{i}vr'] = metrics['auroc_ovr'][i]
    for i in range(len(metrics['auroc_ordinal'])):
        lhs = [str(i) for i in range(i+1)]
        rhs = [str(i) for i in range(i+1, len(metrics['auroc_ordinal'])+1)]
        lhs, rhs = ''.join(lhs), ''.join(rhs)
        metrics[f'auroc_ordinal_{lhs}v{rhs}'] = metrics['auroc_ordinal'][i]
    for i in range(len(metrics['mae'])):
        metrics[f'mae_{i}'] = metrics['mae'][i]

    for k in ['auroc_ovr', 'auroc_ordinal', 'mae']:
        metrics.pop(k)

    return metrics


def log_loss_fp32(label, score, **kwargs):
    """When score is `float32`, computing `log(score)` will introduce
            extreme values """
    return log_loss(label, score.astype(np.float64), **kwargs)


def multiclass_brier_score_loss(label_onehot, score):
    return np.mean(np.sum((label_onehot - score)**2, axis=1))


def mean_absolute_error(y_true, y_pred):
    n = max(y_true.max(), y_pred.max())
    mae = []
    for i in range(n+1):
        I = (y_true == i)
        e = np.mean(np.abs(y_true[I] - y_pred[I]))
        mae.append(e)
    return mae


def auc_ordinal(label_onehot, score):
    """`label_onehot, score`    (n_samples, n_classes) """
    def rowwise_reverse_cumsum(a):
        return np.flip(np.flip(a, 1).cumsum(1), 1)

    label_cumsum = rowwise_reverse_cumsum(label_onehot)
    score_cumsum = rowwise_reverse_cumsum(score)

    auc_ordinal = []
    for i in range(1, score.shape[1]):
        auc_ordinal.append(roc_auc_score(
            label_cumsum[:, i], score_cumsum[:, i]))

    return auc_ordinal


def metrics_clustering(label, embedding, ks=[1]):
    """Computes metric for metric learning tasks
        label        (n_samples,)
        embedding    (n_samples, d)
        ks
            Computes recall@k for k in ks
    """
    import faiss

    label = torch_tensor_to_ndarray(label)
    embedding = torch_tensor_to_ndarray(embedding)

    metrics = {}

    # Train K-means to cluster the dataset to #classes clusters
    # then assign embeddings to the centroids of K-means clustering
    kmeans = faiss.Kmeans(embedding.shape[-1],
                          k=len(np.unique(label)),
                          niter=20)
    kmeans.train(embedding)
    centroids = kmeans.centroids
    faiss_index = faiss.IndexFlatL2(centroids.shape[-1])
    faiss_index.add(centroids)
    _, label_kmeans = faiss_index.search(embedding, 1)
    label_kmeans = label_kmeans.squeeze()

    # common metrics for clustering
    metrics['nmi'] = normalized_mutual_info_score(
        label_kmeans, label)
    metrics['f1_score'] = cluster_f1_score(
        label_kmeans, label, embedding, centroids)

    # Compute `max(ks)` nearest neighbors for `embedding` for recall computation
    # `recall@k` is simply the fraction of times that a embedding's `k` closest
    # neighbors contains the true label for that embedding
    I = k_closest(embedding,
                  nq='all',
                  k=int(np.max(ks)+1))
    k_closest_label = label[I[:, 1:]]

    def compute_recallatk(labels, retrieved, k):
        recalled = [
            1 for l, preds in zip(labels, retrieved)
            if l in preds[:k]]
        recallatk = np.sum(recalled)/len(labels)
        return recallatk

    for k in ks:
        metrics[f'recall@{k}'] = compute_recallatk(
            label, k_closest_label, k)

    return metrics


def k_closest(embeddings, nq='all', k=4):
    """Fetch `k` closest neighbor for `nq` randomly sampled
        entries w.r.t. l2 distance.
        embeddings    (nb, d)
            embeddings stacked columnwise
        nq
            If `all`, use embeddings as query
            otherwise sample `nq` queries
        k
            Number of closest neighbors for search
        Returns
            (nq, k)
    """
    import faiss
    embedding = torch_tensor_to_ndarray(embedding)

    nb, d = embeddings.shape

    # Construct Index
    faiss_index = faiss.IndexFlatL2(d)
    faiss_index.add(embeddings)

    # Construct Query
    if nq != 'all':
        query_idx = np.random.choice(nb, size=nq)
        query = embeddings[query_idx]
    else:
        query = embeddings

    # Search
    _, I = faiss_index.search(query, k+1)

    return I


def cluster_f1_score(model_generated_cluster_labels, target_labels, feature_coll, computed_centroids):
    """
    Taken frmo https://github.com/Confusezius/Deep-Metric-Learning-Baselines/blob/master/auxiliaries.py
    NOTE: MOSTLY ADAPTED FROM https://github.com/wzzheng/HDML on Hardness-Aware Deep Metric Learning.
    Args:
        model_generated_cluster_labels: np.ndarray [n_samples x 1], Cluster labels computed on top of data embeddings.
        target_labels:                  np.ndarray [n_samples x 1], ground truth labels for each data sample.
        feature_coll:                   np.ndarray [n_samples x embed_dim], total data embedding made by network.
        computed_centroids:             np.ndarray [num_cluster=num_classes x embed_dim], cluster coordinates
    Returns:
        float, F1-score
    """
    from scipy.special import comb

    d = np.zeros(len(feature_coll))
    for i in range(len(feature_coll)):
        d[i] = np.linalg.norm(
            feature_coll[i, :] - computed_centroids[model_generated_cluster_labels[i], :])

    labels_pred = np.zeros(len(feature_coll))
    for i in np.unique(model_generated_cluster_labels):
        index = np.where(model_generated_cluster_labels == i)[0]
        ind = np.argmin(d[index])
        cid = index[ind]
        labels_pred[index] = cid

    N = len(target_labels)

    # Cluster n_labels
    avail_labels = np.unique(target_labels)
    n_labels = len(avail_labels)

    # Count the number of objects in each cluster
    count_cluster = np.zeros(n_labels)
    for i in range(n_labels):
        count_cluster[i] = len(np.where(target_labels == avail_labels[i])[0])

    # Build a mapping from item_id to item index
    keys = np.unique(labels_pred)
    num_item = len(keys)
    values = range(num_item)
    item_map = dict()
    for i in range(len(keys)):
        item_map.update([(keys[i], values[i])])

    # Count the number of objects of each item
    count_item = np.zeros(num_item)
    for i in range(N):
        index = item_map[labels_pred[i]]
        count_item[index] = count_item[index] + 1

    # Compute True Positive (TP) plus False Positive (FP) count
    tp_fp = 0
    for k in range(n_labels):
        if count_cluster[k] > 1:
            tp_fp = tp_fp + comb(count_cluster[k], 2)

    # Compute True Positive (TP) count
    tp = 0
    for k in range(n_labels):
        member = np.where(target_labels == avail_labels[k])[0]
        member_ids = labels_pred[member]

        count = np.zeros(num_item)
        for j in range(len(member)):
            index = item_map[member_ids[j]]
            count[index] = count[index] + 1

        for i in range(num_item):
            if count[i] > 1:
                tp = tp + comb(count[i], 2)

    # Compute  False Positive (FP) count
    fp = tp_fp - tp

    # Compute False Negative (FN) count
    count = 0
    for j in range(num_item):
        if count_item[j] > 1:
            count = count + comb(count_item[j], 2)
    fn = count - tp

    # compute F measure
    beta = 1
    P = tp / (tp + fp)
    R = tp / (tp + fn)
    F1 = (beta*beta + 1) * P * R / (beta*beta * P + R)

    return F1

 
def metrics_grounding(label, score, image_shape, device='cpu', ts=[.5], n_jobs=1, reduce=True):
    """Computes metric for detection tasks
        label        (n_samples, 4)
            assume label comes in the form of bounding boxes.
        score        (n_samples, d)
            similarity mapping.
        image_shape  (n_samples, 2)
            in (h, w) format
    """
    from .jpt import jpt_in_notebook

    if not isinstance(label, list):
        raise ValueError('`label` should be List<Tensor|ndarray>')
    if not isinstance(score, list):
        raise ValueError('`score` should be List<Tensor|ndarray>')
    if device != 'cpu':
        raise ValueError("Only `device='cpu'` is implemented.")

    if isinstance(label, list):
        label = [torch_tensor_to_ndarray(x) for x in label]
    if isinstance(score, list):
        score = [torch_tensor_to_ndarray(x) for x in score]
        
    if label[0].shape[-1] == 4: # bbox -> mask
        label = [
            np_box_to_mask(box, shape)
            for box, shape in zip(label, image_shape)]
    
    metrics = {}
    metrics['N'] = len(label)

    def metrics_grounding_step_fn(args):
        m, s, ts = args
        ious, miou = np_mask_ious(m, s, ts)
        cnr = np_contrast_to_noise_ratio(m, s)
        return {'IoUs': ious, 'mIoU': miou, 'cnr': cnr}

    results = joblib_parallel_process(
        fn=metrics_grounding_step_fn,
        iterable=zip(label, score, itertools.repeat(ts)),
        n_jobs=n_jobs,
        prefer='threads',
        use_tqdm=True if jpt_in_notebook() else False)
    IoUs = [x['IoUs'] for x in results]
    mIoU = [x['mIoU'] for x in results]
    cnr  = [x['cnr']  for x in results]

    for i, t in enumerate(ts):
        metrics[f"IoU@{t}"] = [x[i] for x in IoUs]
    metrics["mIoU"] = mIoU
    metrics["cnr"] = cnr

    if reduce:
        for i, t in enumerate(ts):
            metrics[f"IoU@{t}"] = np.mean(metrics[f"IoU@{t}"])
        metrics["mIoU"] = np.mean(metrics["mIoU"])
        metrics["cnr"] = np.mean(metrics["cnr"])
    
    return metrics


def np_contrast_to_noise_ratio(label, score):
    """Compute CNR in https://arxiv.org/abs/2204.09817 
        Here `label` is ground truth mask converted from bbox
        and `score` is predicted similarity values. 
    This metric doesn't require thresholding `score`. 
    Yet to handle situation when `label` has `np.nan`
    """
    A, B = score[label],score[~label]
    mu_A, var_A = np.nanmean(A), np.nanvar(A)
    mu_B, var_B = np.nanmean(B), np.nanvar(B)
    cnr = np.abs(mu_A-mu_B) / (var_A+var_B)**.5
    return cnr


def np_mask_iou(label, score, t=.5, ϵ=1e-6):
    """`label`, `score` are 2d images. 
        20x faster than `np_mask_iou`. """
    pred = score >= t
    label, pred = label.flatten(), pred.flatten()
    intersection = np.logical_and(label, pred).sum()
    union = np.logical_or(label, pred).sum()
    iou = (intersection) / (union + ϵ)
    return iou


def np_mask_ious(label, score, ts=[.5]):
    ious = []
    for t in ts:
        ious.append(np_mask_iou(label, score, t))
    return ious, np.mean(ious)


def np_box_to_mask(box, image_shape):
    """Convert `box` to `mask` given `image_shape` (h,w)."""
    if box.ndim != 2:
        raise ValueError('`boxes` should be (#boxes, 4)')
    mask = np.zeros(image_shape, dtype=bool)
    for b in box.astype(int): # potentially rounding error.
        mask[b[1]:b[3],b[0]:b[2]] = True
    return mask


def torch_ndarray_to_tensor(x):
    import torch
    if isinstance(x, np.ndarray):
        return torch.from_numpy(x)
    return x


def torch_box_to_mask(box, image_shape):
    """Convert a list of boxes to a single mask of `image_shape`
            boxes    (#boxes, 4)

        `image_shape` Union[List[int], torch.Tensor]
    """
    import torch
    if box.ndim != 2:
        raise ValueError('`boxes` should be (#boxes, 4)')
    mask = torch.zeros(
        *image_shape, dtype=torch.bool, device=box.device)
    for b in box.to(int): # potentially rounding error.
        mask[b[1]:b[3],b[0]:b[2]] = True
    return mask


def torch_mask_iou(label, score, t=.5, ϵ=1e-6):
    import torch
    pred = score >= t
    label, pred = label.flatten(), pred.flatten()
    intersection = torch.logical_and(label, pred).sum()
    union = torch.logical_or(label, pred).sum()
    iou = (intersection) / (union + ϵ)
    return iou.item()


def torch_mask_ious(label, score, ts=[.5]):
    ious = []
    for t in ts:
        ious.append(torch_mask_iou(label, score, t))
    return ious, np.mean(ious)