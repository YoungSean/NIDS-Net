import numpy as np

def nms(masks, scores, thresh):

    n = masks.shape[0]
    inters = np.zeros((n, n), dtype=np.float32)
    for i in range(n):
        for j in range(i, n):
            inters[i, j] = np.sum(np.multiply(masks[i], masks[j]))
            inters[j, i] = inters[i, j]

    areas = np.diag(inters)
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        inter = inters[i, order[1:]]
        ovr = inter / (areas[i] + areas[order[1:]] - inter)
        inds = np.where(ovr <= thresh)[0]
        order = order[inds + 1]

    index = np.argsort(areas[keep]).astype(np.int32)
    return np.array(keep)[index]