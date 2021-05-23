import numpy as np


def bb_intersection_over_union(box1: np.ndarray, box2: np.ndarray):
    """

    calculate intersection over union cover percent

    :param box1: box1 with shape (N,4) or (N,2,2) or (2,2) or (4,). first shape is preferred
    :param box2: box2 with shape (N,4) or (N,2,2) or (2,2) or (4,). first shape is preferred
    
    :return: IoU ratio if intersect, else 0

    """

    # first unify all boxes to shape (N,4)

    if box1.shape[-1] == 2 or len(box1.shape) == 1:
        box1 = box1.reshape(1, 4) if len(box1.shape) <= 2 else box1.reshape(box1.shape[0], 4)
    if box2.shape[-1] == 2 or len(box2.shape) == 1:
        box2 = box2.reshape(1, 4) if len(box2.shape) <= 2 else box2.reshape(box2.shape[0], 4)
    point_num = max(box1.shape[0], box2.shape[0])
    b1p1, b1p2, b2p1, b2p2 = box1[:, :2], box1[:, 2:], box2[:, :2], box2[:, 2:]

    # mask that eliminates non-intersecting matrices

    base_mat = np.ones(shape=(point_num,))
    base_mat *= np.all(np.greater(b1p2 - b2p1, 0), axis=1)
    base_mat *= np.all(np.greater(b2p2 - b1p1, 0), axis=1)

    # I area

    intersect_area = np.prod(np.minimum(b2p2, b1p2) - np.maximum(b1p1, b2p1), axis=1)
    # U area

    union_area = np.prod(b1p2 - b1p1, axis=1) + np.prod(b2p2 - b2p1, axis=1) - intersect_area
    # IoU

    intersect_ratio = intersect_area / union_area

    return base_mat * intersect_ratio