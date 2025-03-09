import numpy as np
from scipy.spatial import ConvexHull
from shapely.geometry import Polygon
from scipy.spatial.transform import Rotation as R
import argparse
import glob 
import tqdm

from open3d.ml.contrib import iou_3d_cpu as iou_3d

# def iou_3d(gt_boxes, pred_boxes):
#     out_score = np.zeros((len(gt_boxes), len(pred_boxes)))
#     for i, gt_box in enumerate(gt_boxes):
#         for j, pred_box in enumerate(pred_boxes):
#             out_score[i, j] = iou_3d_single(gt_box, pred_box) 
#     return out_score

# def iou_3d_single(box1, box2):
#     corners_1 = compute_bbox_corners(*box1)
#     corners_2 = compute_bbox_corners(*box2)

#     # fix aspect ratio of plt
#     p = Polygon(corners_1[:2, [2,3,7,6]].T)
#     q = Polygon(corners_2[:2, [2,3,7,6]].T)
 
#     area1 = p.area
#     area2 = q.area
#     inter_area = p.intersection(q).area

#     iou_2d = inter_area/(area1+area2-inter_area)
#     zmax = min(corners_1[2].max(),corners_2[2].max())
#     zmin = max(corners_1[2].min(),corners_2[2].min())

#     inter_vol = inter_area * max(0.0, zmax-zmin)

#     vol1 = box3d_vol(corners_1.T)
#     vol2 = box3d_vol(corners_2.T)
#     iou = inter_vol / (vol1 + vol2 - inter_vol)

#     return iou


def poly_area(x,y):
    """ Ref: http://stackoverflow.com/questions/24467972/calculate-area-of-polygon-given-x-y-coordinates """
    return 0.5*np.abs(np.dot(x,np.roll(y,1))-np.dot(y,np.roll(x,1)))


def convex_hull_intersection(p1, p2):
    """ Compute area of two convex hull's intersection area.
        p1,p2 are a list of (x,y) tuples of hull vertices.
        return a list of (x,y) for the intersection and its volume
    """
    inter_p = polygon_clip(p1,p2)
    if inter_p is not None:
        hull_inter = ConvexHull(inter_p)
        return inter_p, hull_inter.volume
    else:
        return None, 0.0  
    
def box3d_vol(corners):
    ''' corners: (8,3) no assumption on axis direction '''
    a = np.sqrt(np.sum((corners[0,:] - corners[1,:])**2))
    b = np.sqrt(np.sum((corners[1,:] - corners[2,:])**2))
    c = np.sqrt(np.sum((corners[0,:] - corners[4,:])**2))
    return a*b*c

def compute_bbox_corners(x,y,z,w,l,h,r) -> np.ndarray:
    """
    Returns the bounding box corners.
    :param wlh_factor: Multiply w, l, h by a factor to scale the box.
    :return: <np.float: 3, 8>. First four corners are the ones facing forward.
        The last four are the ones facing backwards.
    """

    # 3D bounding box corners. (Convention: x points forward, y to the left, z up.)
    x_corners = l / 2 * np.array([1,  1,  1,  1, -1, -1, -1, -1])
    y_corners = w / 2 * np.array([1, -1, -1,  1,  1, -1, -1,  1])
    z_corners = h / 2 * np.array([1,  1, -1, -1,  1,  1, -1, -1])
    corners = np.vstack((x_corners, y_corners, z_corners))

    # Rotate
    r_matrix = R.from_euler('zyx',[ r, 0, 0] , degrees=False).as_matrix()
    corners = np.dot(r_matrix, corners)

    # Translate
    corners[0, :] = corners[0, :] + x
    corners[1, :] = corners[1, :] + y
    corners[2, :] = corners[2, :] + z

    return corners


def polygon_clip(subjectPolygon, clipPolygon):
   """ Clip a polygon with another polygon.

   Ref: https://rosettacode.org/wiki/Sutherland-Hodgman_polygon_clipping#Python

   Args:
     subjectPolygon: a list of (x,y) 2d points, any polygon.
     clipPolygon: a list of (x,y) 2d points, has to be *convex*
   Note:
     **points have to be counter-clockwise ordered**

   Return:
     a list of (x,y) vertex point for the intersection polygon.
   """
   def inside(p):
      return(cp2[0]-cp1[0])*(p[1]-cp1[1]) > (cp2[1]-cp1[1])*(p[0]-cp1[0])
 
   def computeIntersection():
      dc = [ cp1[0] - cp2[0], cp1[1] - cp2[1] ]
      dp = [ s[0] - e[0], s[1] - e[1] ]
      n1 = cp1[0] * cp2[1] - cp1[1] * cp2[0]
      n2 = s[0] * e[1] - s[1] * e[0] 
      n3 = 1.0 / (dc[0] * dp[1] - dc[1] * dp[0])
      return [(n1*dp[0] - n2*dc[0]) * n3, (n1*dp[1] - n2*dc[1]) * n3]
 
   outputList = subjectPolygon
   cp1 = clipPolygon[-1]
 
   for clipVertex in clipPolygon:
      cp2 = clipVertex
      inputList = outputList
      outputList = []
      s = inputList[-1]
 
      for subjectVertex in inputList:
         e = subjectVertex
         if inside(e):
            if not inside(s):
               outputList.append(computeIntersection())
            outputList.append(e)
         elif inside(s):
            outputList.append(computeIntersection())
         s = e
      cp1 = cp2
      if len(outputList) == 0:
          return None
   return(outputList)

def filter_data(data, labels, diffs=None):
    """Filters the data to fit the given labels and difficulties.
    Args:
        data (dict): Dictionary with the data (as numpy arrays).
            {
                'label':      [...], # expected
                'difficulty': [...]  # if diffs not None
                ...
            }
        labels (number[]): List of labels which should be maintained.
        difficulties (number[]): List of difficulties which should maintained.
            (optional)

    Returns:
        Tuple with dictionary with same as format as input, with only the given labels
        and difficulties and the indices.
    """
    cond = np.any([data['label'] == label for label in labels], axis=0)
    if diffs is not None and 'difficulty' in data:
        dcond = np.any([
            np.all([data['difficulty'] >= 0, data['difficulty'] <= diff],
                   axis=0) for diff in diffs
        ],
                       axis=0)
        cond = np.all([cond, dcond], axis=0)
    idx = np.where(cond)[0]

    result = {}
    for k in data:
        result[k] = data[k][idx]
    return result, idx


def precision_3d(pred,
                 target,
                 classes=[0],
                 difficulties=[0],
                 min_overlap=[0.5],
                 bev=True,
                 similar_classes={}):
    """Computes precision quantities for each predicted box.
    Args:
        pred (dict): Dictionary with the prediction data (as numpy arrays).
            {
                'bbox':       [...],
                'label':      [...],
                'score':      [...],
                'difficulty': [...],
                ...
            }
        target (dict): Dictionary with the target data (as numpy arrays).
            {
                'bbox':       [...],
                'label':      [...],
                'difficulty': [...],
                ...
            }
        classes (number[]): List of classes which should be evaluated.
            Default is [0].
        difficulties (number[]): List of difficulties which should evaluated.
            Default is [0].
        min_overlap (number[]): Minimal overlap required to match bboxes.
            One entry for each class expected. Default is [0.5].
        bev (boolean): Use BEV IoU (else 3D IoU is used).
            Default is True.
        similar_classes (dict): Assign classes to similar classes that were not part of the training data so that they are not counted as false negatives.
            Default is {}.

    Returns:
        A tuple with a list of detection quantities
        (score, true pos., false. pos) for each box
        and a list of the false negatives.
    """
    sim_values = list(similar_classes.values())

    # pre-filter data, remove unknown classes
    pred = filter_data(pred, classes)[0]
    target = filter_data(target, classes + sim_values)[0]

    
    overlap = iou_3d(pred['bbox'].astype(np.float32),
                        target['bbox'].astype(np.float32))

    detection = np.zeros(
        (len(classes), len(difficulties), len(pred['bbox']), 3))
    fns = np.zeros((len(classes), len(difficulties), 1), dtype="int64")
    for i, label in enumerate(classes):
        # filter only with label
        pred_label, pred_idx_l = filter_data(pred, [label])
        target_label, target_idx_l = filter_data(
            target, [label, similar_classes.get(label)])
        overlap_label = overlap[pred_idx_l][:, target_idx_l]
        for j, diff in enumerate(difficulties):
            # filter with difficulty
            pred_idx = filter_data(pred_label, [label], [diff])[1]
            target_idx = filter_data(target_label, [label], [diff])[1]

            if len(pred_idx) > 0:
                # no matching gt box (filtered preds vs all targets)
                fp = np.all(overlap_label[pred_idx] < min_overlap[i],
                            axis=1).astype("float32")

                # identify all matches (filtered preds vs filtered targets)
                match_cond = np.any(
                    overlap_label[pred_idx][:, target_idx] >= min_overlap[i],
                    axis=-1)
                tp = np.zeros((len(pred_idx),))

                # all matches first fp
                fp[np.where(match_cond)] = 1

                # only best match can be tp
                max_idx = np.argmax(overlap_label[:, target_idx], axis=0)
                max_cond = [idx in max_idx for idx in pred_idx]
                match_cond = np.all([max_cond, match_cond], axis=0)
                tp[match_cond] = 1
                fp[match_cond] = 0

                # no matching pred box (all preds vs filtered targets)
                fns[i, j] = np.sum(
                    np.all(overlap_label[:, target_idx] < min_overlap[i],
                           axis=0))
                detection[i, j, [pred_idx]] = np.stack(
                    [pred_label['score'][pred_idx], tp, fp], axis=-1)
            else:
                fns[i, j] = len(target_idx)

    return detection, fns


def sample_thresholds(scores, gt_cnt, sample_cnt=41):
    """Computes equally spaced sample thresholds from given scores

    Args:
        scores (list): list of scores
        gt_cnt (number): amount of gt samples
        sample_cnt (number): amount of samples
            Default is 41.

    Returns:
        Returns a list of equally spaced samples of the input scores.
    """
    scores = np.sort(scores)[::-1]
    current_recall = 0
    thresholds = []
    for i, score in enumerate(scores):
        l_recall = (i + 1) / gt_cnt
        r_recall = (i + 2) / gt_cnt if i < (len(scores) - 1) else l_recall
        if (((r_recall - current_recall) < (current_recall - l_recall)) and
            (i < (len(scores) - 1))):
            continue
        thresholds.append(score)
        current_recall += 1 / (sample_cnt - 1.0)
    return thresholds


def mAP(pred,
        target,
        classes=[0],
        difficulties=[0],
        min_overlap=[0.5],
        bev=True,
        samples=41,
        similar_classes={}):
    """Computes mAP of the given prediction (11-point interpolation).

    Args:
        pred (dict): List of dictionaries with the prediction data (as numpy arrays).
            {
                'bbox':       [...],
                'label':      [...],
                'score':      [...],
                'difficulty': [...]
            }[]
        target (dict): List of dictionaries with the target data (as numpy arrays).
            {
                'bbox':       [...],
                'label':      [...],
                'difficulty': [...]
            }[]
        classes (number[]): List of classes which should be evaluated.
            Default is [0].
        difficulties (number[]): List of difficulties which should evaluated.
            Default is [0].
        min_overlap (number[]): Minimal overlap required to match bboxes.
            One entry for each class expected. Default is [0.5].
        bev (boolean): Use BEV IoU (else 3D IoU is used).
            Default is True.
        samples (number): Count of used samples for mAP calculation.
            Default is 41.
        similar_classes (dict): Assign classes to similar classes that were not part of the training data so that they are not counted as false negatives.
            Default is {}.

    Returns:
        Returns the mAP for each class and difficulty specified.
    """
    if len(min_overlap) != len(classes):
        assert len(min_overlap) == 1
        min_overlap = min_overlap * len(classes)
    assert len(min_overlap) == len(classes)

    cnt = 0
    box_cnts = [0]
    for p in pred:
        cnt += len(filter_data(p, classes)[1])
        box_cnts.append(cnt)

    gt_cnt = np.zeros((len(classes), len(difficulties)))
    for i, c in enumerate(classes):
        for j, d in enumerate(difficulties):
            for t in target:
                gt_cnt[i, j] += len(filter_data(t, [c], [d])[1])

    detection = np.zeros((len(classes), len(difficulties), box_cnts[-1], 3))
    fns = np.zeros((len(classes), len(difficulties), 1), dtype='int64')
    for i in range(len(pred)):
        d, f = precision_3d(pred=pred[i],
                            target=target[i],
                            classes=classes,
                            difficulties=difficulties,
                            min_overlap=min_overlap,
                            bev=bev,
                            similar_classes=similar_classes)
        detection[:, :, box_cnts[i]:box_cnts[i + 1]] = d
        fns += f

    mAP = np.zeros((len(classes), len(difficulties), 1))
    if samples <= 0:
        # No samples to compute mAP against, so all results are zero.
        return mAP

    for i in range(len(classes)):
        for j in range(len(difficulties)):
            det = detection[i, j, np.argsort(-detection[i, j, :, 0])]

            #gt_cnt = np.sum(det[:,1]) + fns[i, j]
            thresholds = sample_thresholds(det[np.where(det[:, 1] > 0)[0], 0],
                                           gt_cnt[i, j], samples)
            if len(thresholds) == 0:
                # No predictions met cutoff thresholds, skipping AP computation to avoid NaNs.
                continue

            prec = np.zeros((len(thresholds),))
            for ti in range(len(thresholds))[::-1]:
                d = det[np.where(det[:, 0] >= thresholds[ti])]
                tp_acc = np.sum(d[:, 1])
                fp_acc = np.sum(d[:, 2])
                if (tp_acc + fp_acc) > 0:
                    prec[ti] = tp_acc / (tp_acc + fp_acc)
                prec[ti] = np.max(prec[ti:], axis=-1)

            if len(prec[::4]) < int(samples / 4 + 1):
                mAP[i, j] = np.sum(prec) / len(prec) * 100
            else:
                mAP[i, j] = np.sum(prec[::4]) / int(samples / 4 + 1) * 100

    return mAP


if __name__=='__main__':
    all_map = []
    arg = argparse.ArgumentParser()
    arg.add_argument('--label_path', type=str, required=True)
    arg.add_argument('--output_path',type=str, required=True)
    args = arg.parse_args()
    
    gt_paths = glob.glob(f"{args.label_path}/*.txt")
    pred_paths = glob.glob(f"{args.output_path}/*.txt")

    assert len(gt_paths) == len(pred_paths) , "The output and ground truth numbers are not equal"
    for gt_path, pred_path in tqdm.tqdm(zip(gt_paths, pred_paths)):
        gt_boxes = np.loadtxt(gt_path,dtype=str)[:, 1:]
        pred_boxes = np.loadtxt(pred_path,dtype=str)[:, 1:]

        target = [{
            'bbox' : gt_boxes.astype(np.float32),
            'label': np.zeros(len(gt_boxes)),
            'score': np.ones(len(gt_boxes)),
            'difficult': np.zeros(len(gt_boxes))
        }]

        pred = [{
            'bbox' : pred_boxes.astype(np.float32),
            'label': np.zeros(len(pred_boxes)),
            'score': np.ones(len(pred_boxes)),
            'difficult': np.zeros(len(pred_boxes))
        }]

        map_score = mAP(target, pred)
        all_map.append(map_score)

    print(np.mean(all_map))