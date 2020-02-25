import os
import torch
import numpy as np
import operator
from sklearn.metrics import average_precision_score
from functools import partial

def multi_apply(func, *args, **kwargs):
    pfunc = partial(func, **kwargs) if kwargs else func
    map_results = map(pfunc, *args)
    return tuple(map(list, zip(*map_results)))

def distance2bbox(points, distance, max_shape=None):
    """Decode distance prediction to bounding box.
    Args:
        points (Tensor): Shape (n).
        distance (Tensor): Distance from the given point to 2
            boundaries (front, back).
        max_shape (tuple): Shape of the video.
    Returns:
        Tensor: Decoded bboxes.
    """
    s = points[:, 0] - distance[:, 0]
    e = points[:, 0] + distance[:, 1]
   
    if max_shape is not None:
        s = s.clamp(min=0, max=max_shape - 1)
        e = e.clamp(min=0, max=max_shape - 1)
    return torch.stack([s, e], -1)

def iou_with_anchors(anchors_min, anchors_max, box_min, box_max):
    """Compute jaccard score between a box and the anchors.
    """
    len_anchors = anchors_max - anchors_min
    int_xmin = np.maximum(anchors_min, box_min)
    int_xmax = np.minimum(anchors_max, box_max)
    inter_len = np.maximum(int_xmax - int_xmin, 0.)
    union_len = len_anchors - inter_len + box_max - box_min
    # print inter_len,union_len
    jaccard = np.divide(inter_len, union_len)
    return jaccard


def ioa_with_anchors(anchors_min, anchors_max, box_min, box_max):
    """Compute intersection between score a box and the anchors.
    """
    len_anchors = anchors_max - anchors_min
    int_xmin = np.maximum(anchors_min, box_min)
    int_xmax = np.minimum(anchors_max, box_max)
    inter_len = np.maximum(int_xmax - int_xmin, 0.)
    scores = np.divide(inter_len, len_anchors)
    return scores


def sigmoid(X):
    # map [0,1] -> [0.5,0.73] (almost linearly) ([-1, 0] -> [0.26, 0.5])
    return 1.0 / (1.0 + np.exp(-1.0 * X))


def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def area_of(left, right) -> torch.Tensor:
    """Compute the areas of rectangles given two corners.
    Args:
        left (...): start corner.
        right (...): end corner.
        the same size
        
    Returns:
        area (N): return the area.
    """
    hw = torch.clamp(right - left, min=0.0)  # clamp把负值变成0
    return hw

def iou_of(boxes0, boxes1, eps=1e-5):
    """Compute the iou_of overlap of two sets of corner boxes.  The iou_of overlap
    is simply the intersection over union of two boxes.  Here we operate on
    ground truth boxes and default boxes.
    
    E.g.:
        A ∩ B / A ∪ B = A ∩ B / (area(A) + area(B) - A ∩ B)
        
    Args:
        boxes0: (tensor) Ground truth bounding boxes, Shape: [num_objects,2]
        boxes1: (tensor) Prior boxes from priorbox layers, Shape: [num_priors,2]
        
    Return:
        iou_of overlap: (tensor) Shape: [box_a.size(0), box_b.size(0)]
    """
    overlap_left_top = torch.max(
        boxes0[..., 0], boxes1[..., 0])  # (b, a) 把boxes1的每一个与boxes0相比
    overlap_right_bottom = torch.min(boxes0[..., 1],
                                     boxes1[..., 1])  # (b,a)

    overlap_area = area_of(overlap_left_top, overlap_right_bottom)
    area0 = area_of(boxes0[..., 0], boxes0[..., 0])
    area1 = area_of(boxes1[..., 1], boxes1[..., 1])
    return overlap_area / (area0 + area1 - overlap_area + eps)


def hard_nms(box_scores, iou_threshold, top_k=-1, candidate_size=200):
    """
    Args:
        box_scores(tensor): shape: `(N, 5)`: boxes in corner-form and probabilities.
        iou_threshold(float): intersection over union threshold.
        top_k(int): keep top_k results. If k <= 0, keep all the results.
        candidate_size(int): only consider the candidates with the highest scores.
        
    Returns:
         picked: a list of indexes of the kept boxes
    """
    scores = box_scores[:, -1]
    boxes = box_scores[:, :-1]
    picked = []
    _, indexes = scores.sort(descending=True)
    indexes = indexes[:candidate_size]  # 只选这么多参与比较，如果实际多比这个阈值小，size不变
    while len(indexes) > 0:
        current = indexes[0]
        picked.append(current.item())
        if 0 < (top_k == len(picked)) or len(indexes) == 1:
            break
        current_box = boxes[current, :]
        indexes = indexes[1:]
        rest_boxes = boxes[indexes, :]
        iou = iou_of(
            rest_boxes,
            current_box.unsqueeze(0),
        )
        # 过滤掉和现有的confidence最大的box重叠的box
        indexes = indexes[iou <= iou_threshold]
    return box_scores[picked, :]

def multiclass_nms(multi_bboxes,
                   multi_scores,
                   score_thr,
                   nms_thr,
                   max_num=-1,
                   score_factors=None,
                   pre_nms=100):
    """NMS.
    Args:
        multi_bboxes (Tensor): shape (n, #class*4) or (n, 4)
        multi_scores (Tensor): shape (n, #class)
        score_thr (float): bbox threshold, bboxes with scores lower than it
            will not be considered.
        nms_thr (float): NMS IoU threshold
        max_num (int): if there are more than max_num bboxes after NMS,
            only top max_num will be kept.
        score_factors (Tensor): The factors multiplied to scores before
            applying NMS
    Returns:
        tuple: (bboxes, labels), tensors of shape (k, 5) and (k, 1). Labels
            are 0-based.
    """
    print(multi_scores.shape)
    num_classes = multi_scores.shape[1]

    bboxes, labels = [], []

    for i in range(1, num_classes):
        cls_inds = multi_scores[:, i] > score_thr
        if not cls_inds.any():
            continue
        # get bboxes and scores of this class
        if multi_bboxes.shape[1] == 2:
            _bboxes = multi_bboxes[cls_inds, :]
        else:
            _bboxes = multi_bboxes[cls_inds, i * 2:(i + 1) * 2]
        _scores = multi_scores[cls_inds, i]
        #乘上centerness
        if score_factors is not None:
            _scores *= score_factors[cls_inds]
        cls_dets = torch.cat([_bboxes, _scores[:, None]], dim=1) #(n, 3)

        cls_dets = hard_nms(  # (y, 3) y是每一类经过nms后剩下的box数
                cls_dets, nms_thr, max_num, pre_nms)
        cls_labels = multi_bboxes.new_full((cls_dets.shape[0], ),
                                           i,
                                           dtype=torch.long)
        
        bboxes.append(cls_dets)
        labels.append(cls_labels)
    if bboxes:
        bboxes = torch.cat(bboxes)
        labels = torch.cat(labels)
        if bboxes.shape[0] > max_num:
            _, inds = bboxes[:, -1].sort(descending=True)
            inds = inds[:max_num]
            bboxes = bboxes[inds]
            labels = labels[inds]
    else:
        bboxes = multi_bboxes.new_zeros((0, 5))
        labels = multi_bboxes.new_zeros((0, ), dtype=torch.long)
        
    return bboxes, labels

def get_bboxes_single(config,
                      criterion,
                      cls_scores,
                      bbox_preds,
                      centernesses,
                      mlvl_points,
                      video_shape,
                      scale_factor):
        assert len(cls_scores) == len(bbox_preds) == len(mlvl_points)
        mlvl_bboxes = []
        mlvl_scores = []
        mlvl_centerness = []
        
        for cls_score, bbox_pred, centerness, points in zip(
                cls_scores, bbox_preds, centernesses, mlvl_points):
            assert cls_score.size()[-1] == bbox_pred.size()[-1]
            print(cls_score.size)
            print(cls_score.shape)

            scores = cls_score.permute(1, 0).reshape(
                -1, criterion.cls_out_channels).sigmoid()
            centerness = centerness.permute(1, 0).reshape(-1).sigmoid()

            bbox_pred = bbox_pred.permute(1, 0).reshape(-1, 2)
            nms_pre = config.nms_pre

            if nms_pre > 0 and scores.shape[0] > nms_pre:
                max_scores, _ = (scores * centerness[:, None]).max(dim=1)
                _, topk_inds = max_scores.topk(nms_pre)
                points = points[topk_inds, :]
                bbox_pred = bbox_pred[topk_inds, :]
                scores = scores[topk_inds, :]

                centerness = centerness[topk_inds]
            bboxes = distance2bbox(points, bbox_pred, max_shape=video_shape)
            mlvl_bboxes.append(bboxes)
            mlvl_scores.append(scores)
            mlvl_centerness.append(centerness)
        mlvl_bboxes = torch.cat(mlvl_bboxes)
        mlvl_scores = torch.cat(mlvl_scores)
        padding = mlvl_scores.new_zeros(mlvl_scores.shape[0], 1)
        mlvl_scores = torch.cat([padding, mlvl_scores], dim=1)
        mlvl_centerness = torch.cat(mlvl_centerness)

        det_bboxes, det_labels = multiclass_nms(
            mlvl_bboxes,
            mlvl_scores,
            config.score_thresold,
            config.nms_thresold,
            config.max_per_video,
            score_factors=mlvl_centerness,
            pre_nms=config.nms_pre)

        return det_bboxes, det_labels

def get_bboxes(config,
               criterion,
               cls_scores,
               bbox_preds,
               centernesses,
               video_num,
               video_start_id=0,
               ):#rescale:是否再次缩放
        assert len(cls_scores) == len(bbox_preds)
        num_levels = len(cls_scores)
        #featmap_sizes = [featmap.shape()[-1] for featmap in cls_scores]
        featmap_sizes = [1 for featmap in cls_scores]
        mlvl_points = criterion.get_points(featmap_sizes, bbox_preds[0].dtype,
                                      bbox_preds[0].device) 
        result_list = []
        for video_id in range(video_num):
            cls_score_list = [
                cls_scores[i][video_id].detach() for i in range(num_levels)
            ]
            bbox_pred_list = [
                bbox_preds[i][video_id].detach() for i in range(num_levels)
            ]
            centerness_pred_list = [
                centernesses[i][video_id].detach() for i in range(num_levels)
            ]
            video_shape = config.window_size
            scale_factor = 1.0

            det_bboxes, det_labels = get_bboxes_single(config,
                                                       criterion,
                                                       cls_score_list, 
                                                       bbox_pred_list,
                                                       centerness_pred_list,
                                                       mlvl_points, video_shape,
                                                       scale_factor)
            id = torch.ones(det_labels.size(0), 1, dtype=torch.float32) * (video_id + video_start_id)
            result_list.append(
                torch.cat(
                    [
                        det_labels.reshape(-1, 1).type(torch.float32), #label
                        id.cuda(),
                        det_bboxes[:,-1].reshape(-1, 1), #prob
                        det_bboxes[:,:-1].reshape(-1, 2), #box
                    ],
                    dim=1
                )
            )
        video_start_id = video_start_id + video_num
        return result_list, video_start_id




def cal_iou(config, corrected_min, corrected_max, a_scores, gt_box):
    rank = 0
    iou = []
    for idx in range(len(corrected_min)):
        if rank > 10:
            break
        gt_intersection = -max(corrected_min[idx], gt_box[0]) + min(corrected_max[idx], gt_box[1]) 
        gt_union = -min(corrected_min[idx], gt_box[0]) + max(corrected_max[idx], gt_box[1])
        tmp_ioa = max(float(float(gt_intersection) / float(gt_union)), 0.)
        rank += 1
        iou.append(tmp_ioa)
    #     print("[%.3f, %.3f]\tpred_iou: %.4f\tgt_iou: %.4f\n" % (float(corrected_min[idx]), float(corrected_max[idx]), a_scores[idx], tmp_ioa))
    # print("rank 1: max_iou = %.4f\trank 5: max_iou = %.4f\trank 10: max_iou = %.4f\t" % (iou[0], max(iou[:5]), max(iou[:10])))
    return iou
