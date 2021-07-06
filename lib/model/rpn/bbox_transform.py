# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------
# --------------------------------------------------------
# Reorganized and modified by Jianwei Yang and Jiasen Lu
# --------------------------------------------------------

import torch
import numpy as np
import pdb

def bbox_transform(ex_rois, gt_rois):
    ex_widths = ex_rois[:, 2] - ex_rois[:, 0] + 1.0
    ex_heights = ex_rois[:, 3] - ex_rois[:, 1] + 1.0
    ex_ctr_x = ex_rois[:, 0] + 0.5 * ex_widths
    ex_ctr_y = ex_rois[:, 1] + 0.5 * ex_heights

    gt_widths = gt_rois[:, 2] - gt_rois[:, 0] + 1.0
    gt_heights = gt_rois[:, 3] - gt_rois[:, 1] + 1.0
    gt_ctr_x = gt_rois[:, 0] + 0.5 * gt_widths
    gt_ctr_y = gt_rois[:, 1] + 0.5 * gt_heights

    targets_dx = (gt_ctr_x - ex_ctr_x) / ex_widths
    targets_dy = (gt_ctr_y - ex_ctr_y) / ex_heights
    targets_dw = torch.log(gt_widths / ex_widths)
    targets_dh = torch.log(gt_heights / ex_heights)

    targets = torch.stack(
        (targets_dx, targets_dy, targets_dw, targets_dh),1)

    return targets

def bbox_transform_batch(ex_rois, gt_rois):

    if ex_rois.dim() == 2:
        ex_widths = ex_rois[:, 2] - ex_rois[:, 0] + 1.0
        ex_heights = ex_rois[:, 3] - ex_rois[:, 1] + 1.0
        ex_ctr_x = ex_rois[:, 0] + 0.5 * ex_widths
        ex_ctr_y = ex_rois[:, 1] + 0.5 * ex_heights

        gt_widths = gt_rois[:, :, 2] - gt_rois[:, :, 0] + 1.0
        gt_heights = gt_rois[:, :, 3] - gt_rois[:, :, 1] + 1.0
        gt_ctr_x = gt_rois[:, :, 0] + 0.5 * gt_widths
        gt_ctr_y = gt_rois[:, :, 1] + 0.5 * gt_heights

        ex_x1 = ex_rois[:, 0]
        ex_y1 = ex_rois[:, 1]
        ex_x2 = ex_rois[:, 2]
        ex_y2 = ex_rois[:, 1]

        gt_x1 = gt_rois[:, :, 4]
        gt_y1 = gt_rois[:, :, 5]
        gt_x2 = gt_rois[:, :, 6]
        gt_y2 = gt_rois[:, :, 7]
        gt_h = gt_rois[:, :, 8]
      #  print '~~~~~~~~~~~~~size of ex_x1 is '+str(ex_x1.size())
      #  print '~~~~~~~~~~~~~~~size of gt_ctr_x is '+str(gt_ctr_x.size())
        tmp = ex_x1.contiguous().view(1,-1)
       # print '~~~~~~~change tmp size is '+str(tmp.size())
        dx1 = (gt_x1 - ex_x1.contiguous().view(1,-1).expand_as(gt_ctr_x)) / ex_widths
        dy1 = (gt_y1 - ex_y1.contiguous().view(1,-1).expand_as(gt_ctr_y)) / ex_heights
        dx2 = (gt_x2 - ex_x2.contiguous().view(1, -1).expand_as(gt_ctr_x)) / ex_widths
        dy2 = (gt_y2 - ex_y2.contiguous().view(1, -1).expand_as(gt_ctr_y)) / ex_heights
        dh = torch.log(gt_h / ex_heights.contiguous().view(1,-1).expand_as(gt_widths))


      #  targets_dx = (gt_ctr_x - ex_ctr_x.view(1,-1).expand_as(gt_ctr_x)) / ex_widths
      #  targets_dy = (gt_ctr_y - ex_ctr_y.view(1,-1).expand_as(gt_ctr_y)) / ex_heights
      #  targets_dw = torch.log(gt_widths / ex_widths.view(1,-1).expand_as(gt_widths))
      #  targets_dh = torch.log(gt_heights / ex_heights.view(1,-1).expand_as(gt_heights))

    elif ex_rois.dim() == 3:
        ex_widths = ex_rois[:, :, 2] - ex_rois[:, :, 0] + 1.0
        ex_heights = ex_rois[:,:, 3] - ex_rois[:,:, 1] + 1.0
        ex_ctr_x = ex_rois[:, :, 0] + 0.5 * ex_widths
        ex_ctr_y = ex_rois[:, :, 1] + 0.5 * ex_heights

        gt_widths = gt_rois[:, :, 2] - gt_rois[:, :, 0] + 1.0
        gt_heights = gt_rois[:, :, 3] - gt_rois[:, :, 1] + 1.0
        gt_ctr_x = gt_rois[:, :, 0] + 0.5 * gt_widths
        gt_ctr_y = gt_rois[:, :, 1] + 0.5 * gt_heights

        ex_x1 = ex_rois[:, 0]
        ex_y1 = ex_rois[:, 1]
        ex_x2 = ex_rois[:, 2]
        ex_y2 = ex_rois[:, 1]

        gt_x1 = gt_rois[:, :, 4]
        gt_y1 = gt_rois[:, :, 5]
        gt_x2 = gt_rois[:, :, 6]
        gt_y2 = gt_rois[:, :, 7]
        gt_h = gt_rois[:, :, 8]

        dx1 = (gt_x1 - ex_x1) / ex_widths
        dy1 = (gt_y1 - ex_y1) / ex_heights
        dx2 = (gt_x2 - ex_x2) / ex_widths
        dy2 = (gt_y2 - ex_y2) / ex_heights
        dh = torch.log(gt_h / ex_heights)

       # targets_dx = (gt_ctr_x - ex_ctr_x) / ex_widths
       # targets_dy = (gt_ctr_y - ex_ctr_y) / ex_heights
       # targets_dw = torch.log(gt_widths / ex_widths)
       # targets_dh = torch.log(gt_heights / ex_heights)
    else:
        raise ValueError('ex_roi input dimension is not correct.')

    targets = torch.stack(
        (dx1, dy1, dx2, dy2,dh),2)

    return targets

def bbox_transform_batch_reg(ex_rois, gt_rois):

    if ex_rois.dim() == 2:
        ex_widths = ex_rois[:, 2] - ex_rois[:, 0] + 1.0
        ex_heights = ex_rois[:, 3] - ex_rois[:, 1] + 1.0
        ex_ctr_x = ex_rois[:, 0] + 0.5 * ex_widths
        ex_ctr_y = ex_rois[:, 1] + 0.5 * ex_heights

        ex_pts = [ex_rois[:, 0], ex_rois[:, 1],
                  ex_rois[:, 2], ex_rois[:, 1],
                  ex_rois[:, 2], ex_rois[:, 3],
                  ex_rois[:, 0], ex_rois[:, 3]]

        gt_widths = gt_rois[:, :, 2] - gt_rois[:, :, 0] + 1.0
        gt_heights = gt_rois[:, :, 3] - gt_rois[:, :, 1] + 1.0
        gt_ctr_x = gt_rois[:, :, 0] + 0.5 * gt_widths
        gt_ctr_y = gt_rois[:, :, 1] + 0.5 * gt_heights

        gt_pts = []
        targets_pts = []

        for i in range(4, 9):
            gt_pts.append(gt_rois[:, :, i])
        for i in range(0, 4, 2):
            target_x = (gt_pts[i] - ex_pts[i]) / ex_widths
            target_y = (gt_pts[i + 1] - ex_pts[i + 1]) / ex_heights
            targets_pts.append(target_x)
            targets_pts.append(target_y)
        targets_pts.append(torch.log(gt_pts[4] / ex_heights.view(1,-1).expand_as(gt_pts[4])))
        targets_dx = (gt_ctr_x - ex_ctr_x.view(1,-1).expand_as(gt_ctr_x)) / ex_widths
        targets_dy = (gt_ctr_y - ex_ctr_y.view(1,-1).expand_as(gt_ctr_y)) / ex_heights
        targets_dw = torch.log(gt_widths / ex_widths.view(1,-1).expand_as(gt_widths))
        targets_dh = torch.log(gt_heights / ex_heights.view(1,-1).expand_as(gt_heights))

    elif ex_rois.dim() == 3:
        ex_widths = ex_rois[:, :, 2] - ex_rois[:, :, 0] + 1.0
        ex_heights = ex_rois[:,:, 3] - ex_rois[:,:, 1] + 1.0
        ex_ctr_x = ex_rois[:, :, 0] + 0.5 * ex_widths
        ex_ctr_y = ex_rois[:, :, 1] + 0.5 * ex_heights

        ex_pts = [ex_rois[:, :, 0], ex_rois[:, :, 1],
                  ex_rois[:, :, 2], ex_rois[:, :, 1],
                  ex_rois[:, :, 2], ex_rois[:, :, 3],
                  ex_rois[:, :, 0], ex_rois[:, :, 3]]

        gt_widths = gt_rois[:, :, 2] - gt_rois[:, :, 0] + 1.0
        gt_heights = gt_rois[:, :, 3] - gt_rois[:, :, 1] + 1.0
        gt_ctr_x = gt_rois[:, :, 0] + 0.5 * gt_widths
        gt_ctr_y = gt_rois[:, :, 1] + 0.5 * gt_heights
        gt_pts = []
        targets_pts = []

        for i in range(4, 9):
            gt_pts.append(gt_rois[:, :, i])
        for i in range(0, 4, 2):
            target_x = (gt_pts[i] - ex_pts[i]) / ex_widths
            target_y = (gt_pts[i + 1] - ex_pts[i + 1]) / ex_heights
            targets_pts.append(target_x)
            targets_pts.append(target_y)

        targets_pts.append(torch.log(gt_pts[4] / ex_heights))
        targets_dx = (gt_ctr_x - ex_ctr_x) / ex_widths
        targets_dy = (gt_ctr_y - ex_ctr_y) / ex_heights
        targets_dw = torch.log(gt_widths / ex_widths)
        targets_dh = torch.log(gt_heights / ex_heights)
    else:
        raise ValueError('ex_roi input dimension is not correct.')

    targets = torch.stack(
        (targets_dx, targets_dy, targets_dw, targets_dh),2)
    poly_targets = torch.stack(targets_pts, 2)
    return targets, poly_targets

def bbox_transform_inv(boxes, deltas, batch_size):
    widths = boxes[:, :, 2] - boxes[:, :, 0] + 1.0
    heights = boxes[:, :, 3] - boxes[:, :, 1] + 1.0
    ctr_x = boxes[:, :, 0] + 0.5 * widths
    ctr_y = boxes[:, :, 1] + 0.5 * heights

    dx = deltas[:, :, 0::4]
    dy = deltas[:, :, 1::4]
    dw = deltas[:, :, 2::4]
    dh = deltas[:, :, 3::4]

    pred_ctr_x = dx * widths.unsqueeze(2) + ctr_x.unsqueeze(2)
    pred_ctr_y = dy * heights.unsqueeze(2) + ctr_y.unsqueeze(2)
    pred_w = torch.exp(dw) * widths.unsqueeze(2)
    pred_h = torch.exp(dh) * heights.unsqueeze(2)

    pred_boxes = deltas.clone()
    # x1
    pred_boxes[:, :, 0::4] = pred_ctr_x - 0.5 * pred_w
    # y1
    pred_boxes[:, :, 1::4] = pred_ctr_y - 0.5 * pred_h
    # x2
    pred_boxes[:, :, 2::4] = pred_ctr_x + 0.5 * pred_w
    # y2
    pred_boxes[:, :, 3::4] = pred_ctr_y + 0.5 * pred_h

    return pred_boxes


def bbox_transform_inv_reg(boxes, deltas, batch_size):
    if boxes.dim() == 3:
        widths = boxes[:, :, 2] - boxes[:, :, 0] + 1.0
        heights = boxes[:, :, 3] - boxes[:, :, 1] + 1.0

        dx = deltas[:, :, 0::4]
        dy = deltas[:, :, 1::4]
        dw = deltas[:, :, 2::4]
        dh = deltas[:, :, 3::4]

        boxes = [boxes[:, :, 0], boxes[:, :, 1],
                 boxes[:, :, 2], boxes[:, :, 1],
                 boxes[:, :, 2], boxes[:, :, 3],
                 boxes[:, :, 0], boxes[:, :, 3]]

    pred_boxes = deltas.clone()
   # print ('deltas size is {}'.format(deltas.size()))
   # print ('heights size is {}'.format(heights.size()))
    for i in range(0, 4, 2):
        px = boxes[i]
        py = boxes[i + 1]
        pred_boxes[:, :, i::5] = deltas[:, :, i::5] * widths.unsqueeze(2) + px.unsqueeze(2)
        pred_boxes[:, :, i + 1::5] = deltas[:, :, i + 1::5] * heights.unsqueeze(2) + py.unsqueeze(2)
    pred_boxes[:, :, 4::5] = torch.exp(deltas[:, :, 4::5]) * heights.unsqueeze(2)
    return reverse_to_quad(pred_boxes)

def reverse_to_quad(rotated):
  x1 = rotated[:, :,  0::5]
  y1 = rotated[:, :,  1::5]
  x2 = rotated[:, :,  2::5]
  y2 = rotated[:, :,  3::5]
  h = rotated[:, :,  4::5]
  # print(rotated.shape[1], '#########')
  pred_boxes = torch.zeros((rotated.shape[0], rotated.shape[1], rotated.shape[2]//5*8)).type_as(rotated)
  eps = 1e-10
  theta = torch.atan((y2 - y1) / (x2 - x1 + eps))
  x4 = x1 - h * torch.sin(theta)
  y4 = y1 + h * torch.cos(theta)
  x3 = x4 + x2 - x1
  y3 = y4 + y2 - y1
  pred_boxes[:, :, 0::8] = x1
  pred_boxes[:, :, 1::8] = y1
  pred_boxes[:, :, 2::8] = x2
  pred_boxes[:, :, 3::8] = y2
  pred_boxes[:, :, 4::8] = x3
  pred_boxes[:, :, 5::8] = y3
  pred_boxes[:, :, 6::8] = x4
  pred_boxes[:, :, 7::8] = y4
  return pred_boxes

def clip_boxes_batch(boxes, im_shape, batch_size):
    """
    Clip boxes to image boundaries.
    """
    num_rois = boxes.size(1)

    boxes[boxes < 0] = 0
    # batch_x = (im_shape[:,0]-1).view(batch_size, 1).expand(batch_size, num_rois)
    # batch_y = (im_shape[:,1]-1).view(batch_size, 1).expand(batch_size, num_rois)

    batch_x = im_shape[:, 1] - 1
    batch_y = im_shape[:, 0] - 1

    boxes[:,:,0][boxes[:,:,0] > batch_x] = batch_x
    boxes[:,:,1][boxes[:,:,1] > batch_y] = batch_y
    boxes[:,:,2][boxes[:,:,2] > batch_x] = batch_x
    boxes[:,:,3][boxes[:,:,3] > batch_y] = batch_y

    return boxes

def clip_boxes(boxes, im_shape, batch_size):

    for i in range(batch_size):
        boxes[i,:,0::4].clamp_(0, im_shape[i, 1]-1)
        boxes[i,:,1::4].clamp_(0, im_shape[i, 0]-1)
        boxes[i,:,2::4].clamp_(0, im_shape[i, 1]-1)
        boxes[i,:,3::4].clamp_(0, im_shape[i, 0]-1)

    return boxes


def bbox_overlaps(anchors, gt_boxes):
    """
    anchors: (N, 4) ndarray of float
    gt_boxes: (K, 4) ndarray of float

    overlaps: (N, K) ndarray of overlap between boxes and query_boxes
    """
    N = anchors.size(0)
    K = gt_boxes.size(0)

    gt_boxes_area = ((gt_boxes[:,2] - gt_boxes[:,0] + 1) *
                (gt_boxes[:,3] - gt_boxes[:,1] + 1)).view(1, K)

    anchors_area = ((anchors[:,2] - anchors[:,0] + 1) *
                (anchors[:,3] - anchors[:,1] + 1)).view(N, 1)

    boxes = anchors.view(N, 1, 4).expand(N, K, 4)
    query_boxes = gt_boxes.view(1, K, 4).expand(N, K, 4)

    iw = (torch.min(boxes[:,:,2], query_boxes[:,:,2]) -
        torch.max(boxes[:,:,0], query_boxes[:,:,0]) + 1)
    iw[iw < 0] = 0

    ih = (torch.min(boxes[:,:,3], query_boxes[:,:,3]) -
        torch.max(boxes[:,:,1], query_boxes[:,:,1]) + 1)
    ih[ih < 0] = 0

    ua = anchors_area + gt_boxes_area - (iw * ih)
    overlaps = iw * ih / ua

    return overlaps

def bbox_overlaps_batch(anchors, gt_boxes):
    """
    anchors: (N, 4) ndarray of float
    gt_boxes: (b, K, 5) ndarray of float

    overlaps: (N, K) ndarray of overlap between boxes and query_boxes
    """
    batch_size = gt_boxes.size(0)


    if anchors.dim() == 2:

        N = anchors.size(0)
        K = gt_boxes.size(1)

        anchors = anchors.view(1, N, 4).expand(batch_size, N, 4).contiguous()
        gt_boxes = gt_boxes[:,:,:4].contiguous()


        gt_boxes_x = (gt_boxes[:,:,2] - gt_boxes[:,:,0] + 1)
        gt_boxes_y = (gt_boxes[:,:,3] - gt_boxes[:,:,1] + 1)
        gt_boxes_area = (gt_boxes_x * gt_boxes_y).view(batch_size, 1, K)

        anchors_boxes_x = (anchors[:,:,2] - anchors[:,:,0] + 1)
        anchors_boxes_y = (anchors[:,:,3] - anchors[:,:,1] + 1)
        anchors_area = (anchors_boxes_x * anchors_boxes_y).view(batch_size, N, 1)

        gt_area_zero = (gt_boxes_x == 1) & (gt_boxes_y == 1)
        anchors_area_zero = (anchors_boxes_x == 1) & (anchors_boxes_y == 1)

        boxes = anchors.view(batch_size, N, 1, 4).expand(batch_size, N, K, 4)
        query_boxes = gt_boxes.view(batch_size, 1, K, 4).expand(batch_size, N, K, 4)

        iw = (torch.min(boxes[:,:,:,2], query_boxes[:,:,:,2]) -
            torch.max(boxes[:,:,:,0], query_boxes[:,:,:,0]) + 1)
        iw[iw < 0] = 0

        ih = (torch.min(boxes[:,:,:,3], query_boxes[:,:,:,3]) -
            torch.max(boxes[:,:,:,1], query_boxes[:,:,:,1]) + 1)
        ih[ih < 0] = 0
        ua = anchors_area + gt_boxes_area - (iw * ih)
        overlaps = iw * ih / ua

        # mask the overlap here.
        overlaps.masked_fill_(gt_area_zero.view(batch_size, 1, K).expand(batch_size, N, K), 0)
        overlaps.masked_fill_(anchors_area_zero.view(batch_size, N, 1).expand(batch_size, N, K), -1)

    elif anchors.dim() == 3:
        N = anchors.size(1)
        K = gt_boxes.size(1)

        if anchors.size(2) == 4:
            anchors = anchors[:,:,:4].contiguous()
        else:
            anchors = anchors[:,:,1:5].contiguous()

        gt_boxes = gt_boxes[:,:,:4].contiguous()

        gt_boxes_x = (gt_boxes[:,:,2] - gt_boxes[:,:,0] + 1)
        gt_boxes_y = (gt_boxes[:,:,3] - gt_boxes[:,:,1] + 1)
        gt_boxes_area = (gt_boxes_x * gt_boxes_y).view(batch_size, 1, K)

        anchors_boxes_x = (anchors[:,:,2] - anchors[:,:,0] + 1)
        anchors_boxes_y = (anchors[:,:,3] - anchors[:,:,1] + 1)
        anchors_area = (anchors_boxes_x * anchors_boxes_y).view(batch_size, N, 1)

        gt_area_zero = (gt_boxes_x == 1) & (gt_boxes_y == 1)
        anchors_area_zero = (anchors_boxes_x == 1) & (anchors_boxes_y == 1)

        boxes = anchors.view(batch_size, N, 1, 4).expand(batch_size, N, K, 4)
        query_boxes = gt_boxes.view(batch_size, 1, K, 4).expand(batch_size, N, K, 4)

        iw = (torch.min(boxes[:,:,:,2], query_boxes[:,:,:,2]) -
            torch.max(boxes[:,:,:,0], query_boxes[:,:,:,0]) + 1)
        iw[iw < 0] = 0

        ih = (torch.min(boxes[:,:,:,3], query_boxes[:,:,:,3]) -
            torch.max(boxes[:,:,:,1], query_boxes[:,:,:,1]) + 1)
        ih[ih < 0] = 0
        ua = anchors_area + gt_boxes_area - (iw * ih)

        overlaps = iw * ih / ua

        # mask the overlap here.
        overlaps.masked_fill_(gt_area_zero.view(batch_size, 1, K).expand(batch_size, N, K), 0)
        overlaps.masked_fill_(anchors_area_zero.view(batch_size, N, 1).expand(batch_size, N, K), -1)
    else:
        raise ValueError('anchors input dimension is not correct.')

    return overlaps
