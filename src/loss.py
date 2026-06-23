# src/loss.py

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import NUM_CLASSES, SIZE


def yolo_boxes(pred, anchors):
    device = pred.device
    grid_size = pred.shape[1]
    anchors_tensor = torch.tensor(anchors, dtype=pred.dtype, device=device).view(1, 1, 1, -1, 2)

    box_xy = torch.sigmoid(pred[..., 0:2])
    box_wh = pred[..., 2:4]
    score = torch.sigmoid(pred[..., 4:5])
    class_probs = torch.sigmoid(pred[..., 5:])

    grid_y, grid_x = torch.meshgrid(
        torch.arange(grid_size, device=device),
        torch.arange(grid_size, device=device),
        indexing="ij",
    )
    grid = torch.stack((grid_x, grid_y), dim=-1).float().unsqueeze(2)

    b_xy = (box_xy + grid) / float(grid_size)
    b_wh = (torch.exp(box_wh) * anchors_tensor) / float(SIZE)

    box_x1y1 = b_xy - b_wh / 2
    box_x2y2 = b_xy + b_wh / 2
    bbox = torch.cat([box_x1y1, box_x2y2], dim=-1)

    return bbox, score, class_probs


def box_iou(box1, box2):
    box1 = box1.unsqueeze(1)
    box2 = box2.unsqueeze(0)

    inter_x1 = torch.maximum(box1[..., 0], box2[..., 0])
    inter_y1 = torch.maximum(box1[..., 1], box2[..., 1])
    inter_x2 = torch.minimum(box1[..., 2], box2[..., 2])
    inter_y2 = torch.minimum(box1[..., 3], box2[..., 3])

    inter_w = torch.clamp(inter_x2 - inter_x1, min=0)
    inter_h = torch.clamp(inter_y2 - inter_y1, min=0)
    inter_area = inter_w * inter_h

    area1 = torch.clamp(box1[..., 2] - box1[..., 0], min=0) * torch.clamp(box1[..., 3] - box1[..., 1], min=0)
    area2 = torch.clamp(box2[..., 2] - box2[..., 0], min=0) * torch.clamp(box2[..., 3] - box2[..., 1], min=0)
    union = area1 + area2 - inter_area + 1e-7
    return inter_area / union


def non_max_suppression(boxes, scores, iou_threshold=0.45):
    if boxes.numel() == 0:
        return torch.empty(0, dtype=torch.long, device=boxes.device)

    keep = []
    order = scores.argsort(descending=True)

    while order.numel() > 0:
        current = order[0]
        keep.append(current)

        if order.numel() == 1:
            break

        ious = box_iou(boxes[current].unsqueeze(0), boxes[order[1:]]).squeeze(0)
        order = order[1:][ious <= iou_threshold]

    return torch.stack(keep)


class YoloLoss(nn.Module):
    def __init__(self, classes=NUM_CLASSES):
        super().__init__()
        self.classes = classes
        self.bce = nn.BCELoss(reduction="none")
        self.mse = nn.MSELoss(reduction="none")
        self.noobj_scale = 0.05

    def forward(self, y_pred, y_true, scale_anchors):
        device = y_pred.device
        scale_anchors = scale_anchors.to(device)

        grid_size = y_pred.shape[1]

        pred_xy = torch.sigmoid(y_pred[..., 0:2])
        pred_wh = y_pred[..., 2:4]
        pred_obj = torch.sigmoid(y_pred[..., 4:5])
        pred_class = torch.sigmoid(y_pred[..., 5:])

        true_box = y_true[..., 0:4]
        true_obj = y_true[..., 4:5]
        true_class_idx = y_true[..., 5].long()

        grid_y, grid_x = torch.meshgrid(
            torch.arange(grid_size, device=device),
            torch.arange(grid_size, device=device),
            indexing="ij",
        )
        grid = torch.stack((grid_x, grid_y), dim=-1).float().unsqueeze(2)

        true_xy = true_box[..., 0:2] * grid_size - grid
        epsilon = 1e-7
        true_wh = torch.log((true_box[..., 2:4] * float(SIZE)) / scale_anchors + epsilon)
        true_wh = torch.where(torch.isinf(true_wh), torch.zeros_like(true_wh), true_wh)

        box_loss_scale = 2.0 - true_box[..., 2:3] * true_box[..., 3:4]
        obj_mask = (true_obj.squeeze(-1) == 1).float()
        noobj_mask = (true_obj.squeeze(-1) == 0).float()

        xy_loss = obj_mask.unsqueeze(-1) * box_loss_scale * self.mse(pred_xy, true_xy)
        wh_loss = obj_mask.unsqueeze(-1) * box_loss_scale * self.mse(pred_wh, true_wh)

        obj_loss = (
            obj_mask.unsqueeze(-1) * self.bce(pred_obj, torch.ones_like(pred_obj))
            + self.noobj_scale * noobj_mask.unsqueeze(-1) * self.bce(pred_obj, torch.zeros_like(pred_obj))
        )

        true_class_one_hot = F.one_hot(true_class_idx, num_classes=self.classes).float()
        class_loss = obj_mask.unsqueeze(-1) * self.bce(pred_class, true_class_one_hot)

        total_loss = xy_loss.sum() + wh_loss.sum() + obj_loss.sum() + class_loss.sum()
        normalizer = obj_mask.sum().clamp(min=1.0)

        return total_loss / normalizer
