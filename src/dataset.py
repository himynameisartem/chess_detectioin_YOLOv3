import os
import glob
import cv2
import torch
from torch.utils.data import Dataset

from .config import SIZE, GRID_SIZES, ANCHORS, IGNORE_IOU_THRESHOLD


def iou_width_height(wh1, wh2):
    inter = torch.minimum(wh1[0], wh2[:, 0]) * torch.minimum(wh1[1], wh2[:, 1])
    union = wh1[0] * wh1[1] + wh2[:, 0] * wh2[:, 1] - inter
    return inter / union


class ChessDataset(Dataset):
    def __init__(self, images_path, labels_path):
        self.img_paths = sorted(glob.glob(os.path.join(images_path, "*.jpg")))
        self.labels_path = labels_path

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path = str(self.img_paths[idx])
        img_filename = os.path.basename(img_path)
        label_filename = img_filename.replace(".jpg", ".txt")
        label_path = os.path.join(self.labels_path, label_filename)

        img = cv2.imread(img_path)
        if img is None:
            raise FileNotFoundError(f"Unable to read image: {img_path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        orig_h, orig_w = img.shape[:2]
        img_resized = cv2.resize(img, (SIZE, SIZE))
        img_tensor = torch.tensor(img_resized).permute(2, 0, 1).float() / 255.0

        y_trues = [torch.zeros((S, S, 3, 6)) for S in GRID_SIZES]

        if os.path.exists(label_path):
            with open(label_path, "r") as f:
                for line in f.readlines():
                    parts = line.strip().split()

                    if len(parts) == 5:
                        class_idx, x, y, w, h = map(float, parts)
                        class_idx = int(class_idx)
                    elif len(parts) == 9:
                        class_idx = int(float(parts[0]))
                        coords = list(map(float, parts[1:]))
                        xs = coords[0::2]
                        ys = coords[1::2]
                        x_min, x_max = min(xs), max(xs)
                        y_min, y_max = min(ys), max(ys)
                        x = (x_min + x_max) / 2.0
                        y = (y_min + y_max) / 2.0
                        w = x_max - x_min
                        h = y_max - y_min
                    else:
                        continue

                    if class_idx == 0:
                        continue

                    class_idx -= 1

                    flat_anchors = torch.tensor([a for scale in ANCHORS for a in scale])
                    box_wh = torch.tensor([w * SIZE, h * SIZE])

                    ious = iou_width_height(box_wh, flat_anchors)
                    best_anchor_idx = torch.argmax(ious).item()

                    scale_idx = int(best_anchor_idx // 3)
                    anchor_on_scale = int(best_anchor_idx % 3)
                    grid_size = GRID_SIZES[scale_idx]

                    grid_x = int(x * grid_size)
                    grid_y = int(y * grid_size)

                    grid_x = min(grid_x, grid_size - 1)
                    grid_y = min(grid_y, grid_size - 1)

                    for anchor_idx, anchor_iou in enumerate(ious):
                        if anchor_idx == best_anchor_idx or anchor_iou <= IGNORE_IOU_THRESHOLD:
                            continue

                        ignore_scale_idx = int(anchor_idx // 3)
                        ignore_anchor_on_scale = int(anchor_idx % 3)
                        ignore_grid_size = GRID_SIZES[ignore_scale_idx]
                        ignore_grid_x = min(int(x * ignore_grid_size), ignore_grid_size - 1)
                        ignore_grid_y = min(int(y * ignore_grid_size), ignore_grid_size - 1)

                        if y_trues[ignore_scale_idx][ignore_grid_y, ignore_grid_x, ignore_anchor_on_scale, 4] == 0:
                            y_trues[ignore_scale_idx][ignore_grid_y, ignore_grid_x, ignore_anchor_on_scale, 4] = -1.0

                    if y_trues[scale_idx][grid_y, grid_x, anchor_on_scale, 4] == 0:
                        y_trues[scale_idx][grid_y, grid_x, anchor_on_scale, 0:4] = torch.tensor([x, y, w, h])
                        y_trues[scale_idx][grid_y, grid_x, anchor_on_scale, 4] = 1.0
                        y_trues[scale_idx][grid_y, grid_x, anchor_on_scale, 5] = class_idx

        return img_tensor, y_trues[0], y_trues[1], y_trues[2], torch.tensor([orig_w, orig_h])
