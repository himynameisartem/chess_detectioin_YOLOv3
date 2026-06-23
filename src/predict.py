# src/predict.py

import cv2
import torch
import matplotlib.pyplot as plt

from .config import SIZE, ANCHORS, NUM_CLASSES, CLASS_NAMES
from .loss import yolo_boxes, non_max_suppression
from .model import YOLOv3


DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
CHECKPOINT_FILE = "notebooks/yolov3_best.pth"


def load_model(checkpoint_file=CHECKPOINT_FILE, num_classes=NUM_CLASSES):
    model = YOLOv3(num_classes=num_classes).to(DEVICE)
    checkpoint = torch.load(checkpoint_file, map_location=DEVICE)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model


def detect_and_scale_boxes(img_path, model, conf_threshold=0.2, iou_threshold=0.45, class_names=CLASS_NAMES):
    orig_img = cv2.imread(img_path)
    if orig_img is None:
        raise FileNotFoundError(f"Не удалось прочитать изображение: {img_path}")

    orig_img = cv2.cvtColor(orig_img, cv2.COLOR_BGR2RGB)
    orig_h, orig_w = orig_img.shape[:2]

    input_img = cv2.resize(orig_img, (SIZE, SIZE), interpolation=cv2.INTER_LINEAR)
    input_tensor = torch.tensor(input_img).permute(2, 0, 1).float().unsqueeze(0).to(DEVICE) / 255.0

    with torch.no_grad():
        outputs = model(input_tensor)

    final_boxes = []
    final_scores = []
    final_classes = []

    for i, pred in enumerate(outputs):
        bbox, obj_score, class_probs = yolo_boxes(pred, ANCHORS[i])

        class_score, class_idx = class_probs.max(dim=-1)
        scores = (obj_score.squeeze(-1) * class_score).reshape(-1)
        classes = class_idx.reshape(-1)
        boxes = bbox.reshape(-1, 4)

        mask = scores > conf_threshold
        if mask.any():
            boxes = boxes[mask]
            scores = scores[mask]
            classes = classes[mask]

            if scores.numel() > 200:
                topk = torch.topk(scores, k=200).indices
                boxes = boxes[topk]
                scores = scores[topk]
                classes = classes[topk]

            final_boxes.append(boxes)
            final_scores.append(scores)
            final_classes.append(classes)

    if not final_boxes:
        print(f"Объекты на картинке {orig_w}x{orig_h} не обнаружены.")
        return

    boxes_tensor = torch.cat(final_boxes, dim=0)
    scores_tensor = torch.cat(final_scores, dim=0)
    classes_tensor = torch.cat(final_classes, dim=0)

    keep = non_max_suppression(boxes_tensor, scores_tensor, iou_threshold=iou_threshold)
    boxes_tensor = boxes_tensor[keep]
    scores_tensor = scores_tensor[keep]
    classes_tensor = classes_tensor[keep]

    boxes_tensor[:, [0, 2]] = boxes_tensor[:, [0, 2]] * orig_w
    boxes_tensor[:, [1, 3]] = boxes_tensor[:, [1, 3]] * orig_h
    boxes_tensor[:, [0, 2]] = boxes_tensor[:, [0, 2]].clamp(0, orig_w)
    boxes_tensor[:, [1, 3]] = boxes_tensor[:, [1, 3]].clamp(0, orig_h)

    fig, ax = plt.subplots(1, figsize=(12, 8))
    ax.imshow(orig_img)

    for box, score, class_idx in zip(
        boxes_tensor.cpu().numpy(),
        scores_tensor.cpu().numpy(),
        classes_tensor.cpu().numpy(),
    ):
        x1, y1, x2, y2 = box
        class_idx = int(class_idx)
        class_name = class_names[class_idx] if 0 <= class_idx < len(class_names) else f"class_{class_idx}"

        rect = plt.Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, color="cyan", linewidth=3)
        ax.add_patch(rect)

        ax.text(
            x1,
            max(0, y1 - 6),
            f"{class_name}: {score:.2f}",
            color="white",
            fontsize=10,
            bbox=dict(facecolor="cyan", alpha=0.75, edgecolor="none", pad=2),
        )

    orientation = "Альбомная" if orig_w > orig_h else "Портретная"
    plt.title(f"Разрешение оригинала: {orig_w}x{orig_h} ({orientation})")
    plt.axis("off")
    plt.show()


if __name__ == "__main__":
    model = load_model()
    test_img_path = "chess_yolo/chess_yolo/test/fdcd6ada676799da8a870f58fdf548db_jpg.rf.b0ea8552b6106bb4ab62ca8957fca40d.jpg"
    detect_and_scale_boxes(test_img_path, model)
