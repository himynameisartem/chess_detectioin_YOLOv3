SIZE = 320
BATCH_SIZE = 4
EPOCHS = 25
NUM_CLASSES = 12

TRAIN_IMAGES = "notebooks/chess_yolo/chess_yolo/train/images"
TRAIN_LABELS = "notebooks/chess_yolo/chess_yolo/train/labels"
VAL_IMAGES = "notebooks/chess_yolo/chess_yolo/valid/images"
VAL_LABELS = "notebooks/chess_yolo/chess_yolo/valid/labels"

CLASS_NAMES = [
    "black-bishop",
    "black-king",
    "black-knight",
    "black-pawn",
    "black-queen",
    "black-rook",
    "white-bishop",
    "white-king",
    "white-knight",
    "white-pawn",
    "white-queen",
    "white-rook",
]

ANCHORS = [
    [[116, 90], [156, 198], [373, 326]],
    [[30, 61], [62, 45], [59, 119]],
    [[10, 13], [16, 30], [33, 23]],
]

GRID_SIZES = [10, 20, 40]
IGNORE_IOU_THRESHOLD = 0.5
