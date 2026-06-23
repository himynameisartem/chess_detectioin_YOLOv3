# src/train.py

import os
import torch
from torch.utils.data import DataLoader

from .config import (
    TRAIN_IMAGES,
    TRAIN_LABELS,
    VAL_IMAGES,
    VAL_LABELS,
    BATCH_SIZE,
    EPOCHS,
    ANCHORS,
)
from .dataset import ChessDataset
from .loss import YoloLoss
from .model import YOLOv3


DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
CHECKPOINT_FILE = "notebooks/yolov3_checkpoint.pth"
BEST_CHECKPOINT_FILE = "notebooks/yolov3_best.pth"


def train():
    train_dataset = ChessDataset(images_path=TRAIN_IMAGES, labels_path=TRAIN_LABELS)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

    valid_dataset = ChessDataset(images_path=VAL_IMAGES, labels_path=VAL_LABELS)
    valid_loader = DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=False)

    model = YOLOv3().to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    criterion = YoloLoss()

    start_epoch = 0

    if os.path.exists(CHECKPOINT_FILE):
        checkpoint = torch.load(CHECKPOINT_FILE, map_location=DEVICE)
        model.load_state_dict(checkpoint["state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        start_epoch = checkpoint["epoch"] + 1
        print(f"Let's continue with the epoch {start_epoch}")

    print("Start training the model...")

    best_val_loss = float("inf")

    for epoch in range(start_epoch, EPOCHS):
        model.train()
        train_loss = 0.0

        for imgs, y_true1, y_true2, y_true3, _ in train_loader:
            imgs = imgs.to(DEVICE)
            y_true1 = y_true1.to(DEVICE)
            y_true2 = y_true2.to(DEVICE)
            y_true3 = y_true3.to(DEVICE)

            optimizer.zero_grad()

            out1, out2, out3 = model(imgs)

            loss1 = criterion(out1, y_true1, torch.tensor(ANCHORS[0], dtype=torch.float32).to(DEVICE))
            loss2 = criterion(out2, y_true2, torch.tensor(ANCHORS[1], dtype=torch.float32).to(DEVICE))
            loss3 = criterion(out3, y_true3, torch.tensor(ANCHORS[2], dtype=torch.float32).to(DEVICE))

            total_loss = loss1 + loss2 + loss3
            total_loss.backward()
            optimizer.step()

            train_loss += total_loss.item()

        model.eval()
        val_loss = 0.0

        with torch.no_grad():
            for imgs, y_true1, y_true2, y_true3, _ in valid_loader:
                imgs = imgs.to(DEVICE)
                y_true1 = y_true1.to(DEVICE)
                y_true2 = y_true2.to(DEVICE)
                y_true3 = y_true3.to(DEVICE)

                out1, out2, out3 = model(imgs)

                loss1 = criterion(out1, y_true1, torch.tensor(ANCHORS[0], dtype=torch.float32).to(DEVICE))
                loss2 = criterion(out2, y_true2, torch.tensor(ANCHORS[1], dtype=torch.float32).to(DEVICE))
                loss3 = criterion(out3, y_true3, torch.tensor(ANCHORS[2], dtype=torch.float32).to(DEVICE))

                total_loss = loss1 + loss2 + loss3
                val_loss += total_loss.item()

        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(valid_loader)

        print(
            f"Epoch [{epoch + 1}/{EPOCHS}] "
            f"Train Loss: {avg_train_loss:.4f} "
            f"Val Loss: {avg_val_loss:.4f}"
        )

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_checkpoint = {
                "state_dict": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "epoch": epoch,
                "val_loss": best_val_loss,
            }
            torch.save(best_checkpoint, BEST_CHECKPOINT_FILE)
            print("Best checkpoint saved.")

        checkpoint = {
            "state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
            "val_loss": avg_val_loss,
        }
        torch.save(checkpoint, CHECKPOINT_FILE)
        print("Progress saved.")


if __name__ == "__main__":
    train()