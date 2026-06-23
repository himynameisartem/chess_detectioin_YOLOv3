import argparse

from src.download_dataset import download_dataset
from src.train import train
from src.predict import load_model, detect_and_scale_boxes


def main():
    parser = argparse.ArgumentParser(description="Chess piece detection with YOLOv3")
    parser.add_argument(
        "mode",
        choices=["download", "train", "predict"],
        help="What to run: download dataset, train model, or run prediction",
    )
    parser.add_argument(
        "--image",
        type=str,
        default=None,
        help="Path to image for prediction",
    )

    args = parser.parse_args()

    if args.mode == "download":
        download_dataset()

    elif args.mode == "train":
        train()

    elif args.mode == "predict":
        if args.image is None:
            raise ValueError("For predict mode, provide --image path")
        model = load_model()
        detect_and_scale_boxes(args.image, model)


if __name__ == "__main__":
    main()
