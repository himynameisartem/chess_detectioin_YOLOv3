import zipfile
import os
import urllib.request

ZIP_URL = "https://storage.yandexcloud.net/academy.ai/CV/chess_yolo.zip"
ZIP_NAME = "chess_yolo.zip"
EXTRACT_DIR = "chess_yolo/chess_yolo"

def download_dataset():
    if not os.path.exists(EXTRACT_DIR):
        print("Loading dataset...")
        urllib.request.urlretrieve(ZIP_URL, ZIP_NAME)

        print("Unzip...")
        with zipfile.ZipFile(ZIP_NAME, 'r') as zip_ref:
            zip_ref.extractall(EXTRACT_DIR)

        os.remove(ZIP_NAME)
        print("Load comlete")
    else:
        print("Dataset is exist.")