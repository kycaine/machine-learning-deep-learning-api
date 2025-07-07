import os

REQUIRED_FOLDERS = [
    "outputs/cleaned",
    "outputs/eda",
    "outputs/feature_engineering",
    "outputs/models",
    "outputs/zips",
    "data/raw"
]

def create_required_folders():
    for folder in REQUIRED_FOLDERS:
        os.makedirs(folder, exist_ok=True)
