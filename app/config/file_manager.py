import os

from fastapi import UploadFile

from app.config.constants import *


def save_raw_data(upload_file: UploadFile):
    filename = upload_file.filename
    save_path = os.path.join(DATA_RAW, filename)

    if os.path.exists(save_path):
        os.remove(save_path)

    with open(save_path, "wb") as f:
        f.write(upload_file.file.read())

    return save_path
