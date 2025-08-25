import datetime
import json
import os
import shutil
import tempfile
from pathlib import Path

import pandas as pd
from fastapi import APIRouter, File, Form, UploadFile
from fastapi.responses import JSONResponse

from app.api.schemas.request import GeneralRequest
from app.config.constants import *
from app.config.file_manager import *
from app.machine_learning.eda import generate_eda
from app.machine_learning.processing import *
from app.machine_learning.preprocessing import clean_data, feature_engineering

router = APIRouter()

@router.post("/clean", tags=["Machine Learning"])
async def clean_data_api(
    file: UploadFile = File(..., description=UPLOAD_DESCRIPTION),
    metadata: str = Form(..., description=REQUEST_DESCRIPTION)
):
    request_data = json.loads(metadata)
    request_obj = GeneralRequest(**request_data)

    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    cleaned_path, summary = clean_data(tmp_path, request_obj.columns, original_filename=file.filename)
    
    save_raw_data(file)

    return JSONResponse(content={
        "download_url": OUTPUTS_CLEANED + "/" + os.path.basename(cleaned_path),
        "summary": summary
    })


@router.post("/eda", tags=["Machine Learning"])
async def run_eda_api(file: UploadFile = File(...)):
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    original_name = file.filename.replace(".csv", "").replace(".xlsx", "")
    eda_output_prefix = f"eda_{original_name}_{timestamp}"

    tmp_path = Path(tempfile.gettempdir()) / f"{eda_output_prefix}.csv"
    tmp_path.write_bytes(await file.read())

    zip_file, summary = generate_eda(tmp_path, file.filename)
    download_url = f"{zip_file.replace(os.sep, '/')}"

    return JSONResponse(content={
        "download_url": download_url,
        "summary": summary
    })


@router.post("/feature-engineering", tags=["Machine Learning"])
async def run_feature_engineering_api(
    file: UploadFile,
    metadata: str = Form(...)
):
    try:
        params_dict = json.loads(metadata)
        feature_columns = params_dict.get("feature_columns")
        columns_schema = params_dict.get("columns")

        df = pd.read_csv(file.file)
        filename = file.filename

        processed_filename, processed_df = feature_engineering(df, filename, feature_columns, columns_schema)

        download_url = f"{OUTPUTS_FEATURE_ENGINNERING}/{processed_filename}"
        preview = processed_df.head().to_dict(orient="records")
        columns_after = list(processed_df.columns)

        return {
            "download_url": download_url,
            "preview": preview,
            "columns_after": columns_after
        }

    except Exception as e:
        return {"status": "error", "message": str(e)}
    

@router.post("/train-and-predict", tags=["Machine Learning"])
async def train_and_predict_endpoint(
    data_file: UploadFile,
    metadata: str = Form(...)
):
    metadata_dict = json.loads(metadata)

    evaluation, predictions, output_path, visual_path = train_and_predict(data_file, metadata_dict)

    return {
        "modeling": evaluation,
        "predictions": predictions,
        "download_predict_url": output_path,
        "download_visual_url": visual_path
    }