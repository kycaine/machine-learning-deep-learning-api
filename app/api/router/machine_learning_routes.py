import datetime
import json
import os
import tempfile
from pathlib import Path

from fastapi import APIRouter, File, Form, UploadFile
from fastapi.responses import JSONResponse

from app.api.schemas.general_request import GeneralRequest
from app.config.paths import *
from app.machine_learning.eda import generate_eda
from app.machine_learning.preprocessing import clean_data_strict

router = APIRouter()

@router.post("/clean")
async def clean_data_api(
    file: UploadFile = File(..., description="CSV or XLSX file to clean"),
    request_str: str = Form(..., description='JSON string, e.g. {"target_column":"price","task_type":"regression","feature_columns":["size","rooms","location"],"columns":{"price":"float","size":"float","rooms":"int","location":"str"}}')
):
    request_data = json.loads(request_str)
    request_obj = GeneralRequest(**request_data)

    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    cleaned_path, summary = clean_data_strict(tmp_path, request_obj.columns, original_filename=file.filename)

    return JSONResponse(content={
        "download_url": OUTPUTS_CLEANED + "/" + os.path.basename(cleaned_path),
        "summary": summary
    })


@router.post("/eda")
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
