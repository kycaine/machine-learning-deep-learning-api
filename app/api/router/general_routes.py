import os
import json
import tempfile

from fastapi import APIRouter
from fastapi.responses import FileResponse
from app.machine_learning.preprocessing import clean_data
from app.api.schemas.request import GeneralRequest
from fastapi import APIRouter, File, Form, UploadFile
from app.config.file_manager import *
from fastapi.responses import JSONResponse

from app.config.constants import *

router = APIRouter()

@router.post("/clean", tags=["General"])
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

@router.get("/download/{file_path:path}", tags=["General"])
async def download_file(file_path: str):
    abs_path = os.path.abspath(file_path)
    
    if os.path.exists(abs_path) and abs_path.startswith(os.path.abspath("outputs")):
        return FileResponse(abs_path, filename=file_path.split("/")[-1])
    else:
        return {"error": FILE_NOT_FOUND_OR_INVALID_PATH}
