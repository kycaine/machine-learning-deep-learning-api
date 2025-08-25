import os

from fastapi import APIRouter
from fastapi.responses import FileResponse

from app.config.constants import *

router = APIRouter()

@router.get("/download/{file_path:path}", tags=["Download"])
async def download_file(file_path: str):
    abs_path = os.path.abspath(file_path)
    
    if os.path.exists(abs_path) and abs_path.startswith(os.path.abspath("outputs")):
        return FileResponse(abs_path, filename=file_path.split("/")[-1])
    else:
        return {"error": FILE_NOT_FOUND_OR_INVALID_PATH}
