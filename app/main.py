from fastapi import FastAPI

from app.api.router import download_routes, machine_learning_routes, deep_learning_routes
from app.config.init_folders import create_required_folders

create_required_folders()

app = FastAPI(
    title="ML-DL General API",
    description="API untuk data cleaning, EDA, feature engineering, modeling, dll",
    version="0.1.0"
)

app.include_router(machine_learning_routes.router)
app.include_router(deep_learning_routes.router)
app.include_router(download_routes.router)
